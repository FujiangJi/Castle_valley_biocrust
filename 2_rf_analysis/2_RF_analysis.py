"""
Combined RF/PCA+RF analysis for biocrust fractional cover estimation (v2: with bare substrate).

Runs 8 configurations sequentially:
  - 2 grouping schemes: 4-class (successional + bare) and 6-class (individual BFTs + bare)
  - 2 models:           RF only and PCA + RF
  - 2 predictor sets:   reflectance only and reflectance + indices

Protocol: nested CV — outer KFold(5, shuffle, 123) for evaluation, inner KFold(5, shuffle, 42)
with RandomizedSearchCV (60 draws) for hyperparameter selection. Outer test folds are predicted
once and never used for selection. The same outer folds are used by the synthetic-CNN branch.

Outputs: per-config prediction CSV and log file in ../2_results/ (override with --out-dir).
Usage: python 2_RF_analysis.py [--n-iter 60] [--schemes 4class 6class] [--out-dir ../2_results]
"""
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class Tee:
    """Write to both console and file."""
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
    def flush(self):
        for f in self.files:
            f.flush()


# Parallelism: the search runs N_WORKERS single-threaded forests in parallel. (Forests with n_jobs=-1
# inside a parallel search oversubscribe the CPU ~30x and starve everything else on the machine.
# n_jobs has no effect on results: RF randomness is fixed by random_state.) 20 of the 32 cores: the
# machine is shared and the CNN branch needs a core or two.
N_WORKERS = 20

# ---- Column definitions ----
ORIG_TARGET_COLS = ["frac_Vegetation", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss", "frac_Bare"]
DERIVED_TARGET_COLS = ["frac_late_successional", "frac_early_successional"]
INDEX_COLS = ["brightness_index", "NDVI", "PRI", "NDNI", "NDWI", "MCARI", "CI", "SCAI", "BSCI"]   # 9 indices
SCHEME_TARGETS = {
    "4class": ["frac_Vegetation", "frac_late_successional", "frac_early_successional", "frac_Bare"],
    "6class": ["frac_Vegetation", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss", "frac_Bare"],
}


def load_data(csv_path, scheme, predictor_set):
    """
    scheme:        '4class' (Veg, late_successional, early_successional, Bare)
                   '6class' (Veg, DkCy, Lichen, LtCy, Moss, Bare)
    predictor_set: 'refl'         — reflectance bands only
                   'refl_indices' — reflectance bands + spectral indices
    """
    df = pd.read_csv(csv_path)
    df["frac_late_successional"] = df["frac_DkCy"] + df["frac_Lichen"] + df["frac_Moss"]
    df["frac_early_successional"] = df["frac_LtCy"]
    target_cols = SCHEME_TARGETS[scheme]

    # Remove bad bands by wavelength name
    bad_bands = [[1320, 1440], [1770, 1960]]
    exclude_wvl = {str(w) for lo, hi in bad_bands for w in range(lo, hi + 1)}

    # Build feature column list: every target column (original or derived) is excluded
    excluded = set(ORIG_TARGET_COLS + DERIVED_TARGET_COLS + ["full"])
    if predictor_set == "refl":
        excluded |= set(INDEX_COLS)
    elif predictor_set != "refl_indices":
        raise ValueError(f"Unknown predictor_set: {predictor_set}")
    feature_cols = [c for c in df.columns if c not in excluded and c not in exclude_wvl]
    assert not any(c.startswith("frac_") for c in feature_cols), "a target leaked into the features"

    y = df[target_cols].values
    X = df[feature_cols].values
    plots = df[["full"]].values

    n_idx = sum(1 for c in feature_cols if c in INDEX_COLS)
    n_bands = len(feature_cols) - n_idx
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features ({n_bands} bands + {n_idx} indices).")
    print(f"Targets ({len(target_cols)}): {target_cols}")
    return X, y, plots, feature_cols, target_cols


def build_pipeline(use_pca):
    """Pipeline: StandardScaler [-> PCA] -> RandomForestRegressor."""
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA()))
    steps.append(("rf", RandomForestRegressor(random_state=42, n_jobs=1)))
    return Pipeline(steps)


def build_param_dist(use_pca):
    """Hyperparameter search space."""
    rf_params = {
        "rf__n_estimators": [300, 500, 800, 1000, 1500],
        "rf__max_depth": [None, 10, 15, 20, 30],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", 0.3, 0.5, 0.7, 1.0],
        "rf__bootstrap": [True, False],
    }
    if use_pca:
        rf_params["pca__n_components"] = [10, 20, 30, 40, 60, 80]
    return rf_params


def run_config(csv_path, out_dir, scheme, model, predictor_set, n_iter=60):
    """Run nested CV for one configuration and save predictions + log."""
    use_pca = (model == "pca_rf")

    tag = f"{scheme}_{model}_{predictor_set}"
    out_csv = out_dir / f"rf_{tag}_predictions.csv"
    log_path = out_dir / f"rf_{tag}.log"

    log_file = open(log_path, "w")
    saved_stdout = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log_file)

    try:
        print("\n" + "#" * 70)
        print(f"# Config: scheme={scheme}, model={model}, predictors={predictor_set}, n_iter={n_iter}")
        print("#" * 70)

        X, y, plots, feature_cols, target_cols = load_data(csv_path, scheme, predictor_set)

        pipe = build_pipeline(use_pca)
        param_dist = build_param_dist(use_pca)

        outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        print(f"\n>>> Starting nested cross-validation ({'PCA + RF' if use_pca else 'RF, no PCA'}) ...")

        all_true, all_pred, all_plot = [], [], []
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            search = RandomizedSearchCV(estimator=pipe, param_distributions=param_dist, n_iter=n_iter,
                                        scoring="r2", cv=inner_cv, random_state=42, n_jobs=N_WORKERS, verbose=0)
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)

            best = search.best_params_
            print(f"\n===== Outer Fold {fold} =====")
            print(f"  Best inner R² = {search.best_score_:.3f}")
            pca_txt = f"PCA={best['pca__n_components']}  " if use_pca else ""
            print(f"  {pca_txt}n_est={best['rf__n_estimators']}  depth={best['rf__max_depth']}  "
                  f"max_feat={best['rf__max_features']}")
            for i, comp in enumerate(target_cols):
                r2 = r2_score(y_test[:, i], y_pred[:, i])
                rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                print(f"  {comp:25s} | R² = {r2:.3f} | RMSE = {rmse:.4f}")

            all_true.append(y_test)
            all_pred.append(y_pred)
            all_plot.append(plots[test_idx])

        all_true = np.vstack(all_true)
        all_pred = np.vstack(all_pred)
        all_plot = np.vstack(all_plot)

        print("\n\n=========== Overall Cross-validated Performance ===========")
        for i, comp in enumerate(target_cols):
            r2 = r2_score(all_true[:, i], all_pred[:, i])
            rmse = np.sqrt(mean_squared_error(all_true[:, i], all_pred[:, i]))
            print(f"  {comp:25s} | R² = {r2:.3f} | RMSE = {rmse:.4f}")

        df_out = pd.concat([pd.DataFrame(all_plot, columns=["full"]),
                            pd.DataFrame(all_true, columns=target_cols),
                            pd.DataFrame(all_pred, columns=[f"pred_{c}" for c in target_cols])], axis=1)
        df_out.to_csv(out_csv, index=False)
        print(f"\nSaved prediction CSV: {out_csv}")
    finally:
        sys.stdout = saved_stdout
        log_file.close()


# ******************************************************************#
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--schemes", nargs="+", default=["4class", "6class"], choices=sorted(SCHEME_TARGETS))
    ap.add_argument("--predictor-sets", nargs="+", default=["refl", "refl_indices"], choices=["refl", "refl_indices"])
    ap.add_argument("--out-dir", default=str(script_dir.parent / "2_results"))
    args = ap.parse_args()

    csv_path = str(script_dir.parent / "1_data" / "Processed_data" / "measured_mixtures_with_indices.csv")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scheme in args.schemes:
        for model in ["rf", "pca_rf"]:
            for pset in args.predictor_sets:
                run_config(csv_path, out_dir, scheme, model, pset, n_iter=args.n_iter)

    print("\nAll configurations finished.")
