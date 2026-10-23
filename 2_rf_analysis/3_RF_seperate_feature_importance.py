"""
Per-class RF regression and feature importance for biocrust fractional cover estimation
(v2: with bare substrate).

Runs 4 configurations sequentially:
  - 2 grouping schemes: 4-class (successional + bare) and 6-class (individual BFTs; only the
                        targets not already covered by 4-class)
  - 2 predictor sets:   reflectance only and reflectance + 10 spectral indices

For each configuration, a separate RF model is trained per target class (rather than
multi-output) so that per-class RF feature importance can be extracted. Same nested-CV
protocol as 2_RF_analysis.py (outer 123 / inner 42 / 60 draws).

Outputs (per configuration): predictions CSV and feature-importance CSV in ../2_results/.
Usage: python 3_RF_seperate_feature_importance.py [--n-iter 60] [--out-dir ../2_results]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


# Parallelism: N_WORKERS single-threaded forests in parallel (see 2_RF_analysis.py); no effect on results.
N_WORKERS = 20

# ---- Column definitions ----
ORIG_TARGET_COLS = ["frac_Vegetation", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss", "frac_Bare"]
INDEX_COLS = ["brightness_index", "NDVI", "PRI", "NDNI", "NDWI", "MCARI", "CI", "SCAI", "BSCI"]   # 9 indices
SCHEME_TARGETS = {
    "4class": ["frac_Vegetation", "frac_late_successional", "frac_early_successional", "frac_Bare"],
    "6class": ["frac_DkCy", "frac_Lichen", "frac_Moss"],   # Veg, LtCy, Bare already covered by 4class
}


def load_data(csv_path, scheme, predictor_set):
    """
    scheme:        '4class' (Veg, late_successional, early_successional, Bare)
                   '6class' (DkCy, Lichen, Moss only — Veg/LtCy/Bare already covered by 4class)
    predictor_set: 'refl'         — reflectance bands only
                   'refl_indices' — reflectance bands + spectral indices
    """
    df = pd.read_csv(csv_path)
    df["frac_late_successional"] = df["frac_DkCy"] + df["frac_Lichen"] + df["frac_Moss"]
    df["frac_early_successional"] = df["frac_LtCy"]
    target_cols = SCHEME_TARGETS[scheme]

    # Remove bad bands by wavelength name
    bad_bands = [[1320, 1440], [1770, 1960]]
    exclude_wvl = set()
    for lo, hi in bad_bands:
        for w in range(lo, hi + 1):
            exclude_wvl.add(str(w))

    # Build feature column list
    derived_targets = {"frac_late_successional", "frac_early_successional"}
    excluded = set(ORIG_TARGET_COLS) | derived_targets | {"full"}
    if predictor_set == "refl":
        excluded |= set(INDEX_COLS)
    elif predictor_set != "refl_indices":
        raise ValueError(f"Unknown predictor_set: {predictor_set}")

    feature_cols = [c for c in df.columns
                    if c not in excluded and c not in exclude_wvl]
    assert not any(c.startswith("frac_") for c in feature_cols), "a target leaked into the features"

    y = df[target_cols].values
    X = df[feature_cols].values
    plots = df[["full"]].values

    n_idx = sum(1 for c in feature_cols if c in INDEX_COLS)
    n_bands = len(feature_cols) - n_idx
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features ({n_bands} bands + {n_idx} indices).")
    print(f"Targets ({len(target_cols)}): {target_cols}")
    return X, y, plots, feature_cols, target_cols


def run_config(csv_path, out_dir, scheme, predictor_set, n_iter=60):
    """Run per-target nested CV and save predictions + feature importance."""
    tag = f"{scheme}_{predictor_set}"
    out_csv = out_dir / f"rf_separate_{tag}_predictions.csv"
    feature_imp_csv = out_dir / f"rf_separate_{tag}_feature_imp.csv"

    print("\n" + "#" * 70)
    print(f"# Config: scheme={scheme}, predictors={predictor_set}")
    print("#" * 70)

    X, y, plots, feature_cols, target_cols = load_data(csv_path, scheme, predictor_set)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    param_dist = {
        "n_estimators": [300, 500, 800, 1000, 1500],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.3, 0.5, 0.7, 1.0],
        "bootstrap": [True, False],
    }

    df_out = None
    feature_imp = None

    for target_idx, target_name in enumerate(target_cols):
        print(f"\n{'='*60}")
        print(f"Target: {target_name}")
        print(f"{'='*60}")
        y_target = y[:, target_idx]

        all_true = []
        all_pred = []
        all_plot = []
        fold_importances = []

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled), 1):
            print(f"  Fold {fold} ...")
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_target[train_idx], y_target[test_idx]
            plot_test = plots[test_idx]

            base_rf = RandomForestRegressor(random_state=42, n_jobs=1)
            rf_search = RandomizedSearchCV(
                estimator=base_rf,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="r2",
                cv=inner_cv,
                verbose=0,
                random_state=42,
                n_jobs=N_WORKERS)

            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
            y_pred = best_rf.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"    R² = {r2:.3f} | RMSE = {rmse:.4f}")

            all_true.append(y_test)
            all_pred.append(y_pred)
            all_plot.append(plot_test)
            fold_importances.append(best_rf.feature_importances_)

        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        all_plot = np.vstack(all_plot).reshape(-1, 1)

        # Overall performance
        r2_all = r2_score(all_true, all_pred)
        rmse_all = np.sqrt(mean_squared_error(all_true, all_pred))
        print(f"  Overall: R² = {r2_all:.3f} | RMSE = {rmse_all:.4f}")

        df_true = pd.DataFrame(all_true, columns=[target_name])
        df_pred = pd.DataFrame(all_pred, columns=[f"pred_{target_name}"])
        df_plot = pd.DataFrame(all_plot, columns=["full"])
        df_final = pd.concat([df_plot, df_true, df_pred], axis=1)

        fold_imp_df = pd.DataFrame(np.vstack(fold_importances))
        fold_imp_df.columns = feature_cols
        fold_imp_df["target_name"] = target_name

        if df_out is None:
            df_out = df_final
            feature_imp = fold_imp_df
        else:
            df_out = pd.merge(df_out, df_final, how='left', on='full')
            feature_imp = pd.concat([feature_imp, fold_imp_df], axis=0)

    df_out.to_csv(out_csv, index=False)
    feature_imp.to_csv(feature_imp_csv, index=False)

    print(f"\nSaved predictions: {out_csv}")
    print(f"Saved feature importance: {feature_imp_csv}")


# ******************************************************************#
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--predictor-sets", nargs="+", default=["refl", "refl_indices"], choices=["refl", "refl_indices"])
    ap.add_argument("--out-dir", default=str(script_dir.parent / "2_results"))
    args = ap.parse_args()
    csv_path = str(script_dir.parent / "1_data" / "Processed_data" / "measured_mixtures_with_indices.csv")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schemes = ["4class", "6class"]
    predictor_sets = args.predictor_sets

    for scheme in schemes:
        for pset in predictor_sets:
            run_config(csv_path, out_dir, scheme, pset, n_iter=args.n_iter)

    print("\nAll configurations finished.")
