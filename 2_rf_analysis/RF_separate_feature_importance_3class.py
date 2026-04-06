import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---- 3 merged targets (matching v5) ----
orig_target_cols = ["frac_Litter", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss", "frac_Vegetation"]
target_cols = ["frac_Litter+Vegetation", "frac_late_successional", "frac_early_successional"]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    bad_bands = [[1320, 1440], [1770, 1960]]
    target_wvl = np.arange(350, 2501, 1)
    exclude_indices = []
    for band_range in bad_bands:
        indices = np.where((target_wvl >= band_range[0]) & (target_wvl <= band_range[1]))[0]
        exclude_indices.extend(indices)

    exclude_indices = np.array(exclude_indices)
    df = df.drop(df.columns[exclude_indices], axis=1)

    # Merge targets
    df["frac_Litter+Vegetation"] = df["frac_Litter"] + df["frac_Vegetation"]
    df["frac_late_successional"] = df["frac_DkCy"] + df["frac_Lichen"] + df["frac_Moss"]
    df["frac_early_successional"] = df["frac_LtCy"]

    y = df[target_cols].values
    feature_cols = [c for c in df.columns
                    if c not in orig_target_cols
                    and c not in target_cols
                    and c != "full"]
    X = df[feature_cols].values
    plots = df[["full"]].values

    feature_cols = [int(x) for x in feature_cols]

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} spectral bands.")
    print(f"Targets ({len(target_cols)}): {target_cols}")
    return X, y, plots, feature_cols


def make_regions(wavelengths, bin_width=50):
    wmin = int(np.floor(wavelengths.min() / bin_width) * bin_width)
    wmax = int(np.ceil(wavelengths.max() / bin_width) * bin_width)
    regions = {}
    for lo in range(wmin, wmax, bin_width):
        hi = lo + bin_width - 1
        mask = (wavelengths >= lo) & (wavelengths <= hi)
        if np.any(mask):
            regions[f"{lo}_{hi}nm"] = (lo, hi)
    return regions


def get_region_indices(wavelengths, regions, min_bands=10):
    region_indices = {}
    for region_name, (lo, hi) in regions.items():
        idx = np.where((wavelengths >= lo) & (wavelengths <= hi))[0]
        if len(idx) >= min_bands:
            region_indices[region_name] = idx
    return region_indices


def region_permutation_importance(model, X_test, y_test, region_indices,
                                  scoring_func=r2_score, n_repeats=10, random_state=42):
    rng = np.random.RandomState(random_state)
    y_pred = model.predict(X_test)
    baseline_score = scoring_func(y_test, y_pred)
    results = {}
    for region_name, cols in region_indices.items():
        scores_after_perm = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(X_test.shape[0])
            X_perm[:, cols] = X_perm[perm_idx][:, cols]
            y_perm_pred = model.predict(X_perm)
            perm_score = scoring_func(y_test, y_perm_pred)
            scores_after_perm.append(perm_score)
        scores_after_perm = np.array(scores_after_perm)
        results[region_name] = baseline_score - scores_after_perm.mean()
    return results


# ******************************************************************#
csv_path = "../1_data/Processed_data/measured_mixtures.csv"
out_csv = "../2_results/rf_feature_importance_3class_predictions.csv"
permu_imp_csv = "../2_results/rf_feature_importance_3class_permu_imp.csv"
feature_imp_csv = "../2_results/rf_feature_importance_3class_feature_imp.csv"

# ******************************************************************#
X, y, plots, feature_cols = load_data(csv_path)
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

var = True
for target_idx, target_name in enumerate(target_cols):
    print(f"\n{'='*60}")
    print(f"Target: {target_name}")
    print(f"{'='*60}")
    y_target = y[:, target_idx]

    all_true = []
    all_pred = []
    all_plot = []
    fold_importances = []

    fold = 1
    start_var = True
    for train_idx, test_idx in outer_cv.split(X_scaled):
        print(f"  Fold {fold} ...")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]
        plot_test = plots[test_idx]

        base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=60,
            scoring="r2",
            cv=inner_cv,
            verbose=0,
            random_state=42,
            n_jobs=-1)

        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        y_pred = best_rf.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"    R² = {r2:.3f} | RMSE = {rmse:.4f}")

        all_true.append(y_test)
        all_pred.append(y_pred)
        all_plot.append(plot_test)

        # Feature importance (from RF)
        fold_importances.append(best_rf.feature_importances_)

        # Regional permutation importance
        regions = make_regions(np.array(feature_cols), bin_width=50)
        region_indices = get_region_indices(np.array(feature_cols), regions)
        reg_imp = region_permutation_importance(
            model=best_rf,
            X_test=X_test,
            y_test=y_test,
            region_indices=region_indices,
            scoring_func=r2_score,
            n_repeats=10,
            random_state=42 + fold)
        reg_imp = pd.DataFrame(list(reg_imp.items()), columns=["region", f"importance_{fold}"])
        fold += 1
        if start_var:
            reg_imp_df = reg_imp
            start_var = False
        else:
            reg_imp_df = pd.merge(reg_imp_df, reg_imp, how='left', on='region')

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
    reg_imp_df["target_name"] = target_name

    fold_importances = pd.DataFrame(np.vstack(fold_importances))
    fold_importances.columns = feature_cols
    fold_importances["target_name"] = target_name

    if var:
        df_out = df_final
        permu_imp = reg_imp_df
        feature_imp = fold_importances
        var = False
    else:
        df_out = pd.merge(df_out, df_final, how='left', on='full')
        permu_imp = pd.concat([permu_imp, reg_imp_df], axis=0)
        feature_imp = pd.concat([feature_imp, fold_importances], axis=0)

df_out.to_csv(out_csv, index=False)
permu_imp.to_csv(permu_imp_csv, index=False)
feature_imp.to_csv(feature_imp_csv, index=False)

print(f"\nSaved predictions: {out_csv}")
print(f"Saved permutation importance: {permu_imp_csv}")
print(f"Saved feature importance: {feature_imp_csv}")
