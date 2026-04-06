import sys
import numpy as np
import pandas as pd
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

#******************************************************************#
csv_path = "../1_data/Processed_data/measured_mixtures.csv"
out_csv = "../2_results/pca_rf_reflectance_only_5class.csv"
log_path = "../2_results/pca_rf_reflectance_only_5class.log"

log_file = open(log_path, "w")
sys.stdout = Tee(sys.__stdout__, log_file)

# Original 6 targets — frac_Litter and frac_Vegetation will be merged
orig_target_cols = ["frac_Litter", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss", "frac_Vegetation"]
# Final 5 targets after merging
target_cols = ["frac_Litter+Vegetation", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss"]

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

    # Merge frac_Litter + frac_Vegetation into one column
    df["frac_Litter+Vegetation"] = df["frac_Litter"] + df["frac_Vegetation"]

    y = df[target_cols].values
    feature_cols = [c for c in df.columns
                    if c not in orig_target_cols
                    and c not in target_cols
                    and c != "full"]
    X = df[feature_cols].values
    plots = df[["full"]].values

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} spectral bands.")
    print(f"Targets ({len(target_cols)}): {target_cols}")
    return X, y, plots, feature_cols

#*********************************************************************************************************#
X, y, plots, feature_cols = load_data(csv_path)

# ===================== v4: PCA + RF =====================
# Changes from v3:
#   - Added PCA between StandardScaler and RF
#   - PCA n_components is tuned as a hyperparameter
#   - Reduces ~1839 correlated spectral bands to fewer components
# Changes from v2:
#   - Merged frac_Litter + frac_Vegetation → frac_Litter+Vegetation (5 targets)
#   - Proper nested CV and pipeline (same as v2/v3)
# ========================================================

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
])

param_dist = {
    "pca__n_components": [10, 20, 30, 40, 60, 80],
    "rf__n_estimators": [300, 500, 800, 1000, 1500],
    "rf__max_depth": [None, 10, 15, 20, 30],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", 0.3, 0.5, 0.7, 1.0],
    "rf__bootstrap": [True, False],
}

outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n>>> Starting nested cross-validation (PCA + RF) ...")

all_true = []
all_pred = []
all_plot = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=60,
        scoring="r2",
        cv=inner_cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)

    best = search.best_params_
    print(f"\n===== Outer Fold {fold} =====")
    print(f"  Best inner R² = {search.best_score_:.3f}")
    print(f"  PCA={best['pca__n_components']}  n_est={best['rf__n_estimators']}  "
          f"depth={best['rf__max_depth']}  max_feat={best['rf__max_features']}")
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

df_true = pd.DataFrame(all_true, columns=target_cols)
df_pred = pd.DataFrame(all_pred, columns=[f"pred_{c}" for c in target_cols])
df_plot = pd.DataFrame(all_plot, columns=["full"])
df_out = pd.concat([df_plot, df_true, df_pred], axis=1)
df_out.to_csv(out_csv, index=False)
print(f"\nSaved prediction CSV: {out_csv}")

log_file.close()
sys.stdout = sys.__stdout__
