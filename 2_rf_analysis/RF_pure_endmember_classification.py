"""
RF classification of pure endmember spectra (no mixing).
Quantifies how separable biocrust functional types are from their spectral signatures alone.
Outputs: confusion matrix, per-band feature importance, classification accuracy (.txt summary).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def parse_component(name):
    parts = str(name).split('_')
    if len(parts) >= 3:
        return parts[2]
    return None


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "1_data" / "ASD_Specra"
    out_dir = script_dir.parent / "2_results"

    # --- Load pure endmember spectra ---
    df = pd.read_csv(data_dir / "ASD_All_Spectra_ContactProbe.csv")
    id_col = df.columns[0]
    band_cols = df.columns[1:]

    df['component'] = df[id_col].astype(str).apply(parse_component)
    df.loc[df['component'] == 'BRT', 'component'] = 'Vegetation'
    df.loc[df['component'] == 'DCY', 'component'] = 'DkCy'
    df.loc[df['component'] == 'LCN', 'component'] = 'Lichen'
    df.loc[df['component'] == 'LCY', 'component'] = 'LtCy'
    df.loc[df['component'] == 'LTR', 'component'] = 'Litter'
    df.loc[df['component'] == 'MSS', 'component'] = 'Moss'
    df.loc[df['component'] == 'ROCK', 'component'] = 'Rock'
    df.loc[df['component'] == 'SOIL', 'component'] = 'Soil'

    # Exclude Rock and Soil
    df = df[~df['component'].isin(['Rock', 'Soil'])]

    # Remove bad bands
    bad_bands = [[1320, 1440], [1770, 1960]]
    target_wvl = np.arange(350, 2501, 1)
    exclude_indices = []
    for band_range in bad_bands:
        indices = np.where((target_wvl >= band_range[0]) & (target_wvl <= band_range[1]))[0]
        exclude_indices.extend(indices)

    all_band_cols = list(band_cols)
    valid_band_cols = [c for i, c in enumerate(all_band_cols) if i not in exclude_indices]
    wavelengths = np.array([float(c) for c in valid_band_cols])

    X = df[valid_band_cols].values.astype(float)
    y_labels = df['component'].values

    print(f"Loaded {X.shape[0]} pure endmember spectra, {X.shape[1]} bands")
    print(f"Samples per class: {dict(zip(*np.unique(y_labels, return_counts=True)))}")

    # --- Standardize ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # =====================================================================
    # 6-class classification
    # =====================================================================
    print("\n" + "=" * 60)
    print("6-class classification")
    print("=" * 60)

    le = LabelEncoder()
    y6 = le.fit_transform(y_labels)
    classes6 = le.classes_

    all_y6_true, all_y6_pred = [], []
    fold_imp_6 = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled, y6), 1):
        print(f"  Fold {fold} ...", end=" ")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y6[train_idx], y6[test_idx]

        rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc_fold = accuracy_score(y_test, y_pred)
        print(f"Acc = {acc_fold:.3f}")

        all_y6_true.extend(y_test)
        all_y6_pred.extend(y_pred)
        fold_imp_6.append(rf.feature_importances_)

    acc6 = accuracy_score(all_y6_true, all_y6_pred)
    cm6 = confusion_matrix(all_y6_true, all_y6_pred)
    report6 = classification_report(all_y6_true, all_y6_pred, target_names=classes6)

    print(f"\n  Overall 6-class accuracy: {acc6:.3f}")
    print(report6)

    # =====================================================================
    # 3-class classification
    # =====================================================================
    print("=" * 60)
    print("3-class classification")
    print("=" * 60)

    y_3class = y_labels.copy()
    y_3class = np.where(np.isin(y_labels, ['DkCy', 'Lichen', 'Moss']),
                        'Late_Successional', y_3class)
    y_3class = np.where(y_labels == 'LtCy', 'Early_Successional', y_3class)
    y_3class = np.where(np.isin(y_labels, ['Vegetation', 'Litter']),
                        'Litter+Vegetation', y_3class)

    le3 = LabelEncoder()
    y3 = le3.fit_transform(y_3class)
    classes3 = le3.classes_

    all_y3_true, all_y3_pred = [], []
    fold_imp_3 = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled, y3), 1):
        print(f"  Fold {fold} ...", end=" ")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y3[train_idx], y3[test_idx]

        rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc_fold = accuracy_score(y_test, y_pred)
        print(f"Acc = {acc_fold:.3f}")

        all_y3_true.extend(y_test)
        all_y3_pred.extend(y_pred)
        fold_imp_3.append(rf.feature_importances_)

    acc3 = accuracy_score(all_y3_true, all_y3_pred)
    cm3 = confusion_matrix(all_y3_true, all_y3_pred)
    report3 = classification_report(all_y3_true, all_y3_pred, target_names=classes3)

    print(f"\n  Overall 3-class accuracy: {acc3:.3f}")
    print(report3)

    # =====================================================================
    # Save outputs
    # =====================================================================

    # Feature importance CSVs
    imp6_df = pd.DataFrame(np.vstack(fold_imp_6), columns=wavelengths.astype(int))
    imp6_df['target_name'] = 'all_6class'
    imp6_df.to_csv(out_dir / "rf_feature_importance_pure_endmember_6class.csv", index=False)

    imp3_df = pd.DataFrame(np.vstack(fold_imp_3), columns=wavelengths.astype(int))
    imp3_df['target_name'] = 'all_3class'
    imp3_df.to_csv(out_dir / "rf_feature_importance_pure_endmember_3class.csv", index=False)

    # Plain-text summary
    cm6_df = pd.DataFrame(cm6, index=classes6, columns=classes6)
    cm3_df = pd.DataFrame(cm3, index=classes3, columns=classes3)

    lines = []
    lines.append("Pure Endmember Classification Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total samples: {X.shape[0]}, Bands: {X.shape[1]}")
    lines.append(f"Samples per class: {dict(zip(*np.unique(y_labels, return_counts=True)))}")
    lines.append("")
    lines.append(f"6-class accuracy: {acc6:.3f}")
    lines.append(report6)
    lines.append("6-class Confusion Matrix:")
    lines.append(cm6_df.to_string())
    lines.append("")
    lines.append(f"3-class accuracy: {acc3:.3f}")
    lines.append(report3)
    lines.append("3-class Confusion Matrix:")
    lines.append(cm3_df.to_string())

    with open(out_dir / "rf_pure_endmember_classification_summary.txt", 'w') as f:
        f.write("\n".join(lines))

    print(f"\nSaved: rf_feature_importance_pure_endmember_6class.csv")
    print(f"Saved: rf_feature_importance_pure_endmember_3class.csv")
    print(f"Saved: rf_pure_endmember_classification_summary.txt")


if __name__ == '__main__':
    main()
