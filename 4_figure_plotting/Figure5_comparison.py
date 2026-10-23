"""
Figure 5 comparison: Feature importance across two scenarios
Row 1: Pure endmember (classification)
Row 2: Measured mixture (regression)

Columns: Late Succ., Early Succ., DkCy, Lichen, Moss, Bare Substrate
"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


def rsquared(x, y):
    """Coefficient of determination R² = 1 - SSE/SST, with x = predicted
    and y = observed. Same definition as Figure3.py / Figure4.py."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sse = np.sum((y - x) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return 1.0 - sse / sst


def class_recall(summary_path):
    """Per-class recall parsed from rf_pure_endmember_classification_summary.txt,
    as {'6class': {class: recall}, '4class': {class: recall}}."""
    out, scheme = {}, None
    for line in summary_path.read_text().splitlines():
        if "-class accuracy:" in line:
            scheme = line.split("-class")[0].strip() + "class"
            out[scheme] = {}
            continue
        parts = line.split()
        # classification-report rows: name precision recall f1 support
        if scheme and len(parts) == 5 and "." in parts[1] and "." in parts[2]:
            out[scheme][parts[0]] = float(parts[2])
    return out


# Target name -> class name in the pure-endmember classification summary
PURE_CLASS = {'frac_late_successional': 'Late_Successional', 'frac_early_successional': 'Early_Successional',
              'frac_DkCy': 'DkCy', 'frac_Lichen': 'Lichen', 'frac_Moss': 'Moss', 'frac_Bare': 'Bare'}


def main():
    root = Path(__file__).resolve().parent
    rf_dir = root.parent.parent / "re_do_analysis" / "RF" / "2_results"
    out_dir = root / "0_exported_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load feature importance data (v2 re-do, with bare substrate) ---
    # Row 1: Pure endmember (classification)
    df_pure_4 = pd.read_csv(rf_dir / "rf_feature_importance_pure_endmember_4class.csv")
    df_pure_6 = pd.read_csv(rf_dir / "rf_feature_importance_pure_endmember_6class.csv")

    # Row 2: Measured mixture (regression) — reflectance only feature importance
    df_meas_4 = pd.read_csv(rf_dir / "rf_separate_4class_refl_feature_imp.csv")
    df_meas_6 = pd.read_csv(rf_dir / "rf_separate_6class_refl_feature_imp.csv")

    # --- Wavelengths ---
    band_cols_4 = [c for c in df_meas_4.columns if c != 'target_name']
    band_cols_6 = [c for c in df_meas_6.columns if c != 'target_name']
    band_cols_pure = [c for c in df_pure_4.columns if c != 'target_name']

    wl_4 = np.array([float(c) for c in band_cols_4])
    wl_6 = np.array([float(c) for c in band_cols_6])
    wl_pure = np.array([float(c) for c in band_cols_pure])

    bad_bands = [[1320, 1440], [1770, 1960]]

    # --- Panel definitions ---
    # (title, target_name_4class, target_name_6class, color)
    panels = [
        ('Late Succ.\n(DkCy+Lichen+Moss)', 'frac_late_successional', None, '#377eb8'),
        ('Early Succ.\n(LtCy)', 'frac_early_successional', None, '#e41a1c'),
        ('DkCy', None, 'frac_DkCy', '#4daf4a'),
        ('Lichen', None, 'frac_Lichen', '#00bfc4'),
        ('Moss', None, 'frac_Moss', '#984ea3'),
        ('Bare Substrate\n(Rock+Soil)', 'frac_Bare', None, '#636363'),
    ]

    row_labels = ['Pure Endmember\n(Classification)', 'Measured Mixture\n(Regression)']

    # Per-class metrics for annotation on each panel
    # Row 0: per-class accuracy (recall) read from
    #        rf_pure_endmember_classification_summary.txt
    #        (4-class: Late/Early/Bare; 6-class: DkCy/Lichen/Moss)
    recall = class_recall(rf_dir / "rf_pure_endmember_classification_summary.txt")
    panel_metrics = {}
    for col_i, (_, t4, t6, _) in enumerate(panels):
        scheme, target = ('4class', t4) if t4 else ('6class', t6)
        panel_metrics[(0, col_i)] = f'Accu = {recall[scheme][PURE_CLASS[target]] * 100:.0f}%'

    # Row 1: R² range (min ~ max) across the four RF-based configurations
    #        (RF and PCA+RF, each with reflectance only and reflectance +
    #        indices), computed from the same prediction CSVs and the same
    #        rsquared() as Figure 3 (4-class: Late/Early/Bare) and Figure 4
    #        (6-class: DkCy/Lichen/Moss), so the numbers match those figures.
    configs = ['rf_refl', 'rf_refl_indices', 'pca_rf_refl', 'pca_rf_refl_indices']
    pred_4 = {c: pd.read_csv(rf_dir / f"rf_4class_{c}_predictions.csv") for c in configs}
    pred_6 = {c: pd.read_csv(rf_dir / f"rf_6class_{c}_predictions.csv") for c in configs}
    for col_i, (_, t4, t6, _) in enumerate(panels):
        target = t4 if t4 else t6
        preds = pred_4 if t4 else pred_6
        r2_vals = sorted(
            rsquared(df[f'pred_{target}'], df[target]) for df in preds.values()
        )
        panel_metrics[(1, col_i)] = f'R² = {r2_vals[0]:.2f} ~ {r2_vals[-1]:.2f}'

    # --- Figure: 2 rows x 6 cols ---
    n_cols = len(panels)
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(2, n_cols, hspace=0.2, wspace=0.25)

    for col_i, (title, t4, t6, color) in enumerate(panels):
        for row_i in range(2):
            ax = fig.add_subplot(gs[row_i, col_i])

            if row_i == 0:
                # Pure endmember
                if t4:
                    df_src = df_pure_4
                else:
                    df_src = df_pure_6
                bcols = band_cols_pure
                wl = wl_pure
            else:
                # Measured mixture
                if t4:
                    df_src = df_meas_4
                    bcols = band_cols_4
                    wl = wl_4
                else:
                    df_src = df_meas_6
                    bcols = band_cols_6
                    wl = wl_6

            # Get target name
            target = t4 if t4 else t6

            # For pure endmember, target_name is 'all_4class' or 'all_6class'
            if row_i == 0:
                if t4:
                    subset = df_src[df_src['target_name'] == 'all_4class']
                else:
                    subset = df_src[df_src['target_name'] == 'all_6class']
            else:
                subset = df_src[df_src['target_name'] == target]

            imp_vals = subset[bcols].values.astype(float)
            mean_imp = imp_vals.mean(axis=0)
            std_imp = imp_vals.std(axis=0)

            # Mask bad bands
            mask = np.zeros(len(wl), dtype=bool)
            for lo, hi in bad_bands:
                mask |= (wl >= lo) & (wl <= hi)
            mean_imp[mask] = np.nan
            std_imp[mask] = np.nan

            # Insert NaN at gaps for clean breaks
            wl_plot = wl.copy()
            mean_plot = mean_imp.copy()
            std_plot = std_imp.copy()
            for lo, hi in bad_bands:
                before = np.where(wl_plot < lo)[0]
                after = np.where(wl_plot > hi)[0]
                if len(before) > 0 and len(after) > 0:
                    insert_idx = before[-1] + 1
                    wl_plot = np.insert(wl_plot, insert_idx, [lo, hi])
                    mean_plot = np.insert(mean_plot, insert_idx, [np.nan, np.nan])
                    std_plot = np.insert(std_plot, insert_idx, [np.nan, np.nan])

            # Plot
            ax.fill_between(wl_plot, mean_plot - std_plot, mean_plot + std_plot,
                            color=color, alpha=0.2, zorder=3)
            ax.plot(wl_plot, mean_plot, color=color, linewidth=0.6, zorder=4)

            # Bad band shading
            for lo, hi in bad_bands:
                ax.axvspan(lo, hi, color='lightgray', alpha=0.2, zorder=0)

            ax.set_xlim(400, 2400)
            in_range = (wl >= 400) & (wl <= 2400) & ~np.isnan(mean_imp)
            if in_range.any():
                y_max = (mean_imp[in_range] + std_imp[in_range]).max()
                ax.set_ylim(0, y_max * 1.05)

            ax.tick_params(labelsize=6)

            # Per-class metric annotation
            metric_text = panel_metrics.get((row_i, col_i))
            if metric_text:
                ax.text(0.97, 0.95, metric_text, transform=ax.transAxes,
                        fontsize=10, va='top', ha='right', fontstyle='italic',
                        color='k')

            # Column titles (top row only)
            if row_i == 0:
                ax.set_title(title, fontsize=9, fontweight='bold')

            # Row labels (first column only)
            if col_i == 0:
                ax.set_ylabel(row_labels[row_i], fontsize=8, fontweight='bold')

            # X labels (bottom row only)
            if row_i == 1:
                ax.set_xlabel('Wavelength (nm)', fontsize=8, fontweight='bold')
            else:
                ax.set_xticklabels([])

    # --- Save ---
    out_file = out_dir / 'Figure5_comparison.png'
    fig.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()
