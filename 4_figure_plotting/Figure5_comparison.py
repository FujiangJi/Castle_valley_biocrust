"""
Figure 5 comparison: Feature importance across two scenarios
Row 1: Pure endmember (classification)
Row 2: Measured mixture (regression)

Columns: Late Succ., Early Succ., DkCy, Lichen, Moss
"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def main():
    root = Path(__file__).resolve().parent
    rf_dir = root.parent.parent / "4_Castle_valley_analysis_RF" / "2_results"
    out_dir = root / "0_exported_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load feature importance data ---
    # Row 1: Pure endmember (classification)
    df_pure_3 = pd.read_csv(rf_dir / "rf_feature_importance_pure_endmember_3class.csv")
    df_pure_6 = pd.read_csv(rf_dir / "rf_feature_importance_pure_endmember_6class.csv")

    # Row 2: Measured mixture (regression) — reflectance only feature importance
    df_meas_3 = pd.read_csv(rf_dir / "rf_separate_3class_refl_feature_imp.csv")
    df_meas_5 = pd.read_csv(rf_dir / "rf_separate_5class_refl_feature_imp.csv")

    # --- Wavelengths ---
    band_cols_3 = [c for c in df_meas_3.columns if c != 'target_name']
    band_cols_5 = [c for c in df_meas_5.columns if c != 'target_name']
    band_cols_pure = [c for c in df_pure_3.columns if c != 'target_name']

    wl_3 = np.array([float(c) for c in band_cols_3])
    wl_5 = np.array([float(c) for c in band_cols_5])
    wl_pure = np.array([float(c) for c in band_cols_pure])

    bad_bands = [[1320, 1440], [1770, 1960]]

    # --- Panel definitions ---
    # (title, target_name_3class, target_name_5class, color)
    panels = [
        ('Late Succ.\n(DkCy+Lichen+Moss)', 'frac_late_successional', None, '#377eb8'),
        ('Early Succ.\n(LtCy)', 'frac_early_successional', None, '#e41a1c'),
        ('DkCy', None, 'frac_DkCy', '#4daf4a'),
        ('Lichen', None, 'frac_Lichen', '#00bfc4'),
        ('Moss', None, 'frac_Moss', '#984ea3'),
    ]

    row_labels = ['Pure Endmember\n(Classification)', 'Measured Mixture\n(Regression)']

    # Per-class metrics for annotation on each panel
    # Row 0: per-class accuracy (recall) from classification report
    # Row 1: R² from measured mixture (refl only / refl+indices, matching Figure 3 & 4)
    panel_metrics = {
        # Row 0: Pure endmember accuracy (recall)
        (0, 0): 'Accu = 93%',   # Late Succ (3-class)
        (0, 1): 'Accu = 89%',   # Early Succ (3-class)
        (0, 2): 'Accu = 69%',   # DkCy (6-class)
        (0, 3): 'Accu = 56%',   # Lichen (6-class)
        (0, 4): 'Accu = 67%',   # Moss (6-class)
        # Row 1: R² (refl only / refl+indices) — from new RF outputs
        (1, 0): 'R² = 0.38 ~ 0.48',   # Late Succ
        (1, 1): 'R² = 0.50 ~ 0.52',   # Early Succ
        (1, 2): 'R² = 0.19 ~ 0.28',   # DkCy
        (1, 3): 'R² = 0.14 ~ 0.17',   # Lichen
        (1, 4): 'R² = 0.18 ~ 0.34',   # Moss
    }

    # --- Figure: 2 rows x 5 cols ---
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(2, 5, hspace=0.2, wspace=0.25)

    for col_i, (title, t3, t5, color) in enumerate(panels):
        for row_i in range(2):
            ax = fig.add_subplot(gs[row_i, col_i])

            if row_i == 0:
                # Pure endmember
                if t3:
                    df_src = df_pure_3
                else:
                    df_src = df_pure_6
                bcols = band_cols_pure
                wl = wl_pure
            else:
                # Measured mixture
                if t3:
                    df_src = df_meas_3
                    bcols = band_cols_3
                    wl = wl_3
                else:
                    df_src = df_meas_5
                    bcols = band_cols_5
                    wl = wl_5

            # Get target name
            target = t3 if t3 else t5

            # For pure endmember, target_name is 'all_3class' or 'all_6class'
            if row_i == 0:
                if t3:
                    subset = df_src[df_src['target_name'] == 'all_3class']
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
