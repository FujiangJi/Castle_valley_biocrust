from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2, p_value


def main():
    root = Path(__file__).resolve().parent
    base = root.parent.parent
    rf_dir = base / "4_Castle_valley_analysis_RF" / "2_results"
    cnn_dir = base / "4_Castle_valley_analysis_synthetic_approach" / "3_results"
    out_dir = root / "0_exported_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load all CSVs ---
    # Row 1: RF (v3)
    df_rf_refl = pd.read_csv(rf_dir / "rf_reflectance_only_5class.csv")
    df_rf_idx = pd.read_csv(rf_dir / "rf_reflectance_indices_5class.csv")

    # Row 2: PCA+RF (v4)
    df_pca_refl = pd.read_csv(rf_dir / "pca_rf_reflectance_only_5class.csv")
    df_pca_idx = pd.read_csv(rf_dir / "pca_rf_reflectance_indices_5class.csv")

    # Row 3: Pretrained 1D-CNN (v6)
    df_pretrained = pd.read_csv(cnn_dir / "pretrained_model_estimation_5class.csv")

    # Row 4: Fine-tuned 1D-CNN (v6)
    df_finetuned = pd.read_csv(cnn_dir / "final_model_estimation_5class.csv")

    # --- 5 targets (columns) ---
    target_cols = ['frac_LtCy', 'frac_DkCy', 'frac_Lichen',
                   'frac_Moss', 'frac_Litter+Vegetation']
    pred_cols = [f"pred_{x}" for x in target_cols]
    col_names = ['LtCy', 'DkCy', 'Lichen', 'Moss', 'Litter + Vegetation']

    # 4 model rows
    row_titles = ['RF', 'PCA + RF', 'Pretrained 1D-CNN', 'Fine-tuned 1D-CNN']

    # Colors for comparison in RF / PCA+RF rows
    color_refl = 'dodgerblue'
    color_idx = 'orangered'

    # --- Figure: 4 rows x 5 cols ---
    fig = plt.figure(figsize=(10, 7), dpi=300)
    gs = GridSpec(4, 5, left=0, right=1, top=1, bottom=0,
                  wspace=0.2, hspace=0.2)

    for row_i in range(4):
        for col_i in range(5):
            tcol = target_cols[col_i]
            pcol = pred_cols[col_i]

            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor((0, 0, 0, 0.01))
            ax.grid(color='gray', linestyle=':', linewidth=0.2)

            # 1:1 line
            ax.plot((0, 1), (0, 1), transform=ax.transAxes,
                    ls='--', c='k', lw=1, zorder=0)

            if row_i == 0:
                # RF: reflectance vs reflectance+indices
                x_refl = df_rf_refl[pcol] * 100
                y_refl = df_rf_refl[tcol] * 100
                x_idx = df_rf_idx[pcol] * 100
                y_idx = df_rf_idx[tcol] * 100

                sns.regplot(x=x_refl, y=y_refl, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': color_refl, 'lw': 1.5})
                sns.regplot(x=x_idx, y=y_idx, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': color_idx, 'lw': 1.5})
                ax.scatter(x_refl, y_refl, color=color_refl, s=15, alpha=0.6,
                           label='Refl. only', zorder=2)
                ax.scatter(x_idx, y_idx, color=color_idx, s=12, alpha=0.5,
                           marker='s', label='Refl. + Indices', zorder=2)

                max_v = max(x_refl.max(), y_refl.max(), x_idx.max(), y_idx.max())
                ax.set_xlim(0, max_v)
                ax.set_ylim(0, max_v)

                r2_r, _ = rsquared(x_refl, y_refl)
                mae_r = (abs(x_refl - y_refl)).mean()
                r2_i, _ = rsquared(x_idx, y_idx)
                mae_i = (abs(x_idx - y_idx)).mean()

                ax.text(0.035, 0.96,
                        f'$\\mathbf{{R^2}}$={r2_r:.3f}, MAE={mae_r:.1f}%',
                        fontsize=7, transform=ax.transAxes,
                        va='top', ha='left', color=color_refl, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                ax.text(0.035, 0.84,
                        f'$\\mathbf{{R^2}}$={r2_i:.3f}, MAE={mae_i:.1f}%',
                        fontsize=7, transform=ax.transAxes,
                        va='top', ha='left', color=color_idx, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                if col_i == 1:
                    leg = ax.legend(fontsize=7, loc='lower right', frameon=True,
                                    markerscale=1.0, handletextpad=0.1, labelspacing=0.3,
                                    facecolor='white', framealpha=0.6, edgecolor='none')
                    for text, c in zip(leg.get_texts(), [color_refl, color_idx]):
                        text.set_color(c)

            elif row_i == 1:
                # PCA+RF: reflectance vs reflectance+indices
                x_refl = df_pca_refl[pcol] * 100
                y_refl = df_pca_refl[tcol] * 100
                x_idx = df_pca_idx[pcol] * 100
                y_idx = df_pca_idx[tcol] * 100

                sns.regplot(x=x_refl, y=y_refl, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': color_refl, 'lw': 1.5})
                sns.regplot(x=x_idx, y=y_idx, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': color_idx, 'lw': 1.5})
                ax.scatter(x_refl, y_refl, color=color_refl, s=15, alpha=0.6,
                           label='Refl. only', zorder=2)
                ax.scatter(x_idx, y_idx, color=color_idx, s=12, alpha=0.5,
                           marker='s', label='Refl. + Indices', zorder=2)

                max_v = max(x_refl.max(), y_refl.max(), x_idx.max(), y_idx.max())
                ax.set_xlim(0, max_v)
                ax.set_ylim(0, max_v)

                r2_r, _ = rsquared(x_refl, y_refl)
                mae_r = (abs(x_refl - y_refl)).mean()
                r2_i, _ = rsquared(x_idx, y_idx)
                mae_i = (abs(x_idx - y_idx)).mean()

                ax.text(0.035, 0.96,
                        f'$\\mathbf{{R^2}}$={r2_r:.3f}, MAE={mae_r:.1f}%',
                        fontsize=7, transform=ax.transAxes,
                        va='top', ha='left', color=color_refl, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                ax.text(0.035, 0.84,
                        f'$\\mathbf{{R^2}}$={r2_i:.3f}, MAE={mae_i:.1f}%',
                        fontsize=7, transform=ax.transAxes,
                        va='top', ha='left', color=color_idx, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                if col_i == 1:
                    leg = ax.legend(fontsize=7, loc='lower right', frameon=True,
                                    markerscale=1.0, handletextpad=0.1, labelspacing=0.3,
                                    facecolor='white', framealpha=0.6, edgecolor='none')
                    for text, c in zip(leg.get_texts(), [color_refl, color_idx]):
                        text.set_color(c)

            elif row_i == 2:
                # Pretrained 1D-CNN (hexbin)
                x = df_pretrained[pcol] * 100
                y = df_pretrained[tcol] * 100
                min_v = min(x.min(), y.min())
                max_v = max(x.max(), y.max())

                ax.hexbin(x, y, gridsize=50, cmap='winter', bins='log',
                          mincnt=1, linewidths=0.2, alpha=0.6, zorder=1)
                ax.set_xlim(min_v, max_v)
                ax.set_ylim(min_v, max_v)

                r2, p = rsquared(x, y)
                ax.text(0.035, 0.96,
                        f'$\\mathbf{{R^2}}$={r2:.3f}\n$\\mathbf{{p}}$={p:.3f}',
                        fontsize=8, transform=ax.transAxes,
                        va='top', ha='left', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            elif row_i == 3:
                # Fine-tuned 1D-CNN (scatter)
                x = df_finetuned[pcol] * 100
                y = df_finetuned[tcol] * 100
                max_v = max(x.max(), y.max())

                sns.regplot(x=x, y=y, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': 'k', 'lw': 1.5})
                ax.scatter(x, y, color='blue', s=15, zorder=2, alpha=0.5)
                ax.set_xlim(0, max_v)
                ax.set_ylim(0, max_v)

                r2, p = rsquared(x, y)
                mae = (abs(x - y)).mean()
                ax.text(0.035, 0.96,
                        f'$\\mathbf{{R^2}}$={r2:.3f}\n$\\mathbf{{MAE}}$={mae:.1f}%\n$\\mathbf{{p}}$={p:.3f}',
                        fontsize=8, transform=ax.transAxes,
                        va='top', ha='left', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            # Axis labels
            ax.set_xlabel('', fontsize=10, fontweight='bold')
            ax.set_ylabel('', fontsize=10, fontweight='bold')
            if row_i == 3:
                ax.set_xlabel('Predicted (%)', fontsize=10, fontweight='bold')
            if col_i == 0:
                ax.set_ylabel('Observed (%)', fontsize=10, fontweight='bold')

            ax.tick_params(labelsize=9)

            # Column titles on top row
            if row_i == 0:
                ax.set_title(col_names[col_i], fontsize=11, fontweight='bold')

        # Row label on right side of last column
        ax.text(1.06, 0.5, row_titles[row_i], transform=ax.transAxes,
                fontsize=11, fontweight='bold', rotation=-90,
                va='center', ha='left')

    # --- Save ---
    out_file = out_dir / 'Figure4.png'
    fig.savefig(out_file, dpi=500, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()
