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
    # Col 1: RF (v5) — reflectance only vs reflectance+indices
    df_rf_refl = pd.read_csv(rf_dir / "rf_reflectance_only_3class.csv")
    df_rf_idx = pd.read_csv(rf_dir / "rf_reflectance_indices_3class.csv")

    # Col 2: PCA+RF (v6) — reflectance only vs reflectance+indices
    df_pca_refl = pd.read_csv(rf_dir / "pca_rf_reflectance_only_3class.csv")
    df_pca_idx = pd.read_csv(rf_dir / "pca_rf_reflectance_indices_3class.csv")

    # Col 3: Pretrained 1D-CNN
    df_pretrained = pd.read_csv(cnn_dir / "pretrained_model_estimation_3class.csv")

    # Col 4: Fine-tuned 1D-CNN
    df_finetuned = pd.read_csv(cnn_dir / "final_model_estimation_3class.csv")

    # --- Targets ---
    target_cols = ['frac_late_successional', 'frac_early_successional',
                   'frac_Litter+Vegetation']
    pred_cols = [f"pred_{x}" for x in target_cols]
    row_names = ['Late Successional', 'Early Successional', 'Litter + Vegetation']

    col_titles = ['RF', 'PCA + RF', 'Pretrained 1D-CNN', 'Fine-tuned 1D-CNN']

    # Colors for comparison
    color_refl = 'dodgerblue'    # bright blue
    color_idx = 'orangered'

    # --- Figure: 3 rows x 4 cols ---
    fig = plt.figure(figsize=(10, 6.5))
    gs = GridSpec(3, 4, left=0, right=1, top=1, bottom=0,
                  wspace=0.2, hspace=0.15)

    for row_i, (tcol, pcol, rname) in enumerate(zip(target_cols, pred_cols, row_names)):
        for col_i in range(4):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor((0, 0, 0, 0.01))
            ax.grid(color='gray', linestyle=':', linewidth=0.2)

            # 1:1 line
            ax.plot((0, 1), (0, 1), transform=ax.transAxes,
                    ls='--', c='k', lw=1, zorder=0)

            if col_i == 0:
                # RF: compare reflectance vs reflectance+indices
                x_refl = df_rf_refl[pcol] * 100
                y_refl = df_rf_refl[tcol] * 100
                x_idx = df_rf_idx[pcol] * 100
                y_idx = df_rf_idx[tcol] * 100

                sns.regplot(x=x_refl, y=y_refl, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': color_refl, 'lw': 1.5})
                sns.regplot(x=x_idx, y=y_idx, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': color_idx, 'lw': 1.5})
                ax.scatter(x_refl, y_refl, color=color_refl, s=20, alpha=0.6,
                           label='Refl. only', zorder=2)
                ax.scatter(x_idx, y_idx, color=color_idx, s=15, alpha=0.5,marker="s",
                           label='Refl. + Indices', zorder=2)

                max_v = max(x_refl.max(), y_refl.max(), x_idx.max(), y_idx.max())
                ax.set_xlim(0, max_v)
                ax.set_ylim(0, max_v)

                r2_r, p_r = rsquared(x_refl, y_refl)
                mae_r = (abs(x_refl - y_refl)).mean()
                r2_i, p_i = rsquared(x_idx, y_idx)
                mae_i = (abs(x_idx - y_idx)).mean()

                ax.text(0.03, 0.97,
                        f'$\\mathbf{{R^2}}$={r2_r:.3f}, MAE={mae_r:.1f}%',
                        fontsize=9, transform=ax.transAxes,
                        va='top', ha='left', color=color_refl,
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                ax.text(0.03, 0.85,
                        f'$\\mathbf{{R^2}}$={r2_i:.3f}, MAE={mae_i:.1f}%',
                        fontsize=9, transform=ax.transAxes,
                        va='top', ha='left', color=color_idx,
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                if row_i == 0:
                    leg = ax.legend(fontsize=9, loc='lower right', frameon=True,
                                    markerscale=1.1, handletextpad=0.1, labelspacing=0.3,
                                    facecolor='white', framealpha=0.6, edgecolor='none')
                    for text, c in zip(leg.get_texts(),
                                       [color_refl, color_idx]):
                        text.set_color(c)

            elif col_i == 1:
                # PCA+RF: compare reflectance vs reflectance+indices
                x_refl = df_pca_refl[pcol] * 100
                y_refl = df_pca_refl[tcol] * 100
                x_idx = df_pca_idx[pcol] * 100
                y_idx = df_pca_idx[tcol] * 100

                sns.regplot(x=x_refl, y=y_refl, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': color_refl, 'lw': 1.5})
                sns.regplot(x=x_idx, y=y_idx, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': color_idx, 'lw': 1.5})
                ax.scatter(x_refl, y_refl, color=color_refl, s=20, alpha=0.6,
                           label='Refl. only', zorder=2)
                ax.scatter(x_idx, y_idx, color=color_idx, s=15, alpha=0.5, marker="s",
                           label='Refl. + Indices', zorder=2)

                max_v = max(x_refl.max(), y_refl.max(), x_idx.max(), y_idx.max())
                ax.set_xlim(0, max_v)
                ax.set_ylim(0, max_v)

                r2_r, p_r = rsquared(x_refl, y_refl)
                mae_r = (abs(x_refl - y_refl)).mean()
                r2_i, p_i = rsquared(x_idx, y_idx)
                mae_i = (abs(x_idx - y_idx)).mean()

                ax.text(0.03, 0.97,
                        f'$\\mathbf{{R^2}}$={r2_r:.3f}, MAE={mae_r:.1f}%',
                        fontsize=9, transform=ax.transAxes,
                        va='top', ha='left', color=color_refl,
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                ax.text(0.03, 0.85,
                        f'$\\mathbf{{R^2}}$={r2_i:.3f}, MAE={mae_i:.1f}%',
                        fontsize=9, transform=ax.transAxes,
                        va='top', ha='left', color=color_idx,
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                if row_i == 0:
                    leg = ax.legend(fontsize=9, loc='lower right', frameon=True,
                                    markerscale=1.1, handletextpad=0.1, labelspacing=0.3,
                                    facecolor='white', framealpha=0.6, edgecolor='none')
                    for text, c in zip(leg.get_texts(),
                                       [color_refl, color_idx]):
                        text.set_color(c)

            elif col_i == 2:
                # Pretrained 1D-CNN (hexbin for synthetic data — many points)
                x = df_pretrained[pcol] * 100
                y = df_pretrained[tcol] * 100
                min_v = min(x.min(), y.min())
                max_v = max(x.max(), y.max())

                ax.hexbin(x, y, gridsize=50, cmap='winter', bins='log',
                          mincnt=1, linewidths=0.2, alpha=0.6, zorder=1)
                ax.set_xlim(min_v, max_v)
                ax.set_ylim(min_v, max_v)

                r2, p = rsquared(x, y)
                ax.text(0.03, 0.97,
                        f'$\\mathbf{{R^2}}$={r2:.3f}\n$\\mathbf{{p}}$={p:.3f}',
                        fontsize=10, transform=ax.transAxes,
                        va='top', ha='left',
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            elif col_i == 3:
                # Fine-tuned 1D-CNN (scatter)
                x = df_finetuned[pcol] * 100
                y = df_finetuned[tcol] * 100
                max_v = max(x.max(), y.max())

                sns.regplot(x=x, y=y, ax=ax, fit_reg=True, ci=95,
                            scatter=False, line_kws={'color': 'k', 'lw': 1.5})
                ax.scatter(x, y, color='blue', s=20, zorder=2, alpha=0.5)
                ax.set_xlim(0, max_v)
                ax.set_ylim(0, max_v)

                r2, p = rsquared(x, y)
                mae = (abs(x - y)).mean()
                ax.text(0.03, 0.97,
                        f'$\\mathbf{{R^2}}$={r2:.3f}\n$\\mathbf{{MAE}}$={mae:.1f}%\n$\\mathbf{{p}}$={p:.3f}',
                        fontsize=10, transform=ax.transAxes,
                        va='top', ha='left',
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            ax.set_xlabel('', fontsize=10, fontweight='bold')
            ax.set_ylabel('', fontsize=10, fontweight='bold')

            # Labels
            if row_i == 2:
                ax.set_xlabel('Predicted fraction (%)', fontsize=11,
                              fontweight='bold')
            if col_i == 0:
                ax.set_ylabel('Observed fraction (%)', fontsize=11,
                              fontweight='bold')

            ax.tick_params(labelsize=10)

            # Column titles on top row
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=12, fontweight='bold')

        # Row label on the right side of the last column
        ax.text(1.06, 0.5, row_names[row_i], transform=ax.transAxes,
                fontsize=12, fontweight='bold', rotation=-90,
                va='center', ha='left')

    # --- Save ---
    out_file = out_dir / 'Figure3.png'
    fig.savefig(out_file, dpi=500, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()
