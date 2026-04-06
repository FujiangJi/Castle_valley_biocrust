"""
Figure 7: Combined figure per block (5-class: DkCy, Lichen, LtCy, Moss)
Top: 2x2 heatmaps (4 treatments, measured vs estimated for 4 targets)
Bottom: 1x4 residual boxplots
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats as sp_stats


def main():
    root = Path(__file__).resolve().parent
    cnn_dir = root.parent.parent / "4_Castle_valley_analysis_synthetic_approach" / "3_results"
    out_dir = root / "0_exported_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cnn_dir / "final_model_estimation_5class.csv")
    df["plot"] = df["full"].apply(lambda x: x.split("_")[0])
    df["treat"] = df["full"].apply(lambda x: x.split("_")[1])
    df["Spectra"] = df["full"].apply(lambda x: x.split("_")[2])

    df.loc[df["treat"] == "CC", "Treatment"] = "Control"
    df.loc[df["treat"] == "LC", "Treatment"] = "Warmed"
    df.loc[df["treat"] == "LW", "Treatment"] = "AltP+Warmed"
    df.loc[df["treat"] == "CW", "Treatment"] = "AltP"

    target_cols = ["frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss"]
    pred_cols = [f"pred_{c}" for c in target_cols]
    heatmap_labels = ["DkCy", "Lichen", "LtCy", "Moss"]
    treats = ["Control", "AltP", "Warmed", "AltP+Warmed"]
    plots_list = ["B1", "B2", "B3", "B4", "B5"]
    target_colors = ['#238b45', '#1f78b4', '#e6550d', '#6a51a3']  # Green, Blue, Orange, Purple
    cmaps_col = ['Greens', 'Blues', 'Oranges', 'Purples']

    for p in plots_list:
        # --- Combined figure: heatmaps (top) + residuals (bottom) ---
        fig = plt.figure(figsize=(12, 8))
        outer = gridspec.GridSpec(2, 1, height_ratios=[2, 0.8], left=0, right=1, top=1, bottom=0, hspace=0.2)

        # Panel A title
        fig.text(-0.04, 1.08, f'a. Measured vs Estimated fractional cover ({p})',
                 fontsize=12, fontweight='bold', va='top', ha='left')

        # === Top: 2x2 heatmap grid ===
        gs_heat = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0],
                                                    hspace=0.2, wspace=0.12)

        for t_i, tr in enumerate(treats):
            row_t, col_t = divmod(t_i, 2)
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 4, subplot_spec=gs_heat[row_t, col_t],
                wspace=0.1, hspace=0.15)

            df_temp = df[(df["plot"] == p) & (df["Treatment"] == tr)].copy()
            nrows_grid, ncols_grid = 4, 3

            if len(df_temp) > 0:
                df_temp["row"] = (df_temp["Spectra"].astype(int) - 1) // ncols_grid
                df_temp["col"] = (df_temp["Spectra"].astype(int) - 1) % ncols_grid

            all_cols = target_cols + pred_cols
            labels2 = ["Measured\nfraction", "Estimated\nfraction"]

            k = 0
            first_ax_pos = None
            for i in range(2):
                for j in range(4):
                    ax = fig.add_subplot(inner[i, j])
                    if k == 0:
                        first_ax_pos = ax.get_position()
                    t_c = all_cols[k]
                    grid = np.full((nrows_grid, ncols_grid), np.nan)
                    if len(df_temp) > 0:
                        grid[df_temp["row"].values, df_temp["col"].values] = df_temp[t_c].values

                    t_idx = j
                    if len(df_temp) > 0:
                        vmin = min(df_temp[target_cols[t_idx]].min(),
                                   df_temp[pred_cols[t_idx]].min())
                        vmax = max(df_temp[target_cols[t_idx]].max(),
                                   df_temp[pred_cols[t_idx]].max())
                    else:
                        vmin, vmax = 0, 1

                    # Convert to percentage for annotation
                    grid_pct = grid * 100
                    sns.heatmap(grid_pct, annot=True, cmap=cmaps_col[j], fmt=".0f",
                                cbar=True, linewidths=0.5, linecolor="white",
                                yticklabels=False, xticklabels=False,
                                vmin=vmin * 100, vmax=vmax * 100,
                                annot_kws={"size": 7, "weight": "bold"},
                                cbar_kws={"shrink": 0.7, "pad": 0.02,
                                          "label": "Fraction (%)"},
                                ax=ax)

                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=6, length=2, pad=1)
                    cbar.ax.yaxis.label.set_size(7)
                    cbar.ax.yaxis.label.set_weight("bold")
                    cbar.ax.yaxis.labelpad = 2

                    if i == 0:
                        ax.set_title(heatmap_labels[j], fontsize=8, fontweight='bold')
                    if j == 0:
                        ax.set_ylabel(labels2[i], fontsize=8, fontweight='bold',
                                      fontstyle='italic')
                    k += 1

            # Subtitle
            sub_label = f'(a.{t_i+1}) {p}: {tr}'
            fig.text(first_ax_pos.x0 - 0.03, first_ax_pos.y1 + 0.03,
                     sub_label, fontsize=10, fontweight='bold',
                     va='bottom', ha='left')

        # Panel B title
        fig.text(-0.04, 0.32, f'b. Prediction residuals ({p})',
                 fontsize=12, fontweight='bold', va='top', ha='left')

        # === Bottom: 1x4 residual boxplots ===
        gs_box = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1],
                                                   wspace=0.25)

        df_plot = df[df["plot"] == p]

        for t_idx, (tcol, pcol, label) in enumerate(zip(target_cols, pred_cols, heatmap_labels)):
            ax = fig.add_subplot(gs_box[0, t_idx])

            residual_data = []
            for tr in treats:
                sub = df_plot[df_plot["Treatment"] == tr]
                residuals = (sub[pcol].values - sub[tcol].values) * 100
                residual_data.append(residuals)

            bp = ax.boxplot(residual_data, patch_artist=True, widths=0.5,
                            medianprops=dict(color='black', linewidth=1.5),
                            whiskerprops=dict(linewidth=1.0),
                            capprops=dict(linewidth=1.0),
                            flierprops=dict(marker='o', markersize=3, alpha=0.5))

            for patch in bp['boxes']:
                patch.set_facecolor(target_colors[t_idx])
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')

            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            ax.set_xticklabels(treats, fontsize=9, rotation=0, ha='center')
            ax.set_ylabel('Residual (%)', fontsize=10, fontweight='bold')
            ax.set_xlabel('Treatments', fontsize=10, fontweight='bold')
            ax.set_title(f'(b.{t_idx+1}) {label}', fontsize=10, fontweight='bold', loc='left')
            ax.tick_params(labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Expand ylim for annotation space
            yl = ax.get_ylim()
            margin = (yl[1] - yl[0]) * 0.05
            ax.set_ylim(yl[0] - margin * 0.3, yl[1] + margin)

            # Per-treatment bias annotation
            for i, tr in enumerate(treats):
                sub = df_plot[df_plot["Treatment"] == tr]
                bias = (sub[pcol].values - sub[tcol].values).mean() * 100
                ax.text(i + 1, ax.get_ylim()[1] * 0.95,
                        f'{bias:+.1f}%',
                        fontsize=8, fontweight='bold',
                        ha='center', va='top', color=target_colors[t_idx])

            # Overall R²/MAE for this target across all treatments
            x_all = df_plot[pcol].values * 100
            y_all = df_plot[tcol].values * 100
            _, _, r_all, _, _ = sp_stats.linregress(x_all, y_all)
            r2_all = r_all**2
            mae_all = np.abs(x_all - y_all).mean()
            ax.text(0.97, 0.05,
                    f'Overall: R²={r2_all:.2f}, MAE={mae_all:.1f}%',
                    fontsize=8, fontweight='bold', transform=ax.transAxes,
                    ha='right', va='bottom', color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # --- Save ---
        out_file = out_dir / f'Figure7_{p}.png'
        fig.savefig(out_file, dpi=500, bbox_inches='tight', pad_inches=0.15,
                    facecolor='white')
        plt.close(fig)
        print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
