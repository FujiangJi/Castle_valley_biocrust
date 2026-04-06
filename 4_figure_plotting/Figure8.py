"""
Figure 8: Stacked bar charts — 5 rows (B1-B5) x 4 columns
Col 1: Measured successional stages
Col 2: Estimated successional stages
Col 3: Measured BFT components
Col 4: Estimated BFT components
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def main():
    root = Path(__file__).resolve().parent
    cnn_dir = root.parent.parent / "4_Castle_valley_analysis_synthetic_approach" / "3_results"
    out_dir = root / "0_exported_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    treats = ["Control", "AltP", "Warmed", "AltP+Warmed"]
    plots_list = ["B1", "B2", "B3", "B4", "B5"]

    # --- Load and prepare successional data (v7, 3-class) ---
    target_cols_3 = ["frac_Litter+Vegetation", "frac_late_successional", "frac_early_successional"]
    pred_cols_3 = [f"pred_{c}" for c in target_cols_3]

    df_3class = pd.read_csv(cnn_dir / "final_model_estimation_3class.csv")
    df_3class["plot_key"] = df_3class["full"].apply(lambda x: "_".join(x.split("_")[:2]))

    df_cover1 = pd.concat([df_3class[["plot_key"]], df_3class[target_cols_3]], axis=1)
    df_cover2 = pd.concat([df_3class[["plot_key"]], df_3class[pred_cols_3]], axis=1)

    df_cover1 = df_cover1.groupby("plot_key").mean().reset_index()
    df_cover2 = df_cover2.groupby("plot_key").mean().reset_index()

    df_succ = pd.merge(df_cover1, df_cover2, how='left', on='plot_key')
    df_succ["plot"] = df_succ["plot_key"].apply(lambda x: x.split("_")[0])
    df_succ["treat"] = df_succ["plot_key"].apply(lambda x: x.split("_")[1])
    df_succ.loc[df_succ["treat"] == "CC", "Treatment"] = "Control"
    df_succ.loc[df_succ["treat"] == "LC", "Treatment"] = "Warmed"
    df_succ.loc[df_succ["treat"] == "LW", "Treatment"] = "AltP+Warmed"
    df_succ.loc[df_succ["treat"] == "CW", "Treatment"] = "AltP"

    # --- Load and prepare BFT data (v6, 5-class) ---
    target_cols_5 = ["frac_Litter+Vegetation", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss"]
    pred_cols_5 = [f"pred_{c}" for c in target_cols_5]

    df_5class = pd.read_csv(cnn_dir / "final_model_estimation_5class.csv")
    df_5class["plot_key"] = df_5class["full"].apply(lambda x: "_".join(x.split("_")[:2]))

    df_cover3 = pd.concat([df_5class[["plot_key"]], df_5class[target_cols_5]], axis=1)
    df_cover4 = pd.concat([df_5class[["plot_key"]], df_5class[pred_cols_5]], axis=1)

    df_cover3 = df_cover3.groupby("plot_key").mean().reset_index()
    df_cover4 = df_cover4.groupby("plot_key").mean().reset_index()

    df_bft = pd.merge(df_cover3, df_cover4, how='left', on='plot_key')
    df_bft["plot"] = df_bft["plot_key"].apply(lambda x: x.split("_")[0])
    df_bft["treat"] = df_bft["plot_key"].apply(lambda x: x.split("_")[1])
    df_bft.loc[df_bft["treat"] == "CC", "Treatment"] = "Control"
    df_bft.loc[df_bft["treat"] == "LC", "Treatment"] = "Warmed"
    df_bft.loc[df_bft["treat"] == "LW", "Treatment"] = "AltP+Warmed"
    df_bft.loc[df_bft["treat"] == "CW", "Treatment"] = "AltP"

    # --- Colors and labels ---
    succ_cols_meas = ["frac_late_successional", "frac_early_successional"]
    succ_cols_pred = ["pred_frac_late_successional", "pred_frac_early_successional"]
    succ_colors = {"frac_late_successional": "#4575b4", "frac_early_successional": "#fc8d59",
                   "pred_frac_late_successional": "#4575b4", "pred_frac_early_successional": "#fc8d59"}
    succ_labels = {"frac_late_successional": "Late Succ.", "frac_early_successional": "Early Succ.",
                   "pred_frac_late_successional": "Late Succ.", "pred_frac_early_successional": "Early Succ."}

    bft_cols_meas = ["frac_Moss", "frac_Lichen", "frac_DkCy", "frac_LtCy"]
    bft_cols_pred = ["pred_frac_Moss", "pred_frac_Lichen", "pred_frac_DkCy", "pred_frac_LtCy"]
    bft_colors = {"frac_LtCy": "#e41a1c", "frac_DkCy": "#4daf4a", "frac_Lichen": "#00bfc4", "frac_Moss": "#984ea3",
                  "pred_frac_LtCy": "#e41a1c", "pred_frac_DkCy": "#4daf4a", "pred_frac_Lichen": "#00bfc4", "pred_frac_Moss": "#984ea3"}
    bft_labels = {"frac_LtCy": "LtCy", "frac_DkCy": "DkCy", "frac_Lichen": "Lichen", "frac_Moss": "Moss",
                  "pred_frac_LtCy": "LtCy", "pred_frac_DkCy": "DkCy", "pred_frac_Lichen": "Lichen", "pred_frac_Moss": "Moss"}

    col_titles = [
        "a. Measured successional stages",
        "b. Estimated successional stages",
        "c. Measured BFT components",
        "d. Estimated BFT components",
    ]

    # --- Figure: 5 rows x 4 cols ---
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(5, 4, left=0, right=1, top=1, bottom=0, hspace=0.1, wspace=0.14)

    for row_i, p in enumerate(plots_list):
        # Column configs: (dataframe, func_cols, colors_dict, labels_dict)
        col_configs = [
            (df_succ, succ_cols_meas, succ_colors, succ_labels),
            (df_succ, succ_cols_pred, succ_colors, succ_labels),
            (df_bft, bft_cols_meas, bft_colors, bft_labels),
            (df_bft, bft_cols_pred, bft_colors, bft_labels),
        ]

        for col_i, (df_src, func_cols, colors, labels) in enumerate(col_configs):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor((0, 0, 0, 0.01))

            df_temp = df_src[df_src["plot"] == p]
            agg = (df_temp.groupby("Treatment")[func_cols].mean()
                   .reindex(treats) * 100)

            x = np.arange(len(treats))
            bar_width = 0.6
            bottom = np.zeros(len(treats))
            for col in func_cols:
                vals = agg[col].fillna(0).values
                ax.bar(x, vals, width=bar_width, bottom=bottom,
                       color=colors[col], edgecolor="none",
                       label=labels[col] if row_i == 0 else None)
                bottom += vals

            ax.set_xticks(x)
            if row_i == 4:
                treat_labels = ["Control", "AltP", "Warmed", "AltP+\nWarmed"]
                ax.set_xticklabels(treat_labels, fontsize=10, rotation=0, ha='center')
                ax.set_xlabel('Treatments', fontsize=9, fontweight='bold', labelpad=0)
            else:
                ax.set_xticklabels([])

            if col_i == 0:
                ax.set_ylabel('Frac. Cover (%)', fontsize=10, fontweight='bold')
                # Block label (B1-B5) on the left, no rotation, bigger
                ax.text(-0.28, 0.5, p, transform=ax.transAxes,
                        fontsize=14, fontweight='bold', rotation=0,
                        va='center', ha='center')
            else:
                ax.set_ylabel('')

            ax.tick_params(labelsize=9)
            ax.grid(axis="y", linestyle="-", linewidth=0.3, alpha=0.4)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Column titles on first row
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=10.5, fontweight='bold', loc='left')

    # --- Figure-level legends below the plot ---
    # Successional legend between col 1 and col 2
    succ_handles = [
        plt.Rectangle((0, 0), 1, 1, fc='#4575b4', ec='none', label='Late Succ. (DkCy+Lichen+Moss)'),
        plt.Rectangle((0, 0), 1, 1, fc='#fc8d59', ec='none', label='Early Succ. (LtCy)'),
    ]
    fig.legend(handles=succ_handles, loc='lower center', ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.25, -0.12))

    # BFT legend between col 3 and col 4
    bft_handles = [
        plt.Rectangle((0, 0), 1, 1, fc='#984ea3', ec='none', label='Moss'),
        plt.Rectangle((0, 0), 1, 1, fc='#00bfc4', ec='none', label='Lichen'),
        plt.Rectangle((0, 0), 1, 1, fc='#4daf4a', ec='none', label='DkCy'),
        plt.Rectangle((0, 0), 1, 1, fc='#e41a1c', ec='none', label='LtCy'),
    ]
    fig.legend(handles=bft_handles, loc='lower center', ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.75, -0.12))

    # --- Save ---
    out_file = out_dir / 'Figure8.png'
    fig.savefig(out_file, dpi=500, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()
