"""
Figure 8 v2: Ecological shifts of biocrusts under treatments
Row 1 (a): Measured vs Estimated — successional stages (averaged across blocks, with block variability)
Row 2 (b): Measured vs Estimated — BFT components
Row 3 (c): Treatment effect (difference from Control) for successional stages
Row 4 (d): Treatment effect (difference from Control) for BFT components
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
    treat_labels = ["Control", "AltP", "Warmed", "AltP+Warmed"]

    # --- Load successional data (v7) ---
    target_cols_3 = ["frac_Litter+Vegetation", "frac_late_successional", "frac_early_successional"]
    pred_cols_3 = [f"pred_{c}" for c in target_cols_3]

    df_3class = pd.read_csv(cnn_dir / "final_model_estimation_3class.csv")
    df_3class["plot"] = df_3class["full"].apply(lambda x: x.split("_")[0])
    df_3class["treat"] = df_3class["full"].apply(lambda x: x.split("_")[1])
    df_3class.loc[df_3class["treat"] == "CC", "Treatment"] = "Control"
    df_3class.loc[df_3class["treat"] == "LC", "Treatment"] = "Warmed"
    df_3class.loc[df_3class["treat"] == "LW", "Treatment"] = "AltP+Warmed"
    df_3class.loc[df_3class["treat"] == "CW", "Treatment"] = "AltP"

    # --- Load BFT data (v6) ---
    target_cols_5 = ["frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss"]
    pred_cols_5 = [f"pred_{c}" for c in target_cols_5]

    df_5class = pd.read_csv(cnn_dir / "final_model_estimation_5class.csv")
    df_5class["plot"] = df_5class["full"].apply(lambda x: x.split("_")[0])
    df_5class["treat"] = df_5class["full"].apply(lambda x: x.split("_")[1])
    df_5class.loc[df_5class["treat"] == "CC", "Treatment"] = "Control"
    df_5class.loc[df_5class["treat"] == "LC", "Treatment"] = "Warmed"
    df_5class.loc[df_5class["treat"] == "LW", "Treatment"] = "AltP+Warmed"
    df_5class.loc[df_5class["treat"] == "CW", "Treatment"] = "AltP"

    # --- Compute per-block, per-treatment means ---
    # Successional: use only biocrust fractions (late + early)
    succ_targets = ["frac_late_successional", "frac_early_successional"]
    succ_preds = ["pred_frac_late_successional", "pred_frac_early_successional"]
    succ_labels_short = ["Late Succ. (DkCy+Lichen+Moss)", "Early Succ. (LtCy)"]
    succ_colors = ['#4575b4', '#fc8d59']

    bft_targets = target_cols_5
    bft_preds = pred_cols_5
    bft_labels_short = ["DkCy", "Lichen", "LtCy", "Moss"]
    bft_colors = ['#4daf4a', '#00bfc4', '#e41a1c', '#984ea3']

    # Aggregate: per block × treatment mean
    succ_meas = df_3class.groupby(["plot", "Treatment"])[succ_targets].mean().reset_index()
    succ_pred = df_3class.groupby(["plot", "Treatment"])[succ_preds].mean().reset_index()
    bft_meas = df_5class.groupby(["plot", "Treatment"])[bft_targets].mean().reset_index()
    bft_pred = df_5class.groupby(["plot", "Treatment"])[bft_preds].mean().reset_index()

    # --- Figure layout: 2 rows × 2 cols ---
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, left=0, right=1, top=1, bottom=0, hspace=0.28, wspace=0.2,
                           width_ratios=[1, 1.5])

    # ===================================================================
    # Panel (a): Measured vs Estimated — Successional stages
    # Paired bars per treatment, averaged across blocks with individual block dots
    # ===================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    from matplotlib.patches import Patch

    x = np.arange(len(treats))
    width = 0.16
    for s_i, (tcol, pcol, label, color) in enumerate(
            zip(succ_targets, succ_preds, succ_labels_short, succ_colors)):
        meas_means, meas_stds, pred_means, pred_stds = [], [], [], []
        meas_blocks, pred_blocks = [], []
        for tr in treats:
            m = succ_meas[succ_meas["Treatment"] == tr][tcol].values * 100
            p = succ_pred[succ_pred["Treatment"] == tr][pcol].values * 100
            meas_means.append(m.mean())
            meas_stds.append(m.std())
            pred_means.append(p.mean())
            pred_stds.append(p.std())
            meas_blocks.append(m)
            pred_blocks.append(p)

        # Tight grouping: meas and est side by side, then next component
        meas_pos = (-1.5 + s_i * 2) * width
        est_pos = (-0.5 + s_i * 2) * width

        # Measured bars
        ax_a.bar(x + meas_pos, meas_means, width,
                 color=color, alpha=0.9, edgecolor='none')
        ax_a.errorbar(x + meas_pos, meas_means, yerr=meas_stds,
                      fmt='none', ecolor='gray', elinewidth=0.8, capsize=2)

        # Estimated bars (hatched)
        ax_a.bar(x + est_pos, pred_means, width,
                 color=color, alpha=0.5, edgecolor=color, linewidth=1.0,
                 hatch='///')
        ax_a.errorbar(x + est_pos, pred_means, yerr=pred_stds,
                      fmt='none', ecolor='gray', elinewidth=0.8, capsize=2)

        # Block dots
        for t_i, tr in enumerate(treats):
            ax_a.scatter(np.full_like(meas_blocks[t_i], x[t_i] + meas_pos),
                         meas_blocks[t_i], color='black', s=8, zorder=5, alpha=0.6)
            ax_a.scatter(np.full_like(pred_blocks[t_i], x[t_i] + est_pos),
                         pred_blocks[t_i], color='black', s=8, zorder=5, alpha=0.6,
                         marker='x')

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(treat_labels, fontsize=10)
    ax_a.set_ylabel('Fractional Cover (%)', fontsize=11, fontweight='bold')
    ax_a.set_xlabel('Treatments', fontsize=11, fontweight='bold')
    ax_a.set_title('(a) Successional stages: Measured vs Estimated',
                    fontsize=11, fontweight='bold', loc='left')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.tick_params(labelsize=10)

    legend_a = [
        Patch(facecolor='gray', alpha=0.9, edgecolor='none', label='Measured (solid)'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='gray', hatch='///', label='Estimated (hatched)'),
        Patch(facecolor=succ_colors[0], edgecolor='none', label='Late Succ. (DkCy+Lichen+Moss)'),
        Patch(facecolor=succ_colors[1], edgecolor='none', label='Early Succ. (LtCy)'),
    ]
    ax_a.legend(handles=legend_a, fontsize=9, frameon=False, loc='upper left', ncol=1,
                 bbox_to_anchor=(0.07, 1.03))

    # ===================================================================
    # Panel (b): Measured vs Estimated — BFT components
    # ===================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    width_b = 0.09
    n_comp = len(bft_targets)

    for s_i, (tcol, pcol, label, color) in enumerate(
            zip(bft_targets, bft_preds, bft_labels_short, bft_colors)):
        meas_means, meas_stds, pred_means, pred_stds = [], [], [], []
        meas_blocks, pred_blocks = [], []
        for tr in treats:
            m = bft_meas[bft_meas["Treatment"] == tr][tcol].values * 100
            p = bft_pred[bft_pred["Treatment"] == tr][pcol].values * 100
            meas_means.append(m.mean())
            meas_stds.append(m.std())
            pred_means.append(p.mean())
            pred_stds.append(p.std())
            meas_blocks.append(m)
            pred_blocks.append(p)

        # 8 bars total (4 components × meas/est), positions: -3.5w to +3.5w
        meas_pos = (-3.5 + s_i * 2) * width_b
        est_pos = (-2.5 + s_i * 2) * width_b

        ax_b.bar(x + meas_pos, meas_means, width_b,
                 color=color, alpha=0.9, edgecolor='none')
        ax_b.errorbar(x + meas_pos, meas_means, yerr=meas_stds,
                      fmt='none', ecolor='gray', elinewidth=0.6, capsize=1.5)

        ax_b.bar(x + est_pos, pred_means, width_b,
                 color=color, alpha=0.5, edgecolor=color, linewidth=0.8,
                 hatch='///')
        ax_b.errorbar(x + est_pos, pred_means, yerr=pred_stds,
                      fmt='none', ecolor='gray', elinewidth=0.6, capsize=1.5)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(treat_labels, fontsize=10)
    ax_b.set_ylim(0, None)
    ax_b.set_ylabel('Fractional Cover (%)', fontsize=11, fontweight='bold')
    ax_b.set_xlabel('Treatments', fontsize=11, fontweight='bold')
    ax_b.set_title('(b) BFT components: Measured vs Estimated',
                    fontsize=11, fontweight='bold', loc='left')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.tick_params(labelsize=10)

    legend_b = [
        Patch(facecolor='gray', alpha=0.9, edgecolor='none', label='Measured (solid)'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='gray', hatch='///', label='Estimated (hatched)'),
        Patch(facecolor='#e41a1c', edgecolor='none', label='LtCy'),
        Patch(facecolor='#4daf4a', edgecolor='none', label='DkCy'),
        Patch(facecolor='#00bfc4', edgecolor='none', label='Lichen'),
        Patch(facecolor='#984ea3', edgecolor='none', label='Moss'),
    ]
    ax_b.legend(handles=legend_b, fontsize=9, frameon=False,
                loc='upper left', ncol=3, columnspacing=1.0)

    # ===================================================================
    # Panel (c): Treatment effect — Successional (difference from Control)
    # ===================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    effect_treats = ["AltP", "Warmed", "AltP+Warmed"]
    effect_labels = ["AltP", "Warmed", "AltP+Warmed"]
    x_eff = np.arange(len(effect_treats))
    width_e = 0.2

    for s_i, (tcol, label, color) in enumerate(
            zip(succ_targets, succ_labels_short, succ_colors)):
        diffs, diff_stds = [], []
        block_diffs = []
        for tr in effect_treats:
            ctrl = succ_meas[succ_meas["Treatment"] == "Control"].set_index("plot")[tcol]
            treat_vals = succ_meas[succ_meas["Treatment"] == tr].set_index("plot")[tcol]
            diff = (treat_vals - ctrl) * 100
            diffs.append(diff.mean())
            diff_stds.append(diff.std())
            block_diffs.append(diff.values)

        offset = (s_i - 0.5) * width_e * 1.3
        ax_c.bar(x_eff + offset, diffs, width_e, color=color, alpha=0.85,
                 edgecolor='none', label=label)
        ax_c.errorbar(x_eff + offset, diffs, yerr=diff_stds,
                      fmt='none', ecolor='gray', elinewidth=0.8, capsize=2)
        # Block dots
        for t_i in range(len(effect_treats)):
            ax_c.scatter(np.full_like(block_diffs[t_i], x_eff[t_i] + offset),
                         block_diffs[t_i], color='black', s=10, zorder=5, alpha=0.6)

    ax_c.axhline(0, color='black', linewidth=0.8, linestyle='-', zorder=0)
    ax_c.set_xticks(x_eff)
    ax_c.set_xticklabels(effect_labels, fontsize=10)
    ax_c.set_ylabel('Δ Fractional Cover (%)\n(Treatment − Control)', fontsize=10, fontweight='bold')
    ax_c.set_xlabel('Treatments', fontsize=11, fontweight='bold')
    ax_c.set_title('(c) Treatment effect on successional stages',
                    fontsize=11, fontweight='bold', loc='left')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.tick_params(labelsize=10)
    # Legend shown in panel (a), not here

    # ===================================================================
    # Panel (d): Treatment effect — BFT components (difference from Control)
    # ===================================================================
    ax_d = fig.add_subplot(gs[1, 1])

    width_d = 0.12

    for s_i, (tcol, label, color) in enumerate(
            zip(bft_targets, bft_labels_short, bft_colors)):
        diffs, diff_stds = [], []
        block_diffs = []
        for tr in effect_treats:
            ctrl = bft_meas[bft_meas["Treatment"] == "Control"].set_index("plot")[tcol]
            treat_vals = bft_meas[bft_meas["Treatment"] == tr].set_index("plot")[tcol]
            diff = (treat_vals - ctrl) * 100
            diffs.append(diff.mean())
            diff_stds.append(diff.std())
            block_diffs.append(diff.values)

        offset = (s_i - (n_comp - 1) / 2) * width_d * 1.3
        ax_d.bar(x_eff + offset, diffs, width_d, color=color, alpha=0.85,
                 edgecolor='none', label=label)
        ax_d.errorbar(x_eff + offset, diffs, yerr=diff_stds,
                      fmt='none', ecolor='gray', elinewidth=0.6, capsize=1.5)
        for t_i in range(len(effect_treats)):
            ax_d.scatter(np.full_like(block_diffs[t_i], x_eff[t_i] + offset),
                         block_diffs[t_i], color='black', s=8, zorder=5, alpha=0.6)

    ax_d.axhline(0, color='black', linewidth=0.8, linestyle='-', zorder=0)
    ax_d.set_xticks(x_eff)
    ax_d.set_xticklabels(effect_labels, fontsize=10)
    ax_d.set_ylabel('Δ Fractional Cover (%)\n(Treatment − Control)', fontsize=10, fontweight='bold')
    ax_d.set_xlabel('Treatments', fontsize=11, fontweight='bold')
    ax_d.set_title('(d) Treatment effect on BFT components',
                    fontsize=11, fontweight='bold', loc='left')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.tick_params(labelsize=10)
    # Legend shown in panel (b), not here

    # --- Save ---
    out_file = out_dir / 'Figure8_v2.png'
    fig.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()
