from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def main():
    root = Path(__file__).resolve().parent
    rf_dir = root.parent.parent / "re_do_analysis" / "RF" / "2_results"
    out_dir = root / "0_exported_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data (v2 re-do; reflectance + indices feature importance) ---
    df4 = pd.read_csv(rf_dir / "rf_separate_4class_refl_indices_feature_imp.csv")
    df6 = pd.read_csv(rf_dir / "rf_separate_6class_refl_indices_feature_imp.csv")

    # Separate wavelength columns from index columns (index set taken from the results file)
    index_cols = {c for c in df4.columns if c != 'target_name' and not c.isdigit()}
    assert index_cols == {c for c in df6.columns if c != 'target_name' and not c.isdigit()}
    band_cols_4 = [c for c in df4.columns if c != 'target_name' and c not in index_cols]
    band_cols_6 = [c for c in df6.columns if c != 'target_name' and c not in index_cols]
    wavelengths_4 = np.array([float(c) for c in band_cols_4])
    wavelengths_6 = np.array([float(c) for c in band_cols_6])

    # --- Compute reflectance band contribution % for each panel ---
    panels = [
        ('Late Succ. (DkCy+Lichen+Moss)', df4, band_cols_4, wavelengths_4,
         'frac_late_successional', '#377eb8'),
        ('Early Succ. (LtCy)', df4, band_cols_4, wavelengths_4,
         'frac_early_successional', '#e41a1c'),
        ('DkCy', df6, band_cols_6, wavelengths_6,
         'frac_DkCy', '#4daf4a'),
        ('Lichen', df6, band_cols_6, wavelengths_6,
         'frac_Lichen', '#00bfc4'),
        ('Moss', df6, band_cols_6, wavelengths_6,
         'frac_Moss', '#984ea3'),
        ('Senescent Vegetation', df4, band_cols_4, wavelengths_4,
         'frac_Vegetation', '#a6761d'),
        ('Bare Substrate', df4, band_cols_4, wavelengths_4,
         'frac_Bare', '#636363'),
    ]

    # Pre-compute band contribution % for subtitles
    band_pcts = []
    for _, df_src, bcols, _, target, _ in panels:
        subset = df_src[df_src['target_name'] == target]
        band_imp = subset[bcols].values.astype(float).mean(axis=0).sum()
        band_pcts.append(band_imp)

    # Bad bands (shaded regions)
    bad_bands = [[1320, 1440], [1770, 1960]]

    # --- Combined figure: nested GridSpec for different spacing ---
    n_cols = 4
    n_panels = len(panels)
    fig = plt.figure(figsize=(11, 7.5))
    # Outer grid: top block (rows 1-2) and bottom block (row 3)
    outer_gs = gridspec.GridSpec(2, 1, hspace=0.28, height_ratios=[2, 1.5], left=0, right=1, top=1, bottom=0)
    # Inner grid for rows 1-2: 7 panels as 4 on top and 3 centred below.
    # Use an 8-column grid; each panel spans 2 columns, and the bottom row
    # is offset by one column so its 3 panels sit centred under the 4 above.
    gs_top = gridspec.GridSpecFromSubplotSpec(2, 2 * n_cols, subplot_spec=outer_gs[0],
                                              hspace=0.45, wspace=0.7)
    # Bottom row grid
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[1])

    # Panel (a) label above the top rows
    fig.text(-0.06, 1.07, '(a) The contribution of spectral bands',
             fontsize=12, fontweight='bold', va='top', ha='left')

    for idx, (title, df, bcols, wl, target, color) in enumerate(panels):
        row, col = divmod(idx, n_cols)
        offset = 0 if row == 0 else 1
        c0 = 2 * col + offset
        ax = fig.add_subplot(gs_top[row, c0:c0 + 2])

        subset = df[df['target_name'] == target]
        imp_vals = subset[bcols].values.astype(float)
        mean_imp = imp_vals.mean(axis=0)
        std_imp = imp_vals.std(axis=0)

        # Insert NaN at bad band gaps for clean line breaks
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

        # Plot mean importance as filled area with std shading
        ax.fill_between(wl_plot, mean_plot - std_plot, mean_plot + std_plot,
                        color=color, alpha=0.2, zorder=3)
        ax.plot(wl_plot, mean_plot, color=color, linewidth=0.8, zorder=4)

        # Shade bad bands
        for lo, hi in bad_bands:
            ax.axvspan(lo, hi, color='lightgray', alpha=0.2, zorder=0)

        ax.set_xlim(400, 2400)
        # Set ylim based on values within 400-2400 nm
        in_range = (wl >= 400) & (wl <= 2400) & ~np.isnan(mean_imp)
        if in_range.any():
            y_max = (mean_imp[in_range] + std_imp[in_range]).max()
            ax.set_ylim(0, y_max * 1.05)
        else:
            ax.set_ylim(0, None)

        # --- Identify and shade important spectral regions (top 5%) ---
        valid = in_range
        if valid.any():
            threshold = np.percentile(mean_imp[valid], 95)
            important = np.zeros(len(wl), dtype=bool)
            important[valid] = mean_imp[valid] >= threshold

            # Find contiguous important regions
            in_region = False
            region_start = 0
            regions_found = []
            for i in range(len(wl)):
                if important[i] and not in_region:
                    region_start = wl[i]
                    in_region = True
                elif not important[i] and in_region:
                    regions_found.append((region_start, wl[i-1]))
                    in_region = False
            if in_region:
                regions_found.append((region_start, wl[-1]))

            # Merge: first expand ±20 nm, then union regions within 50 nm gap
            pad = 20  # nm expansion
            gap_tol = 20 # nm — merge regions closer than this
            expanded = [(rlo - pad, rhi + pad) for rlo, rhi in sorted(regions_found)]

            # First pass: merge overlapping after expansion
            merged = []
            for rlo, rhi in expanded:
                if merged and rlo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], rhi))
                else:
                    merged.append((rlo, rhi))

            # Second pass: merge regions within gap_tol of each other
            final = []
            for rlo, rhi in merged:
                if final and rlo - final[-1][1] <= gap_tol:
                    final[-1] = (final[-1][0], max(final[-1][1], rhi))
                else:
                    final.append((rlo, rhi))

            # Clip to display range
            final = [(max(rlo, 400), min(rhi, 2400)) for rlo, rhi in final
                     if rhi > 400 and rlo < 2400]

            for rlo, rhi in final:
                ax.axvspan(rlo, rhi, color=color, alpha=0.30, zorder=0)
                # Label the region at top of panel
                # Keep the label clear of the axes frame when a region
                # touches the 400 / 2400 nm edge
                label_margin = 45  # nm
                center = np.clip((rlo + rhi) / 2, 400 + label_margin, 2400 - label_margin)
                ylim = ax.get_ylim()
                ax.text(center, ylim[1] * 0.98,
                        f'{int(rlo)}–{int(rhi)} nm',
                        fontsize=6, fontweight='bold', color=color,
                        ha='center', va='top', zorder=5, rotation=90,
                        bbox=dict(facecolor='white', alpha=0.7,
                                  edgecolor='none', pad=0.5))
        ax.tick_params(labelsize=9)
        sub_label = f'(a.{idx + 1})'
        pct = band_pcts[idx]
        ax.set_title(f'{sub_label} {title} ({pct:.0%})', fontsize=8, fontweight='bold', loc='left')

        ax.set_xlabel('Wavelength (nm)', fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Feature\nImportance', fontsize=10, fontweight='bold')

    # =================================================================
    # Bottom row: (b) Stacked bar chart spanning all 3 columns
    # =================================================================
    ax2 = fig.add_subplot(gs_bot[0, 0])
    
    # Stacking order, legend label and colour per index; only indices present in the results are drawn
    index_style = {"NDWI": ("NDWI", "#1f77b4"), "PRI": ("PRI", "#ff7f0e"), "MCARI": ("MCARI", "#2ca02c"),
                   "NDNI": ("NDNI", "#d62728"), "NDVI": ("NDVI", "#9467bd"),
                   "soil_moisture": ("Moisture index", "#8c564b"),
                   "brightness_index": ("Brightness index", "#0bea4e"), "CI": ("CI", "#554358"),
                   "SCAI": ("SCAI", "#ff0ebf"), "BSCI": ("BSCI", "#7f7f7f")}
    assert index_cols <= set(index_style), f"no style for {index_cols - set(index_style)}"
    idx_list = [k for k in index_style if k in index_cols]
    idx_labels = [index_style[k][0] for k in idx_list]
    idx_colors = [index_style[k][1] for k in idx_list]

    targets = [
        ('Late Succ.', df4, 'frac_late_successional'),
        ('Early Succ.', df4, 'frac_early_successional'),
        ('DkCy', df6, 'frac_DkCy'),
        ('Lichen', df6, 'frac_Lichen'),
        ('Moss', df6, 'frac_Moss'),
        ('Senescent Vegetation', df4, 'frac_Vegetation'),
        ('Bare Substrate', df4, 'frac_Bare'),
    ]

    x = np.arange(len(targets))
    bar_width = 0.6
    bottom = np.zeros(len(targets))

    # Store per-index values for annotation
    all_idx_vals = np.zeros((len(idx_list), len(targets)))
    all_idx_bottoms = np.zeros((len(idx_list), len(targets)))

    for i, (idx_name, idx_label) in enumerate(zip(idx_list, idx_labels)):
        vals = []
        for _, df_src, tname in targets:
            sub = df_src[df_src['target_name'] == tname]
            vals.append(sub[idx_name].values.astype(float).mean())
        vals = np.array(vals)
        all_idx_bottoms[i] = bottom.copy()
        all_idx_vals[i] = vals
        ax2.bar(x, vals, bar_width, bottom=bottom, label=idx_label,
                color=idx_colors[i], edgecolor='white', linewidth=0.3)
        bottom += vals

    # Add "Reflectance bands" segment on top
    refl_vals = []
    for _, df_src, tname in targets:
        sub = df_src[df_src['target_name'] == tname]
        all_idx_sum = sub[idx_list].values.astype(float).mean(axis=0).sum()
        refl_vals.append(1.0 - all_idx_sum)
    refl_vals = np.array(refl_vals)
    ax2.bar(x, refl_vals, bar_width, bottom=bottom, label='Reflectance bands',
            color='#e0e0e0', edgecolor='white', linewidth=0.3)

    # Annotate total index % and largest index contribution
    for j in range(len(targets)):
        total_idx = bottom[j]
        ax2.text(j, total_idx + 0.01, f'{total_idx:.0%}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Find the largest index for this target
        max_i = np.argmax(all_idx_vals[:, j])
        max_val = all_idx_vals[max_i, j]
        max_bottom = all_idx_bottoms[max_i, j]
        max_color = idx_colors[max_i]
        # Place label in the center of the largest segment
        ax2.text(j, max_bottom + max_val / 2, f'{max_val:.0%}',
                 ha='center', va='center', fontsize=7, fontweight='bold',
                 color=max_color,
                 bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=1))

    ax2.set_xticks(x)
    ax2.set_xticklabels([t[0] for t in targets], fontsize=10)
    ax2.set_xlabel('Biocrust Functional Types', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Feature Importance\n(fraction of total)', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 1.12)
    ax2.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.4,
               edgecolor='none', ncol=len(idx_list) + 1, columnspacing=0.4, handletextpad=0.1,
               bbox_to_anchor=(1.0, 1.03))
    
    fig.text(-0.06, 0.42, '(b) The contribution of hyperspectral indices',
             fontsize=12, fontweight='bold', va='top', ha='left')

    # --- Save combined figure ---
    out_file = out_dir / 'Figure5_RF_indices.png'
    fig.savefig(out_file, dpi=500, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()
