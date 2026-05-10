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

    # --- Load data (reflectance only, per-class feature importance) ---
    df3 = pd.read_csv(rf_dir / "rf_separate_3class_refl_feature_imp.csv")
    df5 = pd.read_csv(rf_dir / "rf_separate_5class_refl_feature_imp.csv")

    # Band columns (all except target_name)
    band_cols_3 = [c for c in df3.columns if c != 'target_name']
    band_cols_5 = [c for c in df5.columns if c != 'target_name']
    wavelengths_3 = np.array([float(c) for c in band_cols_3])
    wavelengths_5 = np.array([float(c) for c in band_cols_5])

    # --- Define panels ---
    # Senescent Vegetation uses df3 (3-class scheme) with frac_Vegetation
    # since the 5-class scheme only contains DkCy/Lichen/Moss
    panels = [
        ('Late Succ. (DkCy+Lichen+Moss)', df3, band_cols_3, wavelengths_3,
         'frac_late_successional', '#377eb8'),
        ('Early Succ. (LtCy)', df3, band_cols_3, wavelengths_3,
         'frac_early_successional', '#e41a1c'),
        ('DkCy', df5, band_cols_5, wavelengths_5,
         'frac_DkCy', '#4daf4a'),
        ('Lichen', df5, band_cols_5, wavelengths_5,
         'frac_Lichen', '#00bfc4'),
        ('Moss', df5, band_cols_5, wavelengths_5,
         'frac_Moss', '#984ea3'),
        ('Senescent Vegetation', df3, band_cols_3, wavelengths_3,
         'frac_Vegetation', '#a6761d'),
    ]

    # Bad bands (shaded regions)
    bad_bands = [[1320, 1440], [1770, 1960]]

    # --- Figure: 2 rows x 3 cols ---
    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(2, 3, left=0, right=1, top=1, bottom=0, hspace=0.3, wspace=0.25)

    for idx, (title, df, bcols, wl, target, color) in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

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
                center = (rlo + rhi) / 2
                ylim = ax.get_ylim()
                ax.text(center, ylim[1] * 0.98,
                        f'{int(rlo)}–{int(rhi)} nm',
                        fontsize=6, fontweight='bold', color=color,
                        ha='center', va='top', zorder=5, rotation=90,
                        bbox=dict(facecolor='white', alpha=0.7,
                                  edgecolor='none', pad=0.5))
        ax.tick_params(labelsize=9)
        label = chr(ord('a') + idx)
        ax.set_title(f'{label}. {title}', fontsize=10, fontweight='bold', loc='left')

        if row == 1:
            ax.set_xlabel('Wavelength (nm)', fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Feature\nImportance', fontsize=10, fontweight='bold')

    # --- Save ---
    out_file = out_dir / 'Figure5.png'
    fig.savefig(out_file, dpi=500, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()
