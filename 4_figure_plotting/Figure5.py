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

    # --- Load data ---
    df3 = pd.read_csv(rf_dir / "rf_feature_importance_3class_feature_imp.csv")
    df5 = pd.read_csv(rf_dir / "rf_feature_importance_5class_feature_imp.csv")

    # Band columns (all except target_name)
    band_cols_3 = [c for c in df3.columns if c != 'target_name']
    band_cols_5 = [c for c in df5.columns if c != 'target_name']
    wavelengths_3 = np.array([float(c) for c in band_cols_3])
    wavelengths_5 = np.array([float(c) for c in band_cols_5])

    # --- Define panels ---
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
        ('Litter+Vegetation', df5, band_cols_5, wavelengths_5,
         'frac_Litter+Vegetation', '#a6761d'),
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

        # Mask bad bands with NaN
        mask = np.zeros(len(wl), dtype=bool)
        for lo, hi in bad_bands:
            mask |= (wl >= lo) & (wl <= hi)
        mean_imp[mask] = np.nan
        std_imp[mask] = np.nan

        # Plot mean importance as filled area with std shading
        ax.fill_between(wl, mean_imp - std_imp, mean_imp + std_imp,
                        color=color, alpha=0.2)
        ax.plot(wl, mean_imp, color=color, linewidth=0.8)

        # Shade bad bands
        for lo, hi in bad_bands:
            ax.axvspan(lo, hi, color='lightgray', alpha=0.2, zorder=0)


        ax.set_xlim(400, 2400)
        # Set ylim based on values within 400-2400 nm
        in_range = (wl >= 400) & (wl <= 2400) & ~np.isnan(mean_imp)
        if in_range.any():
            y_max = (mean_imp[in_range] + std_imp[in_range]).max()
            ax.set_ylim(0, y_max)
        else:
            ax.set_ylim(0, None)
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
