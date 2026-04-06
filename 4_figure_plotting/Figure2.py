from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def parse_component(name):
    """Extract component code from the ID string (3rd underscore field)."""
    parts = str(name).split('_')
    if len(parts) >= 3:
        return parts[2]
    return None


def main():
    root = Path(__file__).resolve().parent
    rf_dir = root.parent.parent / "4_Castle_valley_analysis_RF" / "1_data"
    out_dir = root / "0_exported_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    df_lib = pd.read_csv(rf_dir / "ASD_Specra" / "ASD_All_Spectra_ContactProbe.csv")
    df_mix = pd.read_csv(rf_dir / "Processed_data" / "measured_mixtures.csv")

    # --- Process spectra library ---
    id_col = df_lib.columns[0]
    band_cols = df_lib.columns[1:]
    wavelengths = np.array([float(c) for c in band_cols])

    df_lib['component'] = df_lib[id_col].astype(str).apply(parse_component)
    df_lib.loc[df_lib['component'] == 'BRT', 'component'] = 'Vegetation'
    df_lib.loc[df_lib['component'] == 'DCY', 'component'] = 'DkCy'
    df_lib.loc[df_lib['component'] == 'LCN', 'component'] = 'Lichen'
    df_lib.loc[df_lib['component'] == 'LCY', 'component'] = 'LtCy'
    df_lib.loc[df_lib['component'] == 'LTR', 'component'] = 'Litter'
    df_lib.loc[df_lib['component'] == 'MSS', 'component'] = 'Moss'
    df_lib.loc[df_lib['component'] == 'ROCK', 'component'] = 'Rock'
    df_lib.loc[df_lib['component'] == 'SOIL', 'component'] = 'Soil'
    df_lib = df_lib[(df_lib['component'] != 'Soil') & (df_lib['component'] != 'Rock')]

    comp_display = {
        'Vegetation': 'Vegetation', 'DkCy': 'Dark Cyanobacteria',
        'Lichen': 'Lichen', 'LtCy': 'Light Cyanobacteria',
        'Litter': 'Litter', 'Moss': 'Moss',
    }
    comp_colors = {
        'Vegetation': '#33a02c', 'DkCy': '#1b7837',
        'Lichen': '#d95f02', 'LtCy': '#7fbf7b',
        'Litter': '#a6761d', 'Moss': '#1f78b4',
    }

    # --- Process measured mixtures ---
    frac_cols = ['frac_Litter', 'frac_DkCy', 'frac_Lichen',
                 'frac_LtCy', 'frac_Moss', 'frac_Vegetation']
    mix_wl_cols = [c for c in df_mix.columns if c not in ['full'] + frac_cols]
    mix_wavelengths = np.array([float(c) for c in mix_wl_cols])

    # Treatment assignment
    df_mix['plot_key'] = df_mix['full'].apply(lambda x: '_'.join(x.split('_')[:2]))
    df_mix['plot'] = df_mix['full'].apply(lambda x: x.split('_')[0])
    df_mix['treat_code'] = df_mix['full'].apply(lambda x: x.split('_')[1])
    df_mix.loc[df_mix['treat_code'] == 'CC', 'Treatment'] = 'Control'
    df_mix.loc[df_mix['treat_code'] == 'LC', 'Treatment'] = 'Warmed'
    df_mix.loc[df_mix['treat_code'] == 'LW', 'Treatment'] = 'AltP+Warmed'
    df_mix.loc[df_mix['treat_code'] == 'CW', 'Treatment'] = 'AltP'

    treat_order = ['Control', 'AltP', 'Warmed', 'AltP+Warmed']
    treat_colors_spec = {
        'Control': 'dodgerblue', 'AltP': 'limegreen',
        'Warmed': 'orangered', 'AltP+Warmed': 'purple',
    }

    # Combine Litter + Vegetation for cover analysis
    df_mix['frac_Litter+Vegetation'] = df_mix['frac_Litter'] + df_mix['frac_Vegetation']

    # Compute early/late successional
    df_mix['frac_late_successional'] = (df_mix['frac_DkCy'] +
                                         df_mix['frac_Lichen'] +
                                         df_mix['frac_Moss'])
    df_mix['frac_early_successional'] = df_mix['frac_LtCy']

    # --- Prepare cover data for stacked bars ---
    target_cols_5 = ['frac_Litter+Vegetation', 'frac_DkCy',
                     'frac_Lichen', 'frac_LtCy', 'frac_Moss']
    df_cover = df_mix[['plot_key', 'Treatment'] + target_cols_5 +
                       ['frac_late_successional', 'frac_early_successional']].copy()

    # Average per plot_key (no normalization)
    cover_grouped = df_cover.groupby('plot_key')[target_cols_5].mean().reset_index()
    succ_grouped = df_cover.groupby('plot_key')[['frac_late_successional',
                                                   'frac_early_successional']].mean().reset_index()

    # Merge treatment info back
    treat_map = df_mix[['plot_key', 'Treatment', 'plot']].drop_duplicates()
    cover_grouped = cover_grouped.merge(treat_map, on='plot_key')
    succ_grouped = succ_grouped.merge(treat_map, on='plot_key')

    # BFT bar colors
    bft_cols = ['frac_Moss', 'frac_Lichen', 'frac_DkCy', 'frac_LtCy']
    bft_colors = {'frac_LtCy': '#e41a1c', 'frac_DkCy': '#4daf4a',
                  'frac_Lichen': '#00bfc4', 'frac_Moss': '#984ea3'}
    bft_labels = {'frac_LtCy': 'LtCy', 'frac_DkCy': 'DkCy',
                  'frac_Lichen': 'Lichen', 'frac_Moss': 'Moss'}

    # Successional bar colors
    succ_cols = ['frac_late_successional', 'frac_early_successional']
    succ_colors = {'frac_early_successional': '#e41a1c',
                   'frac_late_successional': '#377eb8'}
    succ_labels = {'frac_early_successional': 'Early Succ. (LtCy)',
                   'frac_late_successional': 'Late Succ. (DkCy+Lichen+Moss)'}

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 7))
    outer_gs = gridspec.GridSpec(3, 1, height_ratios=[0.7, 0.6, 0.6],left = 0, right = 1, top = 1, bottom= 0, hspace=0.55)

    gs_a = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0],
                                            wspace=0.15)
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[1],
                                            wspace=0.15)
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[2],
                                            wspace=0.2)

    # ===================================================================
    # A. Variations of spectral library
    # ===================================================================
    fig.text(-0.04, 1.06, 'a. Variations of spectral library',
             fontsize=12, fontweight='bold', va='top')

    a_groups = [
        {'idx': 0, 'comps': ['Vegetation', 'Litter'],
         'title': '(a.1) Vegetation & Litter'},
        {'idx': 1, 'comps': ['DkCy', 'Lichen', 'Moss'],
         'title': '(a.2) Late Succ. (DkCy, Lichen, Moss)'},
        {'idx': 2, 'comps': ['LtCy'],
         'title': '(a.3) Early Succ. (Light Cyanobacteria)'},
    ]

    for group in a_groups:
        ax = fig.add_subplot(gs_a[0, group['idx']])

        for comp in group['comps']:
            subset = df_lib[df_lib['component'] == comp]
            spectra = subset[band_cols].values.astype(float)
            color = comp_colors[comp]

            for i in range(spectra.shape[0]):
                ax.plot(wavelengths, spectra[i], color=color, alpha=0.2,
                        linewidth=0.3)

            mean = spectra.mean(axis=0)
            std = spectra.std(axis=0)
            ax.plot(wavelengths, mean, color=color, linewidth=1.5,
                    label=f'{comp_display[comp]}')
            ax.fill_between(wavelengths, mean - std, mean + std,
                            color=color, alpha=0.15, label='\u00b11 Std')

        ax.set_xlim(350, 2500)
        ax.set_ylim(0, 0.65)
        ax.tick_params(labelsize=9)
        ax.set_title(group['title'], fontsize=10, fontweight='bold', loc='left')
        ax.set_xlabel('Wavelength (nm)', fontsize=10, fontweight='bold')
        if group['idx'] == 0:
            ax.set_ylabel('Reflectance', fontsize=10, fontweight='bold')
        if group['idx'] == 0:
            ax.legend(fontsize=7, loc='upper right', frameon=False, ncol=2)
        elif group['idx'] == 1:
            ax.legend(fontsize=7, loc='lower right', frameon=False, ncol=3)
        else:
            ax.legend(fontsize=7, loc='lower right', frameon=False)

    # ===================================================================
    # B. Variations of measured mixed spectra across all plots
    # ===================================================================
    fig.text(-0.04, 0.66, 'b. Variations of measured mixed spectra across all plots',
             fontsize=12, fontweight='bold', va='top')

    for idx, treat in enumerate(treat_order):
        ax = fig.add_subplot(gs_b[0, idx])

        subset = df_mix[df_mix['Treatment'] == treat]
        spectra = subset[mix_wl_cols].values.astype(float)
        color = treat_colors_spec[treat]

        for i in range(spectra.shape[0]):
            ax.plot(mix_wavelengths, spectra[i], color=color, alpha=0.2,
                    linewidth=0.3)

        mean = spectra.mean(axis=0)
        std = spectra.std(axis=0)
        ax.plot(mix_wavelengths, mean, color=color, linewidth=1.5, label='Mean')
        ax.fill_between(mix_wavelengths, mean - std, mean + std,
                        color=color, alpha=0.2, label='\u00b11 Std')

        ax.set_xlim(350, 2500)
        ax.set_ylim(0, 0.55)
        ax.tick_params(labelsize=9)
        ax.set_title(f'(b.{idx+1}) {treat}', fontsize=9,
                     fontweight='bold', loc='left')

        ax.set_xlabel('Wavelength (nm)', fontsize=10, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Reflectance', fontsize=10, fontweight='bold')

        ax.legend(fontsize=6, loc='upper right', frameon=False)

    # ===================================================================
    # C. Shifts of biocrust functional types under treatments across all plots
    # ===================================================================
    fig.text(-0.04, 0.295,
             'c. Shifts of biocrust functional types under treatments across all plots',
             fontsize=12, fontweight='bold', va='top')

    # --- C.1: Successional stage shifts ---
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c1.set_facecolor((0, 0, 0, 0.01))

    succ_agg = succ_grouped.groupby('Treatment')[succ_cols].mean()
    succ_agg = succ_agg.reindex(treat_order) * 100

    x = np.arange(len(treat_order))
    bar_width = 0.6
    bottom = np.zeros(len(treat_order))
    for col in succ_cols:
        vals = succ_agg[col].fillna(0).values
        ax_c1.bar(x, vals, width=bar_width, bottom=bottom,
                  color=succ_colors[col], edgecolor='none',
                  label=succ_labels[col])
        bottom += vals

    ax_c1.set_xticks(x)
    ax_c1.set_xticklabels(treat_order, rotation=0, ha='center', fontsize=9)
    ax_c1.set_ylim(0, 60)
    ax_c1.tick_params(labelsize=9)
    ax_c1.set_ylabel('Fractional Cover (%)', fontsize=10, fontweight='bold')
    ax_c1.set_xlabel('Treatments', fontsize=10, fontweight='bold')
    ax_c1.set_title('(c.1) Successional stage shifts', fontsize=9,
                     fontweight='bold', loc='left')
    ax_c1.grid(axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax_c1.set_axisbelow(True)
    ax_c1.spines['top'].set_visible(False)
    ax_c1.spines['right'].set_visible(False)
    ax_c1.legend(fontsize=10, loc='upper right', frameon=False,
                  bbox_to_anchor=(1.0, 1.15))

    # --- C.2: BFT component shifts ---
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    ax_c2.set_facecolor((0, 0, 0, 0.01))

    bft_agg = cover_grouped.groupby('Treatment')[bft_cols].mean()
    bft_agg = bft_agg.reindex(treat_order) * 100

    x = np.arange(len(treat_order))
    bottom = np.zeros(len(treat_order))
    for col in bft_cols:
        vals = bft_agg[col].fillna(0).values
        ax_c2.bar(x, vals, width=bar_width, bottom=bottom,
                  color=bft_colors[col], edgecolor='none',
                  label=bft_labels[col])
        bottom += vals

    ax_c2.set_xticks(x)
    ax_c2.set_xticklabels(treat_order, rotation=0, ha='center', fontsize=9)
    ax_c2.set_ylim(0, 60)
    ax_c2.tick_params(labelsize=9)
    ax_c2.set_ylabel('Fractional Cover (%)', fontsize=10, fontweight='bold')
    ax_c2.set_xlabel('Treatments', fontsize=10, fontweight='bold')
    ax_c2.set_title('(c.2) BFT component shifts', fontsize=9,
                     fontweight='bold', loc='left')
    ax_c2.grid(axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax_c2.set_axisbelow(True)
    ax_c2.spines['top'].set_visible(False)
    ax_c2.spines['right'].set_visible(False)
    ax_c2.legend(fontsize=10, loc='upper right', frameon=False, ncol=2,
                  bbox_to_anchor=(1.0, 1.15))

    # --- Save ---
    out_file = out_dir / 'Figure2.png'
    fig.savefig(out_file, dpi=500, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()
