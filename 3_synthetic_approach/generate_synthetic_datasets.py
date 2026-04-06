import os
import numpy as np
import pandas as pd

def parse_component(id_str):
    parts = id_str.split("_")
    if len(parts) >= 2:
        return parts[-2]
    else:
        return id_str

rng = np.random.default_rng(42)

def sample_abundances(k, alpha=None):
    """
    k: number of endmembers in this synthetic pixel
    alpha: Dirichlet parameter. If None, randomly sample from [0.3, 1.0]
           to produce a wider variety of mixture compositions.
    """
    if alpha is None:
        alpha = rng.uniform(0.3, 1.0)
    return rng.dirichlet(alpha * np.ones(k))

def synthesize_one_pixel(lib, components, band_cols,
                         n_components_range=(1, 8),
                         bilinear_weight=0.5):
    """
    Bilinear mixing model (Fan et al., 2009):
      mixed = Σ aᵢ·sᵢ  +  bilinear_weight * Σᵢ<ⱼ aᵢ·aⱼ·(sᵢ⊙sⱼ)

    The first term is the standard linear mixture.
    The second term adds pairwise interaction (element-wise product of
    endmember spectra, weighted by the product of their abundances).
    This simulates multiple scattering between layered components
    (e.g., light passing through lichen and reflecting off cyanobacteria).

    bilinear_weight controls the strength of the nonlinear interaction:
      0.0 = purely linear (same as v1/v2)
      1.0 = full bilinear model
      0.5 = moderate nonlinearity (default)
    """
    n_bands = len(band_cols)

    # 1) Random number of components
    k = rng.integers(n_components_range[0], n_components_range[1] + 1)

    # 2) Select k components
    comps_selected = rng.choice(components, size=k, replace=False)

    # 3) For each component, pick a random pure spectrum
    pure_spectra = []
    for comp in comps_selected:
        subset = lib[lib["component"] == comp]
        row = subset.sample(n=1, random_state=rng.integers(0, 1_000_000))
        spectrum = row[band_cols].values.squeeze().astype(np.float64)
        pure_spectra.append(spectrum)
    pure_spectra = np.vstack(pure_spectra)  # (k, n_bands)

    # 4) Generate abundances with variable alpha
    abundances = sample_abundances(k, alpha=None)

    # 5) Linear mixing term: Σ aᵢ·sᵢ
    linear_term = (abundances[:, None] * pure_spectra).sum(axis=0)

    # 6) Bilinear interaction term: Σᵢ<ⱼ aᵢ·aⱼ·(sᵢ⊙sⱼ)
    bilinear_term = np.zeros(n_bands)
    if k >= 2:
        for i in range(k):
            for j in range(i + 1, k):
                bilinear_term += (abundances[i] * abundances[j]
                                  * pure_spectra[i] * pure_spectra[j])

    # 7) Combine: linear + weighted bilinear
    mixed = linear_term + bilinear_weight * bilinear_term

    # 8) Physical constraint [0, 1]
    mixed = np.clip(mixed, 0, 1)

    # 9) Build abundance dictionary
    abundance_dict = {c: 0.0 for c in components}
    for c, f in zip(comps_selected, abundances):
        abundance_dict[c] = float(f)

    return mixed, abundance_dict


def generate_synthetic_dataset(lib, components, band_cols,
                               n_samples=10000,
                               n_components_range=(1, 6),
                               bilinear_weight=0.5):
    n_bands = len(band_cols)
    X = np.zeros((n_samples, n_bands))
    y = np.zeros((n_samples, len(components)))

    for i in range(n_samples):
        mixed, abundance_dict = synthesize_one_pixel(
            lib, components, band_cols,
            n_components_range=n_components_range,
            bilinear_weight=bilinear_weight,
        )
        X[i, :] = mixed
        y[i, :] = [abundance_dict[c] for c in components]

        if (i + 1) % 5000 == 0:
            print(f"{i+1}/{n_samples} synthetic pixels generated")

    X_df = pd.DataFrame(X, columns=band_cols)
    y_df = pd.DataFrame(y, columns=[f"frac_{c}" for c in components])

    synth_df = pd.concat([X_df, y_df], axis=1)
    return synth_df


# =====================================================
# MAIN
# =====================================================
data_path = "../1_data/ASD_Specra/"

lib = pd.read_csv(f"{data_path}ASD_All_Spectra_ContactProbe.csv")
id_col = lib.columns[0]
band_cols = lib.columns[1:]

lib["component"] = lib[id_col].astype(str).apply(parse_component)
lib.loc[lib["component"] == 'BRT', "component"] = 'Vegetation'
lib.loc[lib["component"] == 'DCY', "component"] = 'DkCy'
lib.loc[lib["component"] == 'LCN', "component"] = 'Lichen'
lib.loc[lib["component"] == 'LCY', "component"] = 'LtCy'
lib.loc[lib["component"] == 'LTR', "component"] = 'Litter'
lib.loc[lib["component"] == 'MSS', "component"] = 'Moss'
lib.loc[lib["component"] == 'ROCK', "component"] = 'Rock'
lib.loc[lib["component"] == 'SOIL', "component"] = 'Soil'
lib = lib[(lib["component"] != 'Soil') & (lib["component"] != 'Rock')]

components = sorted(lib["component"].unique())
print("Components:", components)

synth = generate_synthetic_dataset(
    lib, components, band_cols,
    n_samples=10000,
    n_components_range=(1, 6),
    bilinear_weight=0.5,
)

synth.to_csv("../1_data/Processed_data/synthetic_mixtures.csv", index=False)
print(f"Saved {len(synth)} synthetic samples to synthetic_mixtures.csv")
