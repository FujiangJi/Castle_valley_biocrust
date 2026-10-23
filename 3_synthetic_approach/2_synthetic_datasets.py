"""
Export reference synthetic datasets (for inspection / figures) and a realism summary against the
measured spectra. The CNN pipeline does NOT read these files: 5_tune_hparams.py regenerates the
synthetic data on the fly from each trial's `data` config (deterministic, seed 42), so the search
can vary the generation settings. This script exports the default configuration
(prior=uniform, k_max=8, gamma=0.5) for each mixing model and, optionally, other priors.

Usage: python 2_synthetic_datasets.py [--priors uniform matched] [--n-samples 10000]
Outputs: 1_data/Processed_data/synthetic_mixtures_<mixing>_<prior>.csv (bands 350-2500 + frac_<component>)
         1_data/Processed_data/synthetic_summary.md
"""
import argparse

import numpy as np
import pandas as pd

from cnn_common import COMPONENTS, DATA_DIR, MEASURED_COMPOSITION, generate_synthetic, load_library

ap = argparse.ArgumentParser()
ap.add_argument("--priors", nargs="+", default=["uniform"], choices=["uniform", "matched"])
ap.add_argument("--n-samples", type=int, default=10000)
args = ap.parse_args()

lib = load_library()
meas = pd.read_csv(DATA_DIR / "measured_mixtures.csv")
mb = [c for c in meas.columns if c.isdigit()]
lines = ["# Synthetic data realism summary\n",
         f"Measured: {len(meas)} spectra. Synthetic: {args.n_samples} per dataset; endmember classes {COMPONENTS}; "
         "endmember variability on (convex combination of 1-2 library spectra × brightness factor N(1, 0.05)).\n"]
probe = ["550", "850", "1650", "2200"]
lines.append("| dataset | " + " | ".join(f"mean R@{w} (sd)" for w in probe) + " | " + " | ".join(f"mean a({c})" for c in COMPONENTS) + " |")
lines.append("|---|" + "---|" * (len(probe) + len(COMPONENTS)))
lines.append("| measured | " + " | ".join(f"{meas[w].mean():.3f} ({meas[w].std():.3f})" for w in probe) + " | "
             + " | ".join(f"{MEASURED_COMPOSITION[c]:.3f}*" for c in COMPONENTS) + " |")
for mixing in ["linear", "bilinear"]:
    scheme = f"{mixing}_6scheme"
    for prior in args.priors:
        cfg = dict(prior=prior, n_samples=args.n_samples)
        X, A, bands = generate_synthetic(scheme, cfg, lib)
        df = pd.concat([pd.DataFrame(X, columns=bands), pd.DataFrame(A, columns=[f"frac_{c}" for c in COMPONENTS])], axis=1)
        out = DATA_DIR / f"synthetic_mixtures_{mixing}_{prior}.csv"
        df.to_csv(out, index=False, float_format="%.5f")
        bi = {w: bands.index(w) for w in probe}
        lines.append(f"| {mixing}, {prior} | " + " | ".join(f"{X[:, bi[w]].mean():.3f} ({X[:, bi[w]].std():.3f})" for w in probe)
                     + " | " + " | ".join(f"{A[:, i].mean():.3f}" for i in range(len(COMPONENTS))) + " |")
        print(f"Saved {out.name}: {len(df)} rows, {len(bands)} bands  | mean reflectance @850 nm {X[:, bi['850']].mean():.3f}")
lines.append("\n\\* measured composition = mean field cover used as inclusion weights by the matched prior "
             "(Vegetation = Live + Dead canopy layers; ground layer sums to ~1.1 because litter overlaps biocrust).")
(DATA_DIR / "synthetic_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
