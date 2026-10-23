"""
Build the measured plot-level dataset shared by the RF branch and the synthetic-CNN branch.

Reads the raw ASD plot spectra and the USGS point-intercept cover table from this branch's
1_data/, builds the targets, and writes IDENTICAL copies to
    <re_do_analysis>/RF/1_data/Processed_data/        and
    <re_do_analysis>/synthetic/1_data/Processed_data/
plus a manifest.json (rows, columns, md5 per file) so both branches can verify they use the
same data. Row order follows the ASD spectra file and is never changed (the CV folds depend on it).

Targets (v2, with bare substrate):
    frac_Vegetation = Litter + Live + Dead   (multilayer point-intercept cover; can exceed 1)
    frac_Lichen, frac_Moss, frac_DkCy (Dark), frac_LtCy (Light)   (ground-layer fractions)
    frac_Bare       = Rock + Bare            (bare substrate; ground layer)
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BRANCH = HERE.parent                       # RF/ or synthetic/
ROOT = BRANCH.parent                       # re_do_analysis/
OUT_DIRS = [ROOT / "RF" / "1_data" / "Processed_data", ROOT / "synthetic" / "1_data" / "Processed_data"]
TARGET_COLS = ["frac_Vegetation", "frac_Lichen", "frac_Moss", "frac_DkCy", "frac_LtCy", "frac_Bare"]
# 9 spectral indices (the former moisture index, R1850/R1925, was dropped: both bands lie inside the
# excluded 1770-1960 nm water-absorption window)
INDEX_COLS = ["brightness_index", "NDVI", "PRI", "NDNI", "NDWI", "MCARI", "CI", "SCAI", "BSCI"]

df_spectra = pd.read_csv(BRANCH / "1_data" / "ASD_Specra" / "ASD_All_Spectra_PlotLevel.csv")
df_cover = pd.read_csv(BRANCH / "1_data" / "USGS Cover Data" / "FractionaCover_BySpectra_2021_v2.csv")

df_cover.loc[df_cover["Treatment"] == "C", "Treat"] = "CC"
df_cover.loc[df_cover["Treatment"] == "L", "Treat"] = "LC"
df_cover.loc[df_cover["Treatment"] == "LW", "Treat"] = "LW"
df_cover.loc[df_cover["Treatment"] == "W", "Treat"] = "CW"
df_cover["subplots"] = df_cover["Spectra"].str[1:]
df_cover["full"] = (df_cover["Plot"].astype(str) + df_cover["Block"].astype(str) + "_"
                    + df_cover["Treat"].astype(str) + "_" + df_cover["subplots"].astype(str))

df_cover["frac_Vegetation"] = df_cover["Litter"] + df_cover["Live"] + df_cover["Dead"]
df_cover["frac_Bare"] = df_cover["Rock"] + df_cover["Bare"]
df_cover = df_cover.rename(columns={"Lichen": "frac_Lichen", "Moss": "frac_Moss",
                                    "Dark": "frac_DkCy", "Light": "frac_LtCy"})[["full"] + TARGET_COLS]

df_measure = pd.merge(df_spectra, df_cover, how="left", on="full")
df_measure.dropna(subset=TARGET_COLS, how="all", inplace=True)
df_measure.reset_index(drop=True, inplace=True)

eps = 1e-8
d = df_measure
d["brightness_index"] = np.sqrt((d["560"] ** 2 + d["650"] ** 2 + d["850"] ** 2) / 3)   # root mean square (Eq. 1)
d["NDVI"] = (d["800"] - d["670"]) / (d["800"] + d["670"] + eps)
d["PRI"] = (d["531"] - d["570"]) / (d["531"] + d["570"] + eps)
# NDNI (Serrano et al., 2002): [log(1/R1510) - log(1/R1680)] / [log(1/R1510) + log(1/R1680)]
log_1510, log_1680 = np.log(1.0 / (d["1510"] + eps)), np.log(1.0 / (d["1680"] + eps))
d["NDNI"] = (log_1510 - log_1680) / (log_1510 + log_1680)
d["NDWI"] = (d["860"] - d["1240"]) / (d["860"] + d["1240"] + eps)
d["MCARI"] = ((d["700"] - d["670"]) - 0.2 * (d["700"] - d["550"])) * (d["700"] / d["670"] + eps)
d["CI"] = 1 - (d["670"] - d["450"]) / (d["670"] + d["450"] + eps)
d["SCAI"] = (d["1610"] - d["2200"]) / (d["1610"] + d["2200"] + eps)
d["BSCI"] = (1 - 2 * np.abs(d["670"] - d["560"])) / ((d["560"] + d["670"] + d["800"]) / 3)

band_cols = [c for c in df_measure.columns if c.isdigit()]
outputs = {
    "measured_mixtures.csv": df_measure[["full"] + band_cols + TARGET_COLS],
    "measured_mixtures_with_indices.csv": df_measure[["full"] + band_cols + TARGET_COLS + INDEX_COLS],
}

# plot-level aggregates (mean spectrum, summed cover re-normalised) — kept for completeness
df_measure["plot_key"] = df_measure["full"].apply(lambda x: "_".join(x.split("_")[:2]))
for name, extra in [("plot_aggregated_measured_mixtures.csv", []),
                    ("plot_aggregated_measured_mixtures_with_indices.csv", INDEX_COLS)]:
    spec = df_measure[["plot_key"] + band_cols + extra].groupby("plot_key").mean().reset_index()
    cov = df_measure[["plot_key"] + TARGET_COLS].groupby("plot_key").sum()
    cov = cov.div(cov.sum(axis=1), axis=0).reset_index()
    outputs[name] = pd.merge(spec, cov, how="left", on="plot_key").rename(columns={"plot_key": "full"})

manifest = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "source_branch": BRANCH.name,
            "targets": TARGET_COLS, "files": {}}
for name, frame in outputs.items():
    csv_bytes = frame.to_csv(index=False).encode()
    manifest["files"][name] = dict(rows=int(len(frame)), columns=int(frame.shape[1]),
                                   md5=hashlib.md5(csv_bytes).hexdigest())
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_bytes(csv_bytes)
for out_dir in OUT_DIRS:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

m = outputs["measured_mixtures.csv"]
print(f"measured_mixtures.csv: {len(m)} rows, {len(band_cols)} bands, targets {TARGET_COLS}")
print(m[TARGET_COLS].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
print(f"frac_Bare == 0 in {(m['frac_Bare'] == 0).mean():.0%} of samples; > 0.1 in {(m['frac_Bare'] > 0.1).mean():.0%}")
print("Written to:", *[str(p) for p in OUT_DIRS], sep="\n  ")
