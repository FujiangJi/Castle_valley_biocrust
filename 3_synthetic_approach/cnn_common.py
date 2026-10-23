"""
Shared code for the Castle Valley synthetic-CNN pipeline (v2, with bare substrate): generate
synthetic mixtures from the contact-probe endmember library, pre-train a 1D-CNN on them, then
fine-tune on measured ASD spectra under nested cross-validation.
All numbered scripts in this folder are thin CLIs over this module.

Evaluation protocol (mirrors ../RF/0_src_code/2_RF_analysis.py):
  outer KFold(5, shuffle, seed 123) -> 192 train / 48 test per fold, identical to RF's folds
  inner KFold(5, shuffle, seed 42)  -> on the 192, used for epoch selection and for scoring
                                       hyperparameter trials
  The 48 test samples of a fold are predicted exactly once and are never used for any selection.
  Final model per outer fold (FINAL protocol, default 'ensemble'): the mean of the five inner-fold
  models, each early-stopped on its own inner-validation fold.

Three configuration blocks make up one trial (all searched jointly by 5_tune_hparams.py):
  data_cfg      how the synthetic mixtures are generated (prior, k_max, gamma)
  pretrain_cfg  input normalisation, architecture (head, width) and pre-training optimisation
  finetune_cfg  fine-tuning optimisation
The mixing model (linear / bilinear) and the target grouping are fixed by the scheme.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# The GPU does the training; CPU-side work is tiny. Capping PyTorch's CPU thread pool keeps these
# processes from adding dozens of idle/spinning threads on a shared machine (no effect on results).
torch.set_num_threads(4)

# ------------------------------------------------------------------ paths & constants
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent                                   # synthetic/
DATA_DIR = ROOT / "1_data" / "Processed_data"
LIBRARY_CSV = ROOT / "1_data" / "ASD_Specra" / "ASD_All_Spectra_ContactProbe.csv"
MODEL_DIR = ROOT / "2_saved_models"
RESULT_DIR = ROOT / "3_results"
RF_RESULT_DIR = ROOT.parent / "RF" / "2_results"
V1_RESULT_DIR = ROOT / "3_results_v1_no_bare"            # archived run without the Bare target
V1_RF_RESULT_DIR = ROOT.parent / "RF" / "2_results_v1_no_bare"

OUTER_SEED = 123   # == RF outer KFold seed
INNER_SEED = 42    # == RF inner KFold seed
TRAIN_SEED = 42
N_OUTER, N_INNER = 5, 5
BAD_BANDS = [(1320, 1440), (1770, 1960)]   # atmospheric water-absorption windows (nm)

# synthetic endmember classes (contact-probe library codes -> names)
CODE2COMP = {"BRT": "Vegetation", "LTR": "Litter", "DCY": "DkCy", "LCN": "Lichen",
             "LCY": "LtCy", "MSS": "Moss", "SOIL": "Soil", "ROCK": "Rock"}
COMPONENTS = ["Vegetation", "Litter", "DkCy", "Lichen", "LtCy", "Moss", "Soil", "Rock"]
# mean field cover per component (FractionaCover_BySpectra_2021_v2.csv; Vegetation = Live + Dead),
# used by the 'matched' prior as endmember inclusion weights (floored at 0.02)
MEASURED_COMPOSITION = dict(Vegetation=0.451, Litter=0.576, DkCy=0.062, Lichen=0.028,
                            LtCy=0.291, Moss=0.079, Soil=0.004, Rock=0.056)

TARGETS_6 = ["frac_Vegetation", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss", "frac_Bare"]
TARGETS_4 = ["frac_Vegetation", "frac_late_successional", "frac_early_successional", "frac_Bare"]
SCHEMES = {
    "linear_4scheme":   dict(mixing="linear",   targets=TARGETS_4, rf="4class", v1="linear_3scheme"),
    "linear_6scheme":   dict(mixing="linear",   targets=TARGETS_6, rf="6class", v1="linear_5scheme"),
    "bilinear_4scheme": dict(mixing="bilinear", targets=TARGETS_4, rf="4class", v1="bilinear_3scheme"),
    "bilinear_6scheme": dict(mixing="bilinear", targets=TARGETS_6, rf="6class", v1="bilinear_5scheme"),
}

FINAL_PROTOCOLS = ("ensemble", "retrain_annealed", "retrain_snapshot")
DEFAULT_FINAL = "ensemble"
FINAL_DESCRIPTION = {
    "ensemble": "the outer-test fold is predicted by the mean of the five inner-fold models, each "
                "early-stopped on its own inner-validation fold (a cross-validation ensemble)",
    "retrain_annealed": "the outer-test fold is predicted by one model re-trained on the full outer-training "
                        "set from fresh pretrained weights for the inner-selected epoch count, with the cosine "
                        "schedule annealed to that count",
    "retrain_snapshot": "the outer-test fold is predicted by one model re-trained on the full outer-training "
                        "set for the inner-selected epoch count under the full-length schedule (un-annealed snapshot)",
}

DEFAULT_DATA_CFG = dict(prior="uniform", k_max=8, gamma=0.5, n_samples=10000, variability=True, seed=42)
DEFAULT_PRETRAIN_CFG = dict(norm="none", head="gap", width=1.0, lr=1e-3, wd=1e-4, noise_std=0.0, scale_std=0.0,
                            n_epochs=600, batch_size=32, dropout=0.3)
DEFAULT_FINETUNE_CFG = dict(lr=5e-4, wd=5e-5, weighting="plain", scheduler="cos", freeze_conv=False,
                            dropout=0.3, batch_size=32, noise_std=0.0,
                            max_epochs=500, patience=80, min_epochs=20)
NORM_MODES = ("none", "brightness", "deriv")
HEADS = ("gap", "pool16")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=TRAIN_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def state_hash(sd: dict) -> str:
    h = hashlib.sha1()
    for k in sorted(sd):
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def full_data_cfg(scheme, data_cfg):
    """Fill defaults; gamma is meaningless for linear mixing, so it is pinned to 0 there (stable hash)."""
    cfg = {**DEFAULT_DATA_CFG, **data_cfg}
    if SCHEMES[scheme]["mixing"] == "linear":
        cfg["gamma"] = 0.0
    return cfg


# ------------------------------------------------------------------ model
class SpectralCNN(nn.Module):
    """3 conv blocks -> pooling head -> FC. head='gap' averages over wavelength; 'pool16' keeps 16
    wavelength positions before the FC layer (position-aware)."""
    def __init__(self, n_bands, n_targets, head="gap", width=1.0, dropout=0.3):
        super().__init__()
        assert head in HEADS, head
        c1, c2, c3 = int(round(32 * width)), int(round(64 * width)), int(round(128 * width))
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.BatchNorm1d(c2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1), nn.BatchNorm1d(c3), nn.ReLU(),
        )
        n_pos = 1 if head == "gap" else 16
        self.pool = nn.AdaptiveAvgPool1d(n_pos)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c3 * n_pos, 128), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(128, n_targets))

    def forward(self, x):
        return self.fc(self.pool(self.conv_block(x)))


def build_model(n_bands, n_targets, pre_cfg, dropout=None):
    return SpectralCNN(n_bands, n_targets, head=pre_cfg["head"], width=pre_cfg["width"],
                       dropout=pre_cfg["dropout"] if dropout is None else dropout)


# ------------------------------------------------------------------ spectra: normalisation & bands
def normalize_spectra(X, mode):
    """Applied to full 350-2500 nm spectra, before bad-band removal, identically to synthetic and measured."""
    assert mode in NORM_MODES, mode
    X = np.asarray(X, dtype=np.float32)
    if mode == "none":
        return X
    if mode == "brightness":
        return X / (X.mean(axis=1, keepdims=True) + 1e-6)
    return np.gradient(X, axis=1).astype(np.float32)          # first derivative along wavelength


def keep_band_mask(band_names):
    w = np.array([int(b) for b in band_names])
    keep = np.ones(len(w), bool)
    for lo, hi in BAD_BANDS:
        keep &= ~((w >= lo) & (w <= hi))
    return keep


# ------------------------------------------------------------------ synthetic data
def load_library():
    """Contact-probe endmember spectra per component: {name: (n_i, 2151) float32}, band names."""
    lib = pd.read_csv(LIBRARY_CSV)
    bands = [c for c in lib.columns if c.isdigit()]
    comp = lib["full"].astype(str).str.split("_").str[-2].map(CODE2COMP)
    spectra = {c: lib.loc[comp == c, bands].values.astype(np.float32) for c in COMPONENTS}
    assert all(len(v) > 0 for v in spectra.values()), "missing endmember class in library"
    return spectra, bands


def generate_synthetic(scheme, data_cfg, library=None):
    """
    Synthetic mixed spectra. Per pixel: k ~ U{1..k_max} endmember classes; abundances ~ Dirichlet(alpha),
    alpha ~ U(0.3, 1) (as in v1); mixing = linear or bilinear (Fan et al. 2009, strength gamma).
      prior='uniform' : classes drawn uniformly
      prior='matched' : classes drawn with probability proportional to their mean field cover
    Endmember variability (always on): each drawn endmember is a random convex combination of 1-2
    library spectra of its class, times a brightness factor ~ N(1, 0.05).
    Returns X (n, 2151) in [0, 1], A (n, 8) abundances over COMPONENTS, band names.
    """
    cfg = full_data_cfg(scheme, data_cfg)
    spectra, bands = library or load_library()
    rng = np.random.default_rng(cfg["seed"])
    n, C, nb = cfg["n_samples"], len(COMPONENTS), len(bands)
    gamma = cfg["gamma"] if SCHEMES[scheme]["mixing"] == "bilinear" else 0.0
    if cfg["prior"] == "matched":
        p = np.array([max(MEASURED_COMPOSITION[c], 0.02) for c in COMPONENTS]); p = p / p.sum()
    else:
        p = None
    X = np.zeros((n, nb), np.float32)
    A = np.zeros((n, C), np.float32)
    for i in range(n):
        k = int(rng.integers(1, cfg["k_max"] + 1))
        sel = rng.choice(C, size=k, replace=False, p=p)
        abund = rng.dirichlet(rng.uniform(0.3, 1.0) * np.ones(k))
        S = np.empty((k, nb), np.float32)
        for j, ci in enumerate(sel):
            pool = spectra[COMPONENTS[ci]]
            if cfg["variability"] and len(pool) >= 2 and rng.random() < 0.5:
                a, b = rng.choice(len(pool), size=2, replace=False)
                w = rng.random()
                s = w * pool[a] + (1.0 - w) * pool[b]
            else:
                s = pool[rng.integers(len(pool))]
            if cfg["variability"]:
                s = s * rng.normal(1.0, 0.05)
            S[j] = s
        mixed = abund @ S
        if gamma > 0 and k >= 2:
            for a_ in range(k):
                for b_ in range(a_ + 1, k):
                    mixed += gamma * abund[a_] * abund[b_] * S[a_] * S[b_]
        X[i] = np.clip(mixed, 0.0, 1.0)
        A[i, sel] = abund
    return X, A, bands


def targets_from_abundances(A, targets):
    """Map abundances over COMPONENTS to the scheme's target columns (Vegetation = Vegetation + Litter)."""
    col = {c: A[:, i] for i, c in enumerate(COMPONENTS)}
    t = {"frac_Vegetation": col["Vegetation"] + col["Litter"], "frac_DkCy": col["DkCy"], "frac_Lichen": col["Lichen"],
         "frac_LtCy": col["LtCy"], "frac_Moss": col["Moss"], "frac_Bare": col["Soil"] + col["Rock"],
         "frac_late_successional": col["DkCy"] + col["Lichen"] + col["Moss"], "frac_early_successional": col["LtCy"]}
    return np.stack([t[name] for name in targets], axis=1).astype(np.float32)


def load_synthetic(scheme, data_cfg, norm="none", library=None):
    """Synthetic dataset for a scheme: X (n, kept bands) after normalisation, y (n, targets), kept band names."""
    X, A, bands = generate_synthetic(scheme, data_cfg, library)
    keep = keep_band_mask(bands)
    Xn = normalize_spectra(X, norm)[:, keep]
    return Xn, targets_from_abundances(A, SCHEMES[scheme]["targets"]), [b for b, k in zip(bands, keep) if k]


def load_measured(scheme, band_cols, norm="none"):
    """Measured plot spectra: X (normalised, in the checkpoint's band order), y, sample IDs (`full`)."""
    spec = SCHEMES[scheme]
    df = pd.read_csv(DATA_DIR / "measured_mixtures.csv")   # frac_Vegetation already = Litter+Live+Dead
    full_bands = [c for c in df.columns if c.isdigit()]
    Xn = normalize_spectra(df[full_bands].values, norm)
    X = pd.DataFrame(Xn, columns=full_bands)[band_cols].values.astype(np.float32)
    df["frac_late_successional"] = df["frac_DkCy"] + df["frac_Lichen"] + df["frac_Moss"]
    df["frac_early_successional"] = df["frac_LtCy"]
    return X, df[spec["targets"]].values.astype(np.float32), df["full"].values


def outer_cv():
    return KFold(n_splits=N_OUTER, shuffle=True, random_state=OUTER_SEED)


def inner_cv():
    return KFold(n_splits=N_INNER, shuffle=True, random_state=INNER_SEED)


# ------------------------------------------------------------------ training primitives
def target_weights(y_train, weighting):
    """Per-target loss weights from TRAINING-fold label spread only (mean-normalised to 1)."""
    std = y_train.std(axis=0) + 1e-6
    w = {"plain": np.ones_like(std), "sqrt": 1.0 / np.sqrt(std), "inv": 1.0 / std}[weighting]
    return w / w.mean()


def weighted_mse(weights: torch.Tensor):
    mse = nn.MSELoss(reduction="none")
    return lambda pred, target: (mse(pred, target) * weights).mean()


def train_epoch(model, X, y, opt, loss_fn, batch_size, noise_std=0.0, scale_std=0.0):
    model.train()
    n = X.shape[0]
    total = torch.zeros((), device=X.device)
    perm = torch.randperm(n).to(X.device)
    for i in range(0, n, batch_size):
        b = perm[i:i + batch_size]
        xb, yb = X[b], y[b]
        if scale_std > 0:
            xb = xb * (1.0 + scale_std * torch.randn(xb.size(0), 1, 1, device=xb.device))
        if noise_std > 0:
            xb = xb + noise_std * torch.randn_like(xb)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.detach() * xb.size(0)
    return (total / n).item()


@torch.no_grad()
def predict(model, X, batch_size=512):
    model.eval()
    return torch.cat([model(X[i:i + batch_size]) for i in range(0, X.shape[0], batch_size)])


def per_target_metrics(y, p, targets):
    return {t: dict(r2=float(r2_score(y[:, i], p[:, i])),
                    rmse=float(np.sqrt(mean_squared_error(y[:, i], p[:, i]))),
                    bias=float(np.mean(p[:, i] - y[:, i])))
            for i, t in enumerate(targets)}


def to_tensor(X, device):
    return torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)


# ------------------------------------------------------------------ pre-training (synthetic data only)
def pretrain(scheme, data_cfg, pre_cfg, device, verbose=True, log_every=50, library=None):
    data_cfg, pre_cfg = full_data_cfg(scheme, data_cfg), {**DEFAULT_PRETRAIN_CFG, **pre_cfg}
    targets = SCHEMES[scheme]["targets"]
    X, y, band_cols = load_synthetic(scheme, data_cfg, pre_cfg["norm"], library)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=TRAIN_SEED)
    scaler = StandardScaler().fit(X_tr)
    Xt = to_tensor(scaler.transform(X_tr), device).unsqueeze(1)
    Xv = to_tensor(scaler.transform(X_va), device).unsqueeze(1)
    yt, yv = to_tensor(y_tr, device), to_tensor(y_va, device)

    set_seed(TRAIN_SEED)
    model = build_model(len(band_cols), len(targets), pre_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=pre_cfg["lr"], weight_decay=pre_cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pre_cfg["n_epochs"], eta_min=1e-6)
    loss_fn = nn.MSELoss()

    best, best_state, best_ep = np.inf, None, 0
    tr_hist, va_hist = [], []
    for ep in range(1, pre_cfg["n_epochs"] + 1):
        tr_hist.append(train_epoch(model, Xt, yt, opt, loss_fn, pre_cfg["batch_size"],
                                   noise_std=pre_cfg["noise_std"], scale_std=pre_cfg["scale_std"]))
        sched.step()
        va = loss_fn(predict(model, Xv), yv).item()
        va_hist.append(va)
        if va < best:
            best, best_ep = va, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % log_every == 0 or ep == 1):
            print(f"  [pretrain {scheme}] epoch {ep:4d}  train {tr_hist[-1]:.4e}  val {va:.4e}", flush=True)
    model.load_state_dict(best_state)
    pv = predict(model, Xv).cpu().numpy()
    return dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                band_cols=band_cols, target_cols=targets, scheme=scheme,
                data_cfg=data_cfg, pretrain_cfg=pre_cfg,
                pretrain_hash=pretrain_key(scheme, data_cfg, pre_cfg), best_epoch=best_ep,
                val_metrics=per_target_metrics(y_va, pv, targets),
                val_y=y_va, val_pred=pv, train_loss=tr_hist, val_loss=va_hist)


def pretrain_key(scheme, data_cfg, pre_cfg):
    return cfg_hash(dict(scheme=scheme, data=full_data_cfg(scheme, data_cfg), pretrain={**DEFAULT_PRETRAIN_CFG, **pre_cfg}))


def pretrain_cache_dir(smoke=False):
    return MODEL_DIR / ("pretrain_cache_smoke" if smoke else "pretrain_cache")


def get_pretrained(scheme, data_cfg, pre_cfg, device, smoke=False, verbose=False, library=None):
    """Return (checkpoint, path, was_cached); generates data + pre-trains on first use of a config."""
    path = pretrain_cache_dir(smoke) / scheme / f"{pretrain_key(scheme, data_cfg, pre_cfg)}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu"), path, True
    ckpt = pretrain(scheme, data_cfg, pre_cfg, device, verbose=verbose, library=library)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return ckpt, path, False


# ------------------------------------------------------------------ fine-tuning
def fit_finetune(pre_sd, pre_cfg, n_bands, n_targets, X_tr, y_tr, cfg, weights, device,
                 n_epochs=None, X_val=None, y_val=None):
    """
    Fine-tune from fresh pretrained weights.
      X_val given : train up to cfg[max_epochs] with early stopping on X_val loss; return the best-epoch model.
      n_epochs    : train exactly n_epochs (no selection of any kind); return the final model.
    """
    assert (n_epochs is None) != (X_val is None), "give either n_epochs or a validation set"
    set_seed(TRAIN_SEED)
    model = build_model(n_bands, n_targets, pre_cfg, dropout=cfg["dropout"]).to(device)
    model.load_state_dict(pre_sd)
    assert state_hash(model.state_dict()) == state_hash(pre_sd), "model did not start from pretrained weights"
    if cfg["freeze_conv"]:
        for p in model.conv_block.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg["lr"], weight_decay=cfg["wd"])
    if cfg["scheduler"] == "warm":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=1, eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=1e-6)
    loss_fn = weighted_mse(to_tensor(weights, device))

    limit = cfg["max_epochs"] if X_val is not None else n_epochs
    best, best_state, best_ep, since = np.inf, None, 0, 0
    tr_hist, va_hist = [], []
    for ep in range(1, limit + 1):
        tr_hist.append(train_epoch(model, X_tr, y_tr, opt, loss_fn, cfg["batch_size"], noise_std=cfg["noise_std"]))
        sched.step()
        if X_val is None:
            continue
        va = loss_fn(predict(model, X_val), y_val).item()
        va_hist.append(va)
        if va < best:
            best, best_ep, since = va, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if ep >= cfg["min_epochs"] and since >= cfg["patience"]:
            break
    if X_val is not None:
        model.load_state_dict(best_state)
        return model, best_ep, tr_hist, va_hist
    return model, limit, tr_hist, None


def measured_tensors(scheme, ckpt, device):
    """Measured data normalised and scaled exactly as the checkpoint's synthetic training data."""
    X, y, ids = load_measured(scheme, ckpt["band_cols"], ckpt["pretrain_cfg"]["norm"])
    Xt = to_tensor((X - ckpt["scaler_mean"]) / ckpt["scaler_scale"], device).unsqueeze(1)
    return X, y, ids, Xt


def zero_shot(scheme, ckpt, device):
    """Pretrained model applied to all measured samples without fine-tuning (no selection involved)."""
    X, y, ids, Xt = measured_tensors(scheme, ckpt, device)
    model = build_model(len(ckpt["band_cols"]), len(ckpt["target_cols"]), ckpt["pretrain_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return ids, y, predict(model, Xt).cpu().numpy()


def nested_finetune(scheme, ckpt, ft_cfg, device, final=DEFAULT_FINAL, verbose=False, keep_models=False):
    """
    Nested-CV fine-tuning for one hyperparameter configuration. Per outer fold:
      1. inner 5-fold on the 192 training samples: each inner model starts from fresh pretrained
         weights and is early-stopped on its own inner-val fold -> inner-val predictions (the
         selection score) and a best epoch per inner fold
      2. the 48 test samples are predicted once, according to `final`:
           ensemble         -> mean of the 5 inner models' predictions
           retrain_annealed -> re-train on all 192 for median(best epochs), cosine annealed to that count
           retrain_snapshot -> re-train on all 192 for median(best epochs) under the full-length schedule
    """
    assert final in FINAL_PROTOCOLS, final
    ft_cfg = {**DEFAULT_FINETUNE_CFG, **ft_cfg}
    pre_cfg, band_cols, targets = ckpt["pretrain_cfg"], ckpt["band_cols"], ckpt["target_cols"]
    n_b, n_t = len(band_cols), len(targets)
    pre_sd = {k: v.to(device) for k, v in ckpt["state_dict"].items()}
    X, y, ids, Xt = measured_tensors(scheme, ckpt, device)
    yt = to_tensor(y, device)
    idx = lambda a: torch.as_tensor(np.asarray(a), device=device)

    test_pred = np.full_like(y, np.nan)
    fold_of = np.full(len(y), -1)
    folds, models, curves = [], [], []
    for k, (tr, te) in enumerate(outer_cv().split(X), 1):
        assert np.intersect1d(tr, te).size == 0
        w = target_weights(y[tr], ft_cfg["weighting"])
        inner_pred = np.full((len(tr), n_t), np.nan)
        best_eps, member_preds, inner_models = [], [], []
        for j, (itr, iva) in enumerate(inner_cv().split(tr), 1):
            assert np.intersect1d(itr, iva).size == 0
            a, b = idx(tr[itr]), idx(tr[iva])
            model, best_ep, tr_h, va_h = fit_finetune(pre_sd, pre_cfg, n_b, n_t, Xt[a], yt[a], ft_cfg, w, device,
                                                      X_val=Xt[b], y_val=yt[b])
            inner_pred[iva] = predict(model, Xt[b]).cpu().numpy()
            best_eps.append(best_ep)
            curves.append(dict(fold=k, inner=j, train=tr_h, val=va_h))
            if final == "ensemble":
                member_preds.append(predict(model, Xt[idx(te)]).cpu().numpy())
                inner_models.append(model)
        assert not np.isnan(inner_pred).any()
        inner_r2 = [float(r2_score(y[tr][:, i], inner_pred[:, i])) for i in range(n_t)]
        n_ep = int(np.clip(round(float(np.median(best_eps))), ft_cfg["min_epochs"], ft_cfg["max_epochs"]))

        if final == "ensemble":
            test_pred[te] = np.mean(member_preds, axis=0)
            fold_models = inner_models
        else:
            cfg_final = ft_cfg if final == "retrain_snapshot" else {**ft_cfg, "scheduler": "cos", "max_epochs": n_ep}
            a, b = idx(tr), idx(te)
            model, _, tr_h, _ = fit_finetune(pre_sd, pre_cfg, n_b, n_t, Xt[a], yt[a], cfg_final, w, device, n_epochs=n_ep)
            test_pred[te] = predict(model, Xt[b]).cpu().numpy()
            curves.append(dict(fold=k, inner=0, train=tr_h, val=None))
            fold_models = [model]
        fold_of[te] = k
        test_r2 = [float(r2_score(y[te][:, i], test_pred[te][:, i])) for i in range(n_t)]
        folds.append(dict(fold=k, inner_r2=inner_r2, inner_r2_mean=float(np.mean(inner_r2)),
                          test_r2=test_r2, test_r2_mean=float(np.mean(test_r2)),
                          n_epochs=n_ep, inner_best_epochs=best_eps, weights=[float(v) for v in w]))
        if keep_models:
            models.append(fold_models)
        if verbose:
            print(f"  fold {k}: inner-val R2 {np.mean(inner_r2):.3f} | inner best epochs {best_eps} (median {n_ep}) "
                  f"| test R2 {np.mean(test_r2):.3f}", flush=True)

    assert (fold_of > 0).all() and not np.isnan(test_pred).any()
    pooled = per_target_metrics(y, test_pred, targets)
    return dict(scheme=scheme, targets=targets, ids=ids, y=y, test_pred=test_pred, fold_of=fold_of,
                folds=folds, objective=float(np.mean([f["inner_r2_mean"] for f in folds])),
                pooled=pooled, pooled_r2_mean=float(np.mean([m["r2"] for m in pooled.values()])),
                finetune_cfg=ft_cfg, final=final, models=models, curves=curves)


def predictions_frame(res):
    """Outer-test predictions, rows ordered fold 1..5 (same layout as the RF prediction CSVs)."""
    rows = []
    for k in range(1, N_OUTER + 1):
        sel = np.where(res["fold_of"] == k)[0]
        rows.append(pd.concat([pd.DataFrame({"full": res["ids"][sel], "fold": k}),
                               pd.DataFrame(res["y"][sel], columns=res["targets"]),
                               pd.DataFrame(res["test_pred"][sel], columns=[f"pred_{t}" for t in res["targets"]])],
                              axis=1))
    return pd.concat(rows, ignore_index=True)


def curves_frame(res):
    rows = []
    for c in res["curves"]:
        for ep, tr in enumerate(c["train"], 1):
            rows.append(dict(fold=c["fold"], inner=c["inner"], epoch=ep, train_loss=tr,
                             val_loss=c["val"][ep - 1] if c["val"] is not None else np.nan))
    return pd.DataFrame(rows)


def load_config(path):
    """best_config.json -> (data_cfg, pretrain_cfg, finetune_cfg, final) with defaults filled in."""
    with open(path) as f:
        c = json.load(f)
    return ({**DEFAULT_DATA_CFG, **c.get("data", {})},
            {**DEFAULT_PRETRAIN_CFG, **c.get("pretrain", {})},
            {**DEFAULT_FINETUNE_CFG, **c.get("finetune", {})},
            c.get("final", DEFAULT_FINAL))
