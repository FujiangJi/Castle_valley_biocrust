"""
Pre-train the 1D-CNN on synthetic mixtures for one scheme and save it as the scheme's
pretrained checkpoint (2_saved_models/pretrained_spectral_cnn_<scheme>.pt).

The synthetic data are generated on the fly from the config's `data` block (deterministic), and
the checkpoint is taken from the pretrain cache filled by 5_tune_hparams.py when available.

Usage: python 3_pretrain_model.py --scheme bilinear_4scheme [--config ../3_results/tuning/<scheme>/best_config.json]
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from cnn_common import (DEFAULT_DATA_CFG, DEFAULT_PRETRAIN_CFG, MODEL_DIR, RESULT_DIR, SCHEMES, full_data_cfg,
                        get_device, get_pretrained, load_config)

ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=sorted(SCHEMES))
ap.add_argument("--config", help="best_config.json from 5_tune_hparams.py (default: built-in defaults)")
args = ap.parse_args()

if args.config:
    data_cfg, pre_cfg, _, _ = load_config(args.config)
else:
    data_cfg, pre_cfg = dict(DEFAULT_DATA_CFG), dict(DEFAULT_PRETRAIN_CFG)
data_cfg = full_data_cfg(args.scheme, data_cfg)
device = get_device()
print(f"Scheme {args.scheme} | device {device}\n  data config     {json.dumps(data_cfg)}\n  pretrain config {json.dumps(pre_cfg)}")

ckpt, cache_path, cached = get_pretrained(args.scheme, data_cfg, pre_cfg, device, verbose=True)
print(f"{'Loaded cached' if cached else 'Trained'} pretrained model ({cache_path.name}), best epoch {ckpt['best_epoch']}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
out = MODEL_DIR / f"pretrained_spectral_cnn_{args.scheme}.pt"
torch.save(ckpt, out)

targets = ckpt["target_cols"]
pd.concat([pd.DataFrame(ckpt["val_y"], columns=targets),
           pd.DataFrame(ckpt["val_pred"], columns=[f"pred_{t}" for t in targets])], axis=1
          ).to_csv(RESULT_DIR / f"pretrained_model_estimation_{args.scheme}.csv", index=False)
pd.DataFrame({"train_loss": ckpt["train_loss"], "test_loss": ckpt["val_loss"]}
             ).to_csv(RESULT_DIR / f"pretrained_model_loss_{args.scheme}.csv", index=False)

print("\nSynthetic validation metrics (20% hold-out of synthetic data):")
for t, m in ckpt["val_metrics"].items():
    print(f"  {t:28s} R2 = {m['r2']:.3f}  RMSE = {m['rmse']:.4f}")
print(f"  mean R2 = {np.mean([m['r2'] for m in ckpt['val_metrics'].values()]):.3f}")
print(f"\nSaved: {out}")
