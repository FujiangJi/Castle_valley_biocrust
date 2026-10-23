"""
Final nested-CV fine-tuning run for one scheme with a given (tuned) configuration.

Outputs (3_results/):
  measured_data_apply_to_pretrained_model_<scheme>.csv   pretrained model, no fine-tuning (zero-shot)
  final_model_estimation_<scheme>.csv                    outer-test predictions, best single config
  final_model_loss_<scheme>.csv                          loss curves (inner folds [+ final re-train] per outer fold)
  final_model_summary_<scheme>.json                      per-target metrics, per-fold details, all configs
  nested_cv_estimation_<scheme>.csv                      copy of the per-fold-best (nested-CV) predictions
Outputs (2_saved_models/): fine_tuned_spectral_cnn_<scheme>_fold{1..5}.pt — `members` holds the state dicts
  of the models that predict that outer fold (5 inner-fold models for the ensemble protocol, 1 otherwise).

Usage: python 4_fine_tune_model.py --scheme bilinear_4scheme --config ../3_results/tuning/<scheme>/best_config.json
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cnn_common import (DEFAULT_FINAL, DEFAULT_FINETUNE_CFG, FINAL_DESCRIPTION, FINAL_PROTOCOLS, MODEL_DIR,
                        RESULT_DIR, SCHEMES, curves_frame, get_device, load_config, nested_finetune,
                        predictions_frame, per_target_metrics, zero_shot)

ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=sorted(SCHEMES))
ap.add_argument("--config", help="best_config.json from 5_tune_hparams.py (default: built-in defaults)")
ap.add_argument("--final", default=None, choices=FINAL_PROTOCOLS,
                help="default: the protocol recorded in --config, else 'ensemble'")
ap.add_argument("--tuning-dir", default=None, help="override tuning folder (default 3_results/tuning/<scheme>)")
args = ap.parse_args()

if args.config:
    _, _, ft_cfg, cfg_final = load_config(args.config)
else:
    ft_cfg, cfg_final = dict(DEFAULT_FINETUNE_CFG), DEFAULT_FINAL
final = args.final or cfg_final
device = get_device()
ckpt_path = MODEL_DIR / f"pretrained_spectral_cnn_{args.scheme}.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
targets = ckpt["target_cols"]
print(f"Scheme {args.scheme} | device {device} | final protocol: {final}")
print(f"Pretrained: {ckpt_path.name} (hash {ckpt['pretrain_hash']})\n  data {json.dumps(ckpt['data_cfg'])}\n"
      f"  pretrain {json.dumps(ckpt['pretrain_cfg'])}\nFine-tune config: {json.dumps(ft_cfg)}")

# ---- zero-shot: pretrained model applied to the measured data (no fine-tuning, no selection)
ids, y, p = zero_shot(args.scheme, ckpt, device)
pd.concat([pd.DataFrame({"full": ids}), pd.DataFrame(y, columns=targets),
           pd.DataFrame(p, columns=[f"pred_{t}" for t in targets])], axis=1
          ).to_csv(RESULT_DIR / f"measured_data_apply_to_pretrained_model_{args.scheme}.csv", index=False)
zs = per_target_metrics(y, p, targets)
print(f"Zero-shot (pretrained only) mean R2 on measured data: {np.mean([m['r2'] for m in zs.values()]):.3f}")

# ---- nested-CV fine-tuning
print("\nNested-CV fine-tuning (outer seed 123 = RF folds, inner seed 42):")
res = nested_finetune(args.scheme, ckpt, ft_cfg, device, final=final, verbose=True, keep_models=True)

predictions_frame(res).to_csv(RESULT_DIR / f"final_model_estimation_{args.scheme}.csv", index=False)
curves_frame(res).to_csv(RESULT_DIR / f"final_model_loss_{args.scheme}.csv", index=False)
for k, fold_models in enumerate(res["models"], 1):
    torch.save({"members": [{kk: v.cpu() for kk, v in m.state_dict().items()} for m in fold_models],
                "final": final, "scaler_mean": ckpt["scaler_mean"], "scaler_scale": ckpt["scaler_scale"],
                "band_cols": ckpt["band_cols"], "target_cols": targets,
                "n_epochs": res["folds"][k - 1]["n_epochs"], "inner_best_epochs": res["folds"][k - 1]["inner_best_epochs"],
                "data_cfg": ckpt["data_cfg"], "pretrain_cfg": ckpt["pretrain_cfg"], "finetune_cfg": ft_cfg},
               MODEL_DIR / f"fine_tuned_spectral_cnn_{args.scheme}_fold{k}.pt")

summary = dict(scheme=args.scheme, final=final,
               protocol="nested CV: outer KFold(5,shuffle,123) == RF; inner KFold(5,shuffle,42); " + FINAL_DESCRIPTION[final],
               data_cfg=ckpt["data_cfg"], pretrain_cfg=ckpt["pretrain_cfg"], pretrain_hash=ckpt["pretrain_hash"],
               pretrain_val_metrics=ckpt["val_metrics"], finetune_cfg=ft_cfg,
               objective_inner_val_r2=res["objective"], test_metrics=res["pooled"],
               test_r2_mean=res["pooled_r2_mean"], zero_shot_metrics=zs, folds=res["folds"])
with open(RESULT_DIR / f"final_model_summary_{args.scheme}.json", "w") as f:
    json.dump(summary, f, indent=2)

tuning_dir = Path(args.tuning_dir) if args.tuning_dir else RESULT_DIR / "tuning" / args.scheme
nested_csv = tuning_dir / "nested_cv_predictions.csv"
if nested_csv.exists():
    shutil.copy(nested_csv, RESULT_DIR / f"nested_cv_estimation_{args.scheme}.csv")

print("\nOuter-test metrics (pooled over the 5 folds, 240 samples):")
for t, m in res["pooled"].items():
    print(f"  {t:28s} R2 = {m['r2']:.3f}  RMSE = {m['rmse']:.4f}  bias = {m['bias']:+.4f}")
print(f"  mean R2 = {res['pooled_r2_mean']:.3f}   (selection objective, inner-val R2 = {res['objective']:.3f})")
print("\nAll folds complete.")
