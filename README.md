# Biocrust Fractional Cover Estimation from Hyperspectral Data

Code repository for the manuscript `under preparation`: *Can hyperspectral signals capture treatment-driven ecological shifts in biocrust communities under mixed-pixel conditions?* 

## Repository Structure (will update frequently)

```
github_code/
├── 1_data_processing/
│   ├── process_measured_data_RF.py        # Merge spectra + fractional cover, compute spectral indices (RF pipeline)
│   └── process_measured_data_CNN.py       # Merge spectra + fractional cover (CNN pipeline)
│
├── 2_rf_analysis/
│   ├── RF_reflectance_only_5class.py              # RF, 5-class targets, reflectance only
│   ├── RF_reflectance_only_3class.py              # RF, 3-class targets, reflectance only
│   ├── PCA_RF_reflectance_only_5class.py          # PCA+RF, 5-class targets, reflectance only
│   ├── PCA_RF_reflectance_only_3class.py          # PCA+RF, 3-class targets, reflectance only
│   ├── RF_reflectance_indices_5class.py           # RF, 5-class, reflectance + spectral indices
│   ├── RF_reflectance_indices_3class.py           # RF, 3-class, reflectance + spectral indices
│   ├── PCA_RF_reflectance_indices_5class.py       # PCA+RF, 5-class, reflectance + spectral indices
│   ├── PCA_RF_reflectance_indices_3class.py       # PCA+RF, 3-class, reflectance + spectral indices
│   ├── RF_separate_feature_importance_3class.py   # Per-target RF with feature importance (3-class)
│   ├── RF_separate_feature_importance_5class.py   # Per-target RF with feature importance (5-class)
│   └── README.md
│
├── 3_synthetic_approach/
│   ├── generate_synthetic_datasets.py     # Generate 10,000 synthetic mixed spectra (linear and bilinear mixing)
│   ├── pretrain_1DCNN_5class.py           # Pre-train 1D-CNN on synthetic data (5-class targets)
│   ├── pretrain_1DCNN_3class.py           # Pre-train 1D-CNN on synthetic data (3-class targets)
│   ├── finetune_1DCNN_5class.py           # Fine-tune pre-trained CNN on measured data (5-class)
│   └── finetune_1DCNN_3class.py           # Fine-tune pre-trained CNN on measured data (3-class)
│
├── 4_figure_plotting/
│   ├── Figure1.py              # Study area map
│   ├── Figure2.py              # Spectral library, mixture spectra, fractional cover distributions
│   ├── Figure3.py              # Model comparison scatter plots (3-class)
│   ├── Figure4.py              # Model comparison scatter plots (5-class)
│   ├── Figure5.py              # Spectral feature importance
│   ├── Figure6.py              # Spatial heatmaps + residuals (3-class, per block)
│   ├── Figure7.py              # Spatial heatmaps + residuals (5-class, per block)
│   ├── Figure8.py              # Stacked bar charts (treatment effects, per block)
│   ├── Figure8_v2.py           # Treatment effects summary (measured vs estimated + delta from control)
│   └── Figure_workflow.py      # Methodological workflow diagram
│
└── README.md
```

## Model Configurations

### Target Variables

| Scheme | Targets | Description |
|--------|---------|-------------|
| 3-class | `frac_Litter+Vegetation`, `frac_late_successional`, `frac_early_successional` | Ecological grouping: Litter+Vegetation, Late Succ. (DkCy+Lichen+Moss), Early Succ. (LtCy) |
| 5-class | `frac_Litter+Vegetation`, `frac_DkCy`, `frac_Lichen`, `frac_LtCy`, `frac_Moss` | Individual biocrust functional types |

### Approach 1: Random Forest Regression

| Script | Pipeline | Targets | Features |
|--------|----------|---------|----------|
| `RF_reflectance_only_5class.py` | Scaler → RF | 5-class | Reflectance |
| `RF_reflectance_indices_5class.py` | Scaler → RF | 5-class | Reflectance + Indices |
| `PCA_RF_reflectance_only_5class.py` | Scaler → PCA → RF | 5-class | Reflectance |
| `PCA_RF_reflectance_indices_5class.py` | Scaler → PCA → RF | 5-class | Reflectance + Indices |
| `RF_reflectance_only_3class.py` | Scaler → RF | 3-class | Reflectance |
| `RF_reflectance_indices_3class.py` | Scaler → RF | 3-class | Reflectance + Indices |
| `PCA_RF_reflectance_only_3class.py` | Scaler → PCA → RF | 3-class | Reflectance |
| `PCA_RF_reflectance_indices_3class.py` | Scaler → PCA → RF | 3-class | Reflectance + Indices |

All RF models use nested 5-fold cross-validation with RandomizedSearchCV (60 iterations).

### Approach 2: Transfer Learning (1D-CNN)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `generate_synthetic_datasets.py` | 10,000 synthetic spectra  |
| 2a | `pretrain_1DCNN_3class.py` | Pre-train on synthetic data (600 epochs, lr=0.001) |
| 2b | `pretrain_1DCNN_5class.py` | Pre-train on synthetic data (600 epochs, lr=0.001) |
| 3a | `finetune_1DCNN_3class.py` | Fine-tune on measured data, 5-fold CV (500 epochs, lr=0.0005) |
| 3b | `finetune_1DCNN_5class.py` | Fine-tune on measured data, 5-fold CV (500 epochs, lr=0.0005) |

### Feature Importance

| Script | Targets | Outputs |
|--------|---------|---------|
| `RF_separate_feature_importance_3class.py` | 3-class | Per-band importance, regional permutation importance (50 nm bins) |
| `RF_separate_feature_importance_5class.py` | 5-class | Per-band importance, regional permutation importance (50 nm bins) |

## Output Files

### RF Analysis
- `rf_reflectance_only_{3,5}class.csv` — Predicted vs observed fractional cover
- `pca_rf_reflectance_only_{3,5}class.csv` — PCA+RF predictions
- `rf_reflectance_indices_{3,5}class.csv` — RF with spectral indices
- `pca_rf_reflectance_indices_{3,5}class.csv` — PCA+RF with spectral indices
- `rf_feature_importance_{3,5}class_predictions.csv` — Per-target predictions
- `rf_feature_importance_{3,5}class_permu_imp.csv` — Regional permutation importance
- `rf_feature_importance_{3,5}class_feature_imp.csv` — Per-band feature importance

### CNN Analysis
- `synthetic_mixtures.csv` — Generated synthetic training data
- `pretrained_model_estimation_{3,5}class.csv` — Pre-trained model predictions
- `final_model_estimation_{3,5}class.csv` — Fine-tuned model predictions
- `pretrained_spectral_cnn_{3,5}class.pt` — Saved pre-trained model weights
- `fine_tuned_spectral_cnn_{3,5}class_fold{k}.pt` — Fine-tuned model per fold

## Dependencies

```
python>=3.7
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
torch
geopandas          # Figure 1 only
pyproj             # Figure 1 only
gdal               # Figure 1 only
```

## Data

Spectral and fractional cover data from the Castle Valley long-term climate manipulation experiment (38.67°N, 109.42°W), Colorado Plateau, Utah. Coming soon!
## Citation

[To be added upon publication]
