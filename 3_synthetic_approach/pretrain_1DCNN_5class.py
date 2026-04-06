import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CONFIGURATION
# ==========================================
csv_path = "../1_data/Processed_data/synthetic_mixtures.csv"
model_saved_path = "../2_saved_models/pretrained_spectral_cnn_5class.pt"
result_path = "../3_results/pretrained_model_estimation_5class.csv"
loss_path = "../3_results/pretrained_model_loss_5class.csv"
batch_size = 32
lr = 0.001
n_epoches = 600
test_size = 0.2
random_seed = 42

# Original 6 targets — will be merged to 5 after loading
raw_target_cols = ["frac_Litter", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss", "frac_Vegetation"]
# Merged targets (Litter + Vegetation combined)
target_cols = ["frac_Litter+Vegetation", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss"]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class SpectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

class SpectralCNN(nn.Module):
    def __init__(self, n_bands, n_targets):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_targets)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    bad_bands = [[1320, 1440], [1770, 1960]]
    target_wvl = np.arange(350, 2501, 1)
    exclude_indices = []
    for band_range in bad_bands:
        indices = np.where((target_wvl >= band_range[0]) & (target_wvl <= band_range[1]))[0]
        exclude_indices.extend(indices)
    exclude_indices = np.array(exclude_indices)
    df = df.drop(df.columns[exclude_indices], axis=1)

    # Merge Litter + Vegetation
    df["frac_Litter+Vegetation"] = df["frac_Litter"] + df["frac_Vegetation"]
    df = df.drop(columns=["frac_Litter", "frac_Vegetation"])

    y = df[target_cols].values
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].values
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} bands, {y.shape[1]} targets.")
    return X, y, feature_cols

def train_val_split_and_scale(X, y, val_split, random_seed):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=random_seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print("Train size:", X_train_scaled.shape[0], "Val size:", X_val_scaled.shape[0])
    return X_train_scaled, X_val_scaled, y_train, y_val, scaler

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_y, all_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += loss_fn(preds, y_batch).item() * X_batch.size(0)
            all_y.append(y_batch.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    all_y, all_pred = np.vstack(all_y), np.vstack(all_pred)
    r2_list = [r2_score(all_y[:, i], all_pred[:, i]) for i in range(all_y.shape[1])]
    rmse_list = [np.sqrt(mean_squared_error(all_y[:, i], all_pred[:, i])) for i in range(all_y.shape[1])]
    return avg_loss, r2_list, rmse_list, all_y, all_pred

def main():
    X, y, band_cols = load_data(csv_path)
    X_train, X_test, y_train, y_test, scaler = train_val_split_and_scale(X, y, test_size, random_seed)

    train_ds = SpectralDataset(X_train, y_train)
    val_ds = SpectralDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                              worker_init_fn=lambda wid: np.random.seed(42 + wid))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                            worker_init_fn=lambda wid: np.random.seed(42 + wid))

    n_bands, n_targets = X_train.shape[1], y_train.shape[1]
    model = SpectralCNN(n_bands, n_targets).to(device)
    print("Model size:", str(float(sum(p.numel() for p in model.parameters()) / 1e6)) + "M")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoches, eta_min=1e-6)

    best_val_loss, best_state = np.inf, None
    loss_all_train, loss_all_test = [], []

    print("\nStart training 1D-CNN 5class...")
    for epoch in range(1, n_epoches + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, r2_list, rmse_list, all_y, all_pred = eval_model(model, val_loader, loss_fn)
        scheduler.step()
        loss_all_train.append(train_loss)
        loss_all_test.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        r2_mean = np.mean(r2_list)
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4e} | Val: {val_loss:.4e} | R² mean: {r2_mean:.3f}")

    model.load_state_dict(best_state)
    _, r2_list, rmse_list, all_y, all_pred = eval_model(model, val_loader, loss_fn)

    pd.concat([pd.DataFrame(all_y, columns=target_cols),
               pd.DataFrame(all_pred, columns=[f"pred_{x}" for x in target_cols])], axis=1
              ).to_csv(result_path, index=False)
    pd.DataFrame(np.stack([loss_all_train, loss_all_test]).T,
                 columns=["train_loss", "test_loss"]).to_csv(loss_path, index=False)

    print("\nFinal validation metrics:")
    for name, r2, rmse in zip(target_cols, r2_list, rmse_list):
        print(f"  {name:25s}  R² = {r2:.3f},  RMSE = {rmse:.4f}")

    torch.save({"state_dict": model.state_dict(), "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_, "band_cols": band_cols,
                "target_cols": target_cols}, model_saved_path)
    print(f"\nSaved model to: {model_saved_path}")

if __name__ == "__main__":
    main()
