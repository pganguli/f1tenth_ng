"""
Evaluate a Lidar1DCNN model (float or quantized) on the dataset.
Prints Weighted SmoothL1, MAE, RMSE, and first 10 predicted vs ground truth samples.

Usage:
    python evaluate_model.py --model_path data/model_wall.pth --quantized 0
    python evaluate_model.py --model_path data/model_wall_quantized.pth --quantized 1
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as tq
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# 1. Argument parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model")
parser.add_argument("--quantized", type=int, default=0, help="1 if model is quantized")
args = parser.parse_args()

# -----------------------------
# 2. Load dataset
# -----------------------------
data = np.load("data/dataset.npz")
X = data["lidar"]
y_left = data["left_wall_dist"]
y_right = data["right_wall_dist"]

y = np.stack([y_left, y_right], axis=1)
y_log = np.log(y + 1e-3)

# Normalization
X_mean = X.mean()
X_std = X.std() + 1e-6
X = (X - X_mean) / X_std

# Angle channel
angles = np.linspace(-1.0, 1.0, X.shape[1])


def add_angle_channel(X):
    ang = np.tile(angles, (X.shape[0], 1))
    return np.stack([X, ang], axis=1)


X = add_angle_channel(X)

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y_log, dtype=torch.float32)

dataset = TensorDataset(X_t, y_t)
loader = DataLoader(dataset, batch_size=64, shuffle=False)


# -----------------------------
# 3. Model definition
# -----------------------------
class Lidar1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


# -----------------------------
# 4. Instantiate model
# -----------------------------
model = Lidar1DCNN()

if args.quantized:
    # Fuse layers
    tq.fuse_modules(model.features, [["0", "1"], ["2", "3"], ["4", "5"]], inplace=True)
    # Set quantization config
    model.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(model, inplace=True)
    # Convert to quantized model
    tq.convert(model, inplace=True)

# Load the weights
model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
model.eval()

# -----------------------------
# 5. Evaluation
# -----------------------------
base_loss = nn.SmoothL1Loss(reduction="none")
loss_weights = torch.tensor([1.0, 1.5])


def weighted_loss(pred, target):
    loss = base_loss(pred, target)
    loss = loss * loss_weights.to(loss.device)
    return loss.mean()


all_preds, all_targets = [], []
total_loss = 0.0

with torch.no_grad():
    for xb, yb in loader:
        pred = model(xb)
        loss = weighted_loss(pred, yb)
        total_loss += loss.item() * xb.size(0)
        all_preds.append(pred)
        all_targets.append(yb)

total_loss /= len(dataset)

all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Convert back from log to meters
pred_m = torch.exp(all_preds)
target_m = torch.exp(all_targets)

mae = (pred_m - target_m).abs().mean(dim=0)
rmse = ((pred_m - target_m) ** 2).mean(dim=0).sqrt()

print(f"\nWeighted SmoothL1 loss (log): {total_loss:.4f}")
print(f"MAE (m): left={mae[0]:.3f}, right={mae[1]:.3f}")
print(f"RMSE (m): left={rmse[0]:.3f}, right={rmse[1]:.3f}")

# -----------------------------
# 6. Show first 10 predictions
# -----------------------------
print("\nFirst 10 samples: Predicted vs Ground Truth (meters)")
print(
    f"{'Sample':>6} | {'Pred Left':>10} | {'True Left':>10} | {'Pred Right':>11} | {'True Right':>11}"
)
print("-" * 60)
for i in range(min(10, len(pred_m))):
    print(
        f"{i + 1:6} | "
        f"{pred_m[i, 0]:10.3f} | {target_m[i, 0]:10.3f} | "
        f"{pred_m[i, 1]:11.3f} | {target_m[i, 1]:11.3f}"
    )
