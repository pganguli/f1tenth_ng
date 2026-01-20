import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def main(num_epochs, data_path, save_path):
    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    data = np.load(data_path)
    X = data["lidar"]  # (N, 1080)
    y = data["theta"]  # (N,)

    print("Dataset loaded:", X.shape)

    # -----------------------------
    # 2. Normalize theta to [-1,1]
    # -----------------------------
    y = y / np.pi  # scale radians to [-1, 1]

    # -----------------------------
    # 3. Train / test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 4. Global normalization of lidar
    # -----------------------------
    X_mean = X_train.mean()
    X_std = X_train.std() + 1e-6
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # -----------------------------
    # 5. Angle channel
    # -----------------------------
    angles = np.linspace(-1.0, 1.0, X.shape[1])

    def add_angle_channel(X):
        ang = np.tile(angles, (X.shape[0], 1))
        return np.stack([X, ang], axis=1)  # (N, 2, 1080)

    X_train = add_angle_channel(X_train)
    X_test = add_angle_channel(X_test)

    # -----------------------------
    # 6. Optional input noise (train only)
    # -----------------------------
    X_train += np.random.normal(0, 0.01, X_train.shape)

    # -----------------------------
    # 7. Convert to PyTorch tensors
    # -----------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # -----------------------------
    # 8. Define improved 1D CNN
    # -----------------------------
    class Lidar1DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=2),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3, padding=2, dilation=4),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),  # predict normalized theta
            )

        def forward(self, x):
            x = self.features(x)
            return self.regressor(x)

    model = Lidar1DCNN()
    print(model)

    # -----------------------------
    # 9. Loss and optimizer
    # -----------------------------
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # -----------------------------
    # 10. Training loop with early stopping
    # -----------------------------
    patience = 20
    best_loss = float("inf")
    patience_ctr = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t).item()

        print(
            f"Epoch {epoch + 1:03d} | Train: {train_loss:.4f} | Test: {test_loss:.4f}"
        )

        if test_loss < best_loss:
            best_loss = test_loss
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # -----------------------------
    # 11. Example inference
    # -----------------------------
    model.load_state_dict(torch.load(save_path))
    model.eval()

    with torch.no_grad():
        pred_norm = model(X_test_t[:1]).squeeze().numpy()
        pred_theta = pred_norm * np.pi  # convert back to radians
        true_theta = y_test[:1].squeeze() * np.pi

        print(f"Predicted theta: {pred_theta:.3f} rad")
        print(f"True theta:      {true_theta:.3f} rad")


if __name__ == "__main__":
    main(
        num_epochs=300,
        data_path="data/dataset.npz",
        save_path="data/model_theta.pth",
    )
