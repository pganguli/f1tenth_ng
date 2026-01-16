import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1. Load dataset
# -----------------------------
data = np.load('lidar_dataset.npz')
X = data['X']  # shape = (num_samples, 1080)
y = data['y']  # shape = (num_samples,)

print("Dataset loaded:", X.shape, y.shape)

# -----------------------------
# 2. Preprocessing
# -----------------------------
# Normalize LiDAR distances to [0,1] using max range
max_range = 30.0  # adjust to your LiDAR max range
X = X / max_range

# Optionally, normalize y (left wall distances) if needed
# y = y / max_range

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape (N,1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# -----------------------------
# 3. Define MLP model
# -----------------------------
class LidarLeftWallNet(nn.Module):
    def __init__(self, input_dim=1080):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # output: left-wall distance
        )

    def forward(self, x):
        return self.net(x)

model = LidarLeftWallNet()
print(model)

# -----------------------------
# 4. Training setup
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50

# -----------------------------
# 5. Training loop
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        test_loss = criterion(y_pred, y_test_tensor).item()

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

# -----------------------------
# 6. Save the trained model
# -----------------------------
torch.save(model.state_dict(), "lidar_leftwall_model.pth")
print("Model saved as lidar_leftwall_model.pth")

# -----------------------------
# 7. Quick prediction example
# -----------------------------
model.eval()
with torch.no_grad():
    sample = X_test_tensor[0].unsqueeze(0)  # shape (1,1080)
    pred = model(sample).item()
    print(f"Predicted left-wall distance: {pred:.3f} m | True: {y_test[0]:.3f} m")
