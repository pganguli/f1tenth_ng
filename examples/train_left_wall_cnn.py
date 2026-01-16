import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load dataset
# -----------------------------
data = np.load('lidar_dataset.npz')
X = data['X']  # shape = (num_samples, 1080)
y = data['y']  # shape = (num_samples,)

print("Dataset loaded:", X.shape, y.shape)

# Normalize LiDAR distances
max_range = 30.0
X = X / max_range

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# -----------------------------
# 2. Define 1D CNN model
# -----------------------------
class Lidar1DCNN(nn.Module):
    def __init__(self, input_len=1080):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),  # preserves length
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # output shape = (batch, 128, 1)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

model = Lidar1DCNN()
print(model)

# -----------------------------
# 3. Training setup
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50

# -----------------------------
# 4. Training loop
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
# 5. Save the model
# -----------------------------
torch.save(model.state_dict(), "lidar_leftwall_cnn.pth")
print("CNN model saved as lidar_leftwall_cnn.pth")

# -----------------------------
# 6. Quick prediction
# -----------------------------
model.eval()
with torch.no_grad():
    sample = X_test_tensor[0].unsqueeze(0)  # (1,1,1080)
    pred = model(sample).item()
    print(f"Predicted left-wall distance: {pred:.3f} m | True: {y_test[0]:.3f} m")
