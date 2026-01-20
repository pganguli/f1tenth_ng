import torch
import torch.nn as nn
import torch.quantization as tq


def main(data_path, model_path, quantized_path):
    # -----------------------------
    # 1. Model definition (same as training)
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

    model = Lidar1DCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # -----------------------------
    # 2. Fuse layers where possible
    # -----------------------------
    # Conv + ReLU fusing
    tq.fuse_modules(model.features, [['0','1'], ['2','3'], ['4','5']], inplace=True)

    # -----------------------------
    # 3. Set quantization config
    # -----------------------------
    model.qconfig = tq.get_default_qconfig("fbgemm")

    # Prepare and convert
    tq.prepare(model, inplace=True)

    # -----------------------------
    # 4. Calibration with dummy data
    # -----------------------------
    # Use a few samples to calibrate (replace with your dataset if possible)
    dummy_input = torch.randn(64, 2, 1080)
    with torch.no_grad():
        model(dummy_input)

    # Convert to quantized
    tq.convert(model, inplace=True)

    # -----------------------------
    # 5. Save quantized model
    # -----------------------------
    torch.save(model.state_dict(), quantized_path)
    print("Quantized model saved.")


if __name__ == "__main__":
    main(
        data_path="data/dataset.npz",
        model_path="data/model_wall.pth",
        quantized_path="data/model_wall_quant.pth",
    )
