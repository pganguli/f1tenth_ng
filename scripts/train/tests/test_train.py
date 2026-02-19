"""
Unit tests for the training scripts.
"""

import numpy as np
import torch

from scripts.train.train import LidarDataModule, LidarDataset, LidarLightningModule


def test_lidar_dataset_normalization(tmp_path):
    """Tests that LidarDataset correctly normalizes LiDAR ranges and handles multi-targets."""
    data_path = tmp_path / "test_data.npz"
    # Create dummy data: 10 samples, 1080 beams
    # Some values > 10 to test clipping
    scans = np.array([[5.0] * 1080, [15.0] * 1080], dtype=np.float32)
    heading = np.array([0.5, -0.5], dtype=np.float32)
    left_dist = np.array([1.0, 2.0], dtype=np.float32)
    right_dist = np.array([3.0, 4.0], dtype=np.float32)

    np.savez(
        data_path,
        scans=scans,
        heading=heading,
        left_dist=left_dist,
        right_dist=right_dist,
    )

    # 1. Test single target (heading)
    ds_h = LidarDataset(str(data_path), "heading")
    x, y = ds_h[0]
    # Dataset divides by 10.0 and unsqueezes(0)
    assert torch.allclose(x, torch.tensor([0.5] * 1080).reshape(1, 1080))
    assert y is not None
    x1, y1 = ds_h[1]
    assert torch.allclose(
        x1, torch.tensor([1.0] * 1080).reshape(1, 1080)
    )  # 15.0 clipped to 10.0 then / 10.0
    assert y1 is not None

    # 2. Test multi-target (wall)
    ds_w = LidarDataset(str(data_path), "left_dist,right_dist")
    x_w, y_w = ds_w[0]
    assert x_w is not None
    assert y_w.shape == (2,)
    assert torch.allclose(y_w, torch.tensor([1.0, 3.0]))


def test_model_architectures():
    """Tests that different architectures produce the expected output shapes."""
    # Heading task
    model_h = LidarLightningModule(
        arch_id=1, task="heading", lr=1e-3, weight_decay=1e-5, lr_patience=5
    )
    out_h = model_h(torch.randn(2, 1, 1080))
    assert out_h.shape == (2, 1)

    # Wall task (Dual head)
    model_w = LidarLightningModule(
        arch_id=5, task="wall", lr=1e-3, weight_decay=1e-5, lr_patience=5
    )
    out_w = model_w(torch.randn(2, 1, 1080))
    assert out_w.shape == (2, 2)


def test_training_step(mocker):
    """Tests the training_step logic with optimizer mocking."""
    model = LidarLightningModule(
        arch_id=1, task="heading", lr=1e-3, weight_decay=1e-5, lr_patience=5
    )

    # Mock self.optimizers() and self.log()
    mock_opt = mocker.Mock()
    mock_opt.param_groups = [{"lr": 1e-3}]
    mocker.patch.object(model, "optimizers", return_value=mock_opt)
    mocker.patch.object(model, "log")

    batch = (torch.randn(4, 1, 1080), torch.randn(4, 1))
    loss = model.training_step(batch, 0)

    assert loss > 0
    assert not torch.isnan(loss)
    # Verify that logging was attempted
    # pylint: disable=no-member
    model.log.assert_any_call(
        "train/loss", loss, prog_bar=True, on_step=True, on_epoch=True
    )


def test_datamodule_setup(tmp_path):
    """Tests that LidarDataModule correctly splits the dataset."""
    data_path = tmp_path / "test_data.npz"
    np.savez(
        data_path,
        scans=np.random.rand(20, 1080).astype(np.float32),
        heading=np.random.rand(20).astype(np.float32),
    )

    config = {
        "data": {
            "train_path": str(data_path),
            "target_col": "heading",
            "batch_size": 4,
            "val_split": 0.25,
        }
    }

    dm = LidarDataModule(config)
    dm.setup()

    # 20 samples total, 0.25 val split -> 5 val, 15 train
    assert len(dm.train_dataset) == 15
    assert len(dm.val_dataset) == 5

    loader = dm.train_dataloader()
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert loader.batch_size == 4
