"""Unit tests for the training scripts."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from f110_scripts.train.train_nn import LidarDataModule, LidarDataset, LidarLightningModule


def test_lidar_dataset_normalization(tmp_path: Path) -> None:
    """Tests that LidarDataset correctly normalizes LiDAR ranges."""
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

    # 2. Test single target (left_dist)
    ds_w = LidarDataset(str(data_path), "left_dist")
    x_w, y_w = ds_w[0]
    assert x_w is not None
    assert y_w.shape == (1,)
    assert torch.allclose(y_w, torch.tensor([1.0]))


def test_model_architectures() -> None:
    """Tests that different architectures produce the expected scalar output shape."""
    for arch_id in range(1, 8):
        model = LidarLightningModule(
            arch_id=arch_id, lr=1e-3, weight_decay=1e-5, lr_patience=5
        )
        out = model(torch.randn(2, 1, 1080))
        assert out.shape == (2, 1), f"arch {arch_id}: expected (2,1) got {out.shape}"


def test_lidar_dataset_track_width_derivation(tmp_path: Path) -> None:
    """track_width is derived as left_wall_dist + right_wall_dist when absent."""
    data_path = tmp_path / "track.npz"
    left = np.array([1.0, 2.0], dtype=np.float32)
    right = np.array([3.0, 1.5], dtype=np.float32)
    np.savez(
        data_path,
        scans=np.ones((2, 1080), dtype=np.float32),
        left_wall_dist=left,
        right_wall_dist=right,
    )
    ds = LidarDataset(str(data_path), "track_width")
    _, y0 = ds[0]
    _, y1 = ds[1]
    assert torch.allclose(y0, torch.tensor([4.0]))
    assert torch.allclose(y1, torch.tensor([3.5]))


def test_training_step(mocker: Any) -> None:
    """Tests the training_step logic with optimizer mocking."""
    model = LidarLightningModule(
        arch_id=1, lr=1e-3, weight_decay=1e-5, lr_patience=5
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


def test_datamodule_setup(tmp_path: Path) -> None:
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


def test_validation_step_logs_val_loss(mocker: Any) -> None:
    """validation_step should compute a finite loss and log it under 'val/loss'."""
    model = LidarLightningModule(
        arch_id=1, lr=1e-3, weight_decay=1e-5, lr_patience=5
    )
    mocker.patch.object(model, "log")

    batch = (torch.randn(4, 1, 1080), torch.randn(4, 1))
    loss = model.validation_step(batch, 0)

    assert loss > 0
    assert not torch.isnan(loss)
    model.log.assert_any_call(  # pylint: disable=no-member
        "val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=False
    )


def test_lidar_dataset_invalid_target_key(tmp_path: Path) -> None:
    """Requesting an unknown target column should raise KeyError."""
    data_path = tmp_path / "tiny.npz"
    np.savez(data_path, scans=np.random.rand(4, 1080).astype(np.float32))

    with pytest.raises(KeyError):
        LidarDataset(str(data_path), "nonexistent_column")
