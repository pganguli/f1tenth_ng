#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Advanced training script for F1Tenth LiDAR DNN models using PyTorch Lightning.
Supports YAML configuration, advanced dataloading, and automatic tuning.
"""

import argparse
import warnings
from pathlib import Path
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from f110_planning.utils.nn_models import get_architecture

# Suppress library-level deprecation warnings for cleaner output
warnings.filterwarnings("ignore", message=".*treespec.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="lightning")


class LidarDataset(Dataset):
    """
    Advanced LiDAR Dataset with memory-resident caching.

    Attributes:
        x (np.ndarray): Normalized LiDAR scans.
        y (np.ndarray): Target values (e.g., heading, path distances).
    """

    def __init__(self, data_path: str, target_col: str) -> None:
        """
        Initializes the dataset.

        Args:
            data_path: Path to the .npz dataset file.
            target_col: Name of the column(s) to use as regression targets.
        """
        data = np.load(data_path)
        self.x = data["scans"].astype(np.float32)

        # Handle multi-target columns (e.g., "left_dist,right_dist")
        if "," in target_col:
            cols = [c.strip() for c in target_col.split(",")]
            self.y = np.stack([data[c].astype(np.float32) for c in cols], axis=1)
        else:
            self.y = data[target_col].astype(np.float32).reshape(-1, 1)

        # Normalize LiDAR ranges (typically [0, 10]m -> [0, 1])
        self.x = np.clip(self.x / 10.0, 0, 1)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple of (lidar_tensor, target_tensor).
        """
        return torch.from_numpy(self.x[idx]).unsqueeze(0), torch.from_numpy(self.y[idx])


class LidarDataModule(L.LightningDataModule):
    """
    Lightning DataModule managing dataset splitting and data loader creation.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initializes the data module.

        Args:
            config: Configuration dictionary containing data parameters.
        """
        super().__init__()
        self.cfg = config["data"]
        self.train_path = self.cfg["train_path"]
        self.target_col = self.cfg["target_col"]
        self.batch_size = self.cfg["batch_size"]

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Loads and splits the dataset.

        Args:
            stage: Current stage (fit, test, etc.).
        """
        full_dataset = LidarDataset(self.train_path, self.target_col)

        val_split = self.cfg.get("val_split", 0.2)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader:
        """Creates the training data loader."""
        if self.train_dataset is None:
            raise ValueError("train_dataset is None. Call setup() first.")
        num_workers = self.cfg.get("num_workers", 4)
        pin_memory = self.cfg.get("pin_memory", True) and torch.cuda.is_available()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.cfg.get("prefetch_factor", 2)
            if num_workers > 0
            else None,
            persistent_workers=num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the validation data loader."""
        if self.val_dataset is None:
            raise ValueError("val_dataset is None. Call setup() first.")
        num_workers = self.cfg.get("num_workers", 4)
        pin_memory = self.cfg.get("pin_memory", True) and torch.cuda.is_available()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.cfg.get("prefetch_factor", 2)
            if num_workers > 0
            else None,
            persistent_workers=num_workers > 0,
        )


class LidarLightningModule(L.LightningModule):
    """
    LightningModule wrapping the F1Tenth LiDAR neural network architectures.
    """

    def __init__(
        self,
        arch_id: int,
        task: str = "heading",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the lightning module.

        Args:
            arch_id: Unique identifier for the network architecture.
            task: Prediction task ("heading" or "wall").
            **kwargs: Hyperparameters passed to save_hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = get_architecture(arch_id, task=task)
        self.criterion = nn.MSELoss()

        self.example_input_array = torch.randn(1, 1, 1080)
        self.train_step_count = 0

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Performs a forward pass.

        Args:
            *args: Should contain the input LiDAR tensor at index 0.
            **kwargs: Additional keyword arguments.

        Returns:
            Network output tensor.
        """
        x_in = args[0] if args else kwargs.get("x")
        return self.model(x_in)

    def training_step(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Executes a training step.

        Args:
            *args: Should contain (batch, batch_idx).
            **kwargs: Additional keyword arguments.

        Returns:
            Calculated loss for the step.
        """
        batch = args[0]
        batch_idx = args[1]
        _ = batch_idx
        x_in, y_target = batch
        y_hat = self(x_in)
        loss = self.criterion(y_hat, y_target)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        opt = self.optimizers()
        if isinstance(opt, list):
            lr = opt[0].param_groups[0]["lr"]
        else:
            lr = opt.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, prog_bar=False)

        self.train_step_count += 1
        return loss

    def on_after_backward(self) -> None:
        """Logs gradient norms to monitor training stability."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False)

    def validation_step(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Executes a validation step.

        Args:
            *args: Should contain (batch, batch_idx).
            **kwargs: Additional keyword arguments.

        Returns:
            Calculated loss for the step.
        """
        batch = args[0]
        batch_idx = args[1]
        _ = batch_idx
        x_in, y_target = batch
        y_hat = self(x_in)
        loss = self.criterion(y_hat, y_target)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)

        mae = torch.mean(torch.abs(y_hat - y_target))
        self.log("val/mae", mae, prog_bar=False, on_epoch=True)

        return loss

    def on_train_batch_end(
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Logs system metrics such as GPU memory usage."""
        _ = args, kwargs
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            self.log("sys/gpu_mem_gb", mem, on_step=True, prog_bar=False)

    def on_train_epoch_end(self) -> None:
        """Logs weight histograms for architecture debugging."""
        if self.logger and hasattr(self.logger, "experiment"):
            tb_logger = self.logger.experiment
            if hasattr(tb_logger, "add_histogram"):
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        tb_logger.add_histogram(
                            f"weights/{name}", param, self.current_epoch
                        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures the optimizer and LR scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=self.hparams.lr_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


def run_single_training(config: dict[str, Any], arch_id: int) -> None:
    """Run a single training session for a specific architecture."""
    # Set matmul precision for better GPU utilization and to suppress warnings
    torch.set_float32_matmul_precision("high")

    L.seed_everything(42)

    # Setup Model
    task = "wall" if "," in config["data"]["target_col"] else "heading"
    module = LidarLightningModule(
        arch_id=arch_id,
        task=task,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        lr_patience=config["training"]["lr_patience"],
    )

    # Logger
    model_name = f"{config['data']['target_col']}_arch{arch_id}"
    logger = TensorBoardLogger("lightning_logs", name=model_name)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"data/models/checkpoints/{model_name}",
            filename="best-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=config["training"]["early_stopping_patience"],
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        devices=1,
        precision=config["training"].get("precision", "32")
        if torch.cuda.is_available()
        else "32",
        logger=logger,
        callbacks=callbacks,
        profiler=config["training"].get("profiler", None),
    )

    # Data
    dm = LidarDataModule(config)

    # 1. Automatic LR Finding
    if config["training"].get("auto_lr_find", False):
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(module, datamodule=dm)
        if lr_finder:
            suggested_lr = lr_finder.suggestion()
            if suggested_lr:
                print(f"Best LR found: {suggested_lr}")
                module.hparams.lr = suggested_lr

    # 2. Resume Support
    ckpt_path = None
    if config["training"].get("resume", True):
        last_ckpt = Path(f"data/models/checkpoints/{model_name}/last.ckpt")
        if last_ckpt.exists():
            print(f"Resuming from {last_ckpt}")
            ckpt_path = str(last_ckpt)

    # 3. Training
    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_path)

    # Save final weights
    out_path = Path("data/models")
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), out_path / f"{model_name}.pth")


def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="scripts/train/config_heading.yaml"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Hyperparameter Search across architectures if specified
    arch_ids = config.get("model", {}).get("arch_ids", [config["model"]["arch_id"]])

    for aid in arch_ids:
        print(f"\n--- Starting training for Architecture {aid} ---")
        run_single_training(config, aid)


if __name__ == "__main__":
    main()
