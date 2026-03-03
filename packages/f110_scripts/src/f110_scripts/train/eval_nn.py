#!/usr/bin/env python3
"""
eval_nn.py – Print a summary table of best validation metrics for all trained models.

Reads TFEvents logs from data/models/lightning_logs/ and reports the best
(minimum) val/loss (MSE) and val/mae achieved across all epochs for every
model version found.  The reported values correspond to the checkpoint that
ModelCheckpoint would have saved.

Usage (from the repo root with the venv active):
    python packages/f110_scripts/src/f110_scripts/train/eval_nn.py
"""

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    log_root = Path("data/models/lightning_logs")

    if not log_root.exists():
        raise FileNotFoundError(
            f"Lightning log directory not found: {log_root.resolve()}\n"
            "Run this script from the repository root."
        )

    rows: list[tuple[str, float, float]] = []

    for version_dir in sorted(log_root.glob("*/version_*")):
        model_name = version_dir.parent.name
        ea = EventAccumulator(str(version_dir))
        ea.Reload()
        try:
            best_mse = min(e.value for e in ea.Scalars("val/loss"))
            best_mae = min(e.value for e in ea.Scalars("val/mae"))
        except KeyError:
            continue
        rows.append((model_name, best_mse, best_mae))

    if not rows:
        print("No model logs found.")
        return

    col_model = max(len(r[0]) for r in rows)
    col_model = max(col_model, len("Model"))

    header = f"{'Model':<{col_model}}  {'Best val/loss (MSE)':>20}  {'Best val/mae (MAE)':>18}"
    separator = "-" * len(header)

    print(header)
    print(separator)
    for model_name, best_mse, best_mae in rows:
        print(f"{model_name:<{col_model}}  {best_mse:>20.6f}  {best_mae:>18.6f}")


if __name__ == "__main__":
    main()
