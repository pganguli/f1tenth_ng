#!/usr/bin/env python3
"""
Add stochastic errors to waypoint CSV files.
Supports normal, uniform, and Laplace distributions for noise.
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class NoiseConfig:
    """Configuration for stochastic noise injection."""

    mean: float = 0.0
    std: float = 0.05
    distribution: str = "normal"
    columns: str = "x_m,y_m"
    seed: Optional[int] = 42


def _get_target_indices(headers: List[str], columns: str) -> List[int]:
    """Identifies the indices of columns to which noise should be applied."""
    col_names = [c.strip() for c in headers[2].lstrip("#").split(";")]
    target_cols = [c.strip() for c in columns.split(",")]

    indices: List[int] = []
    for tc in target_cols:
        if tc in col_names:
            indices.append(col_names.index(tc))
        else:
            print(f"Warning: Column '{tc}' not found in CSV. Available: {col_names}")
    return indices


def _generate_noise_array(
    rows: int, cols: int, dist: str, mean: float, std: float
) -> np.ndarray:
    """Generates a matrix of noise matching the target data shape."""
    if dist == "normal":
        return np.random.normal(mean, std, size=(rows, cols))
    if dist == "uniform":
        return np.random.uniform(mean - std, mean + std, size=(rows, cols))
    if dist == "laplace":
        return np.random.laplace(mean, std, size=(rows, cols))

    raise ValueError(f"Unsupported distribution: {dist}")


def add_noise_to_waypoints(
    input_csv: str, output_csv: str, config: NoiseConfig
) -> None:
    """
    Reads a waypoint CSV and adds stochastic noise to specified columns.

    Args:
        input_csv: Path to the input CSV file.
        output_csv: Path to save the noisy CSV file.
        config: Configuration parameters for noise generation.
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} does not exist.")
        return

    # Read original file headers
    headers: List[str] = []
    with open(input_csv, "r", encoding="utf-8") as f:
        for _ in range(3):
            line = f.readline().strip()
            if not line:
                break
            headers.append(line)

    target_indices = _get_target_indices(headers, config.columns)
    if not target_indices:
        print("Error: No valid target columns found. Aborting.")
        return

    # Load numerical data and add noise
    data = np.loadtxt(input_csv, delimiter=";", skiprows=3)
    noise = _generate_noise_array(
        data.shape[0],
        len(target_indices),
        config.distribution,
        config.mean,
        config.std,
    )

    data_noisy = data.copy()
    for i, col_idx in enumerate(target_indices):
        data_noisy[:, col_idx] += noise[:, i]

    # Write headers and noisy data to new CSV
    with open(output_csv, "w", encoding="utf-8") as f:
        for h in headers:
            f.write(h + "\n")
        np.savetxt(f, data_noisy, delimiter="; ", fmt="%.7f")

    print(f"Successfully saved noisy waypoints to {output_csv}")
    print(
        f"Added {config.distribution} noise (mean={config.mean}, std={config.std}) "
        f"to columns: {config.columns}"
    )


def main() -> None:
    """
    Main entry point for the waypoint error injection script.
    """
    parser = argparse.ArgumentParser(
        description="Add stochastic errors to waypoint CSV files."
    )
    parser.add_argument("input_csv", help="Path to the input waypoint CSV file.")
    parser.add_argument("output_csv", help="Path to save the modified CSV file.")
    parser.add_argument(
        "--mean", type=float, default=0.0, help="Mean of the error distribution."
    )
    parser.add_argument(
        "--std",
        type=float,
        default=0.05,
        help="Standard deviation of the normal distribution (or scale for others).",
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="normal",
        choices=["normal", "uniform", "laplace"],
        help="Type of distribution to sample errors from.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="x_m,y_m",
        help="Comma-separated names of columns to add noise to (e.g., x_m,y_m,vx_mps).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    config = NoiseConfig(
        mean=args.mean,
        std=args.std,
        distribution=args.dist,
        columns=args.columns,
        seed=args.seed,
    )

    add_noise_to_waypoints(args.input_csv, args.output_csv, config)


if __name__ == "__main__":
    main()
