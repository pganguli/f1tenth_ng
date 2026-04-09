#!/usr/bin/env python3
"""Plot exploratory low-CCR per-method-lambda study outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def display_strategy_name(strategy: str) -> str:
    """Return a user-facing strategy label."""
    if strategy == "self_normalizing_momentum":
        return "SRP (Ours)"
    if strategy == "srpv2":
        return "SRPv2"
    if strategy == "never_query":
        return "Never query"
    return strategy.replace("_", " ")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot low-CCR per-method lambda exploratory outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cross-map-csv",
        type=str,
        default="data/benchmarks/low_ccr_per_method_lambda_eval_cross_map.csv",
    )
    parser.add_argument(
        "--acceptable-csv",
        type=str,
        default="data/benchmarks/low_ccr_per_method_lambda_eval_acceptable.csv",
    )
    parser.add_argument(
        "--win-counts-csv",
        type=str,
        default="data/benchmarks/low_ccr_per_method_lambda_eval_win_counts.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/paper_figures_low_ccr_per_method_lambda",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", type=str, default="png,pdf")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    """Save a figure in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_frame(path: str) -> pd.DataFrame:
    """Load a CSV into a dataframe."""
    return pd.read_csv(path)


def figure_tradeoff(cross_map_df: pd.DataFrame, output_dir: Path, formats: list[str], dpi: int) -> None:
    """Scatter mean CCR against mean max CTE."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.set_title("Held-Out Low-CCR Tradeoff")
    ax.set_xlabel("Mean Cloud Call Rate")
    ax.set_ylabel("Mean Max Cross-Track Error (m)")
    ax.grid(True, alpha=0.25)

    for row in cross_map_df.itertuples(index=False):
        color = {
            "self_normalizing_momentum": "#0f766e",
            "srpv2": "#d97706",
            "never_query": "#9ca3af",
        }.get(row.strategy, "#334155")
        marker = {
            "self_normalizing_momentum": "D",
            "srpv2": "v",
            "never_query": "+",
        }.get(row.strategy, "o")
        size = 70 if row.base_experiment == "fixed_interval_k2" else 48
        ax.scatter(row.mean_cloud_call_rate, row.mean_crosstrack_max, color=color, marker=marker, s=size)
        ax.text(
            row.mean_cloud_call_rate + 0.003,
            row.mean_crosstrack_max,
            row.base_experiment,
            fontsize=8,
            alpha=0.8,
        )

    if "k2_threshold" in cross_map_df.columns:
        threshold = float(cross_map_df["k2_threshold"].iloc[0])
        ax.axhline(threshold, color="#b45309", linestyle="--", linewidth=1.2)
        ax.text(
            float(cross_map_df["mean_cloud_call_rate"].min()),
            threshold,
            "  k2 + 1% threshold",
            color="#b45309",
            va="bottom",
        )
    save_figure(fig, output_dir, "low_ccr_tradeoff", formats, dpi)


def figure_acceptable(acceptable_df: pd.DataFrame, output_dir: Path, formats: list[str], dpi: int) -> None:
    """Bar plot of acceptable configs sorted by CCR."""
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.45 * len(acceptable_df) + 1.5)))
    ax.set_title("Configs Within 1% of fixed_interval_k2")
    ax.set_xlabel("Mean Cloud Call Rate")
    ax.set_ylabel("Experiment")
    ax.grid(axis="x", alpha=0.25)

    frame = acceptable_df.sort_values("mean_cloud_call_rate", kind="stable").copy()
    colors = [
        {
            "self_normalizing_momentum": "#0f766e",
            "srpv2": "#d97706",
            "never_query": "#9ca3af",
        }.get(strategy, "#475569")
        for strategy in frame["strategy"]
    ]
    ax.barh(frame["base_experiment"], frame["mean_cloud_call_rate"], color=colors)
    ax.invert_yaxis()
    save_figure(fig, output_dir, "acceptable_configs", formats, dpi)


def figure_win_counts(wins_df: pd.DataFrame, output_dir: Path, formats: list[str], dpi: int) -> None:
    """Bar plot of per-map win counts."""
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.45 * len(wins_df) + 1.5)))
    ax.set_title("Held-Out Maps Won")
    ax.set_xlabel("Maps Won")
    ax.set_ylabel("Experiment")
    ax.grid(axis="x", alpha=0.25)

    frame = wins_df.copy()
    colors = [
        {
            "self_normalizing_momentum": "#0f766e",
            "srpv2": "#d97706",
            "never_query": "#9ca3af",
        }.get(strategy, "#475569")
        for strategy in frame["strategy"]
    ]
    ax.barh(frame["experiment"], frame["maps_won"], color=colors)
    ax.invert_yaxis()
    save_figure(fig, output_dir, "win_counts", formats, dpi)


def figure_direct_comparison(
    cross_map_df: pd.DataFrame,
    acceptable_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Render a compact fixed_interval_k2 vs SRP comparison table."""
    baseline = cross_map_df[cross_map_df["base_experiment"] == "fixed_interval_k2"].iloc[0]
    srp_family = cross_map_df["strategy"].isin(["self_normalizing_momentum", "srpv2"])
    self_norm = acceptable_df[acceptable_df["strategy"].isin(["self_normalizing_momentum", "srpv2"])].copy()
    if self_norm.empty:
        self_norm = cross_map_df[srp_family].copy()
    self_norm = self_norm.sort_values(
        ["mean_cloud_call_rate", "mean_crosstrack_max", "experiment"],
        kind="stable",
    ).iloc[0]

    comparison = pd.DataFrame(
        [
            {
                "experiment": baseline["base_experiment"],
                "mean_ccr": round(float(baseline["mean_cloud_call_rate"]), 4),
                "mean_max_cte": round(float(baseline["mean_crosstrack_max"]), 4),
                "mean_rmse_cte": round(float(baseline["mean_crosstrack_rmse"]), 4),
                "ccr_delta_vs_k2": round(float(baseline.get("ccr_reduction_vs_k2", 0.0)), 4),
                "max_cte_delta_vs_k2": round(float(baseline.get("max_cte_delta_vs_k2", 0.0)), 4),
            },
            {
                "experiment": self_norm["base_experiment"],
                "mean_ccr": round(float(self_norm["mean_cloud_call_rate"]), 4),
                "mean_max_cte": round(float(self_norm["mean_crosstrack_max"]), 4),
                "mean_rmse_cte": round(float(self_norm["mean_crosstrack_rmse"]), 4),
                "ccr_delta_vs_k2": round(float(self_norm.get("ccr_reduction_vs_k2", 0.0)), 4),
                "max_cte_delta_vs_k2": round(float(self_norm.get("max_cte_delta_vs_k2", 0.0)), 4),
            },
        ]
    )

    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.axis("off")
    ax.set_title("fixed_interval_k2 vs SRP (Ours)")
    table = ax.table(
        cellText=comparison.values,
        colLabels=list(comparison.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    save_figure(fig, output_dir, "direct_comparison", formats, dpi)


def main() -> None:
    """Render the exploratory low-CCR figure set."""
    args = parse_args()
    cross_map_df = load_frame(args.cross_map_csv)
    acceptable_df = load_frame(args.acceptable_csv)
    wins_df = load_frame(args.win_counts_csv)
    output_dir = Path(args.output_dir)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    figure_tradeoff(cross_map_df, output_dir, formats, args.dpi)
    figure_acceptable(acceptable_df, output_dir, formats, args.dpi)
    figure_win_counts(wins_df, output_dir, formats, args.dpi)
    figure_direct_comparison(cross_map_df, acceptable_df, output_dir, formats, args.dpi)


if __name__ == "__main__":
    main()
