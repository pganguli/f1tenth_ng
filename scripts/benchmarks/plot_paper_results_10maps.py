#!/usr/bin/env python3
"""Generate publication-grade figures for the corrected 10-map benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


STRATEGY_COLORS = {
    "always": "#1c1917",
    "fixed_bernoulli": "#157a6e",
    "bernoulli_max_miss": "#517c2c",
    "logistic": "#6c5b9c",
    "exponential": "#bf4d74",
    "piecewise_ramp": "#b55d28",
    "deterministic": "#235fa4",
}

STRATEGY_MARKERS = {
    "always": "o",
    "fixed_bernoulli": "s",
    "bernoulli_max_miss": "D",
    "logistic": "^",
    "exponential": "v",
    "piecewise_ramp": "P",
    "deterministic": "X",
}

STRATEGY_DISPLAY = {
    "always": "Always",
    "fixed_bernoulli": "Bernoulli",
    "bernoulli_max_miss": "Bernoulli+Guard",
    "logistic": "Logistic",
    "exponential": "Exponential",
    "piecewise_ramp": "Piecewise",
    "deterministic": "Threshold",
}

PREFERRED_MAP_ORDER = [
    "Austin",
    "BrandsHatch",
    "Hockenheim",
    "MexicoCity",
    "Montreal",
    "Monza",
    "Oschersleben",
    "Shanghai",
    "Spa",
    "Spielberg",
]

SINGLE_REQUIRED_COLUMNS = {
    "map_name",
    "cloud_latency",
    "experiment",
    "strategy",
    "collision_rate",
    "collision_free_rate",
    "cloud_call_rate_mean",
    "crosstrack_rmse_m_mean",
    "in_target_ccr_band",
    "rank",
}

MULTI_REQUIRED_COLUMNS = {
    "map_name",
    "experiment",
    "collision",
    "crosstrack_rmse_delta_pct_vs_large_always",
    "large_call_reduction_pct_vs_large_always",
}

ORACLE_REQUIRED_COLUMNS = {
    "map_name",
    "crosstrack_rmse_m_mean",
}

BG = "#f6f1e8"
PANEL = "#fffaf3"
TEXT = "#1d1a17"
MUTED = "#6f655b"
GRID = "#d6cbbb"
ACCENT = "#c28a2c"
TARGET_BAND = "#efe0c7"
GOOD = "#2d8a72"
BAD = "#b5524e"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate paper-ready figures from the corrected 10-map results."
    )
    parser.add_argument(
        "--single-summary-csv",
        type=str,
        default="data/benchmarks/single_tier_paper_strategies_10maps_summary.csv",
        help="Path to the aggregated single-tier summary CSV.",
    )
    parser.add_argument(
        "--single-target-csv",
        type=str,
        default="data/benchmarks/single_tier_paper_strategies_10maps_best_target_band.csv",
        help="Path to the aggregated single-tier target-band CSV.",
    )
    parser.add_argument(
        "--multi-summary-csv",
        type=str,
        default="data/benchmarks/multi_tier_cloud_benchmark_10maps_corrected_summary.csv",
        help="Path to the corrected multi-tier summary CSV.",
    )
    parser.add_argument(
        "--oracle-summary-csv",
        type=str,
        default="",
        help="Optional path to an always-hit latency-0 oracle summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/paper_figures_10maps",
        help="Directory where figure files should be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG export DPI.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated export formats.",
    )
    parser.add_argument(
        "--target-low",
        type=float,
        default=0.55,
        help="Lower bound of the target CCR band.",
    )
    parser.add_argument(
        "--target-high",
        type=float,
        default=0.65,
        help="Upper bound of the target CCR band.",
    )
    return parser.parse_args()


def apply_theme() -> None:
    """Apply a restrained paper-oriented theme."""
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.edgecolor": "#bcae9a",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    """Ensure an input dataframe contains all required columns."""
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def ordered_maps(values: Iterable[str]) -> list[str]:
    """Return a stable, preferred map ordering for the provided set."""
    present = {str(value) for value in values}
    ordered = [name for name in PREFERRED_MAP_ORDER if name in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


def load_single_summary(path: str, target_low: float, target_high: float) -> pd.DataFrame:
    """Load, validate, and augment the single-tier summary."""
    summary = pd.read_csv(path)
    validate_columns(summary, SINGLE_REQUIRED_COLUMNS, "single-tier summary")
    summary["map_name"] = pd.Categorical(
        summary["map_name"],
        ordered_maps(summary["map_name"]),
        ordered=True,
    )
    summary["cloud_latency"] = summary["cloud_latency"].astype(float)
    summary["collision_free_rate"] = summary["collision_free_rate"].astype(float)
    summary["collision_rate"] = summary["collision_rate"].astype(float)
    summary["cloud_call_rate_mean"] = summary["cloud_call_rate_mean"].astype(float)
    summary["crosstrack_rmse_m_mean"] = summary["crosstrack_rmse_m_mean"].astype(float)
    summary["in_target_ccr_band"] = summary["cloud_call_rate_mean"].between(
        target_low,
        target_high,
    )

    baselines = (
        summary[summary["strategy"] == "always"][
            ["map_name", "cloud_latency", "crosstrack_rmse_m_mean", "cloud_call_rate_mean"]
        ]
        .rename(
            columns={
                "crosstrack_rmse_m_mean": "always_rmse",
                "cloud_call_rate_mean": "always_ccr",
            }
        )
        .copy()
    )
    summary = summary.merge(baselines, on=["map_name", "cloud_latency"], how="left")
    if summary["always_rmse"].isna().any():
        raise ValueError("single-tier summary is missing always-hit rows for one or more maps")

    summary["rmse_delta_vs_always_pct"] = 100.0 * (
        (summary["crosstrack_rmse_m_mean"] - summary["always_rmse"]) / summary["always_rmse"]
    )
    summary["ccr_delta_vs_always"] = (
        summary["cloud_call_rate_mean"] - summary["always_ccr"]
    )
    return summary.sort_values(["map_name", "cloud_latency", "rank"], kind="stable")


def load_target_band(path: str) -> pd.DataFrame:
    """Load the target-band CSV for validation/reference."""
    target = pd.read_csv(path)
    validate_columns(target, SINGLE_REQUIRED_COLUMNS, "single-tier target-band summary")
    return target


def load_multi_summary(path: str) -> pd.DataFrame:
    """Load and validate the multi-tier summary."""
    summary = pd.read_csv(path)
    validate_columns(summary, MULTI_REQUIRED_COLUMNS, "multi-tier summary")
    summary["map_name"] = pd.Categorical(
        summary["map_name"],
        ordered_maps(summary["map_name"]),
        ordered=True,
    )
    return summary.sort_values(["map_name", "collision"], kind="stable")


def load_oracle_summary(path: str) -> pd.DataFrame:
    """Load and validate the oracle latency-0 summary."""
    summary = pd.read_csv(path)
    validate_columns(summary, ORACLE_REQUIRED_COLUMNS, "oracle summary")
    summary["map_name"] = pd.Categorical(
        summary["map_name"],
        ordered_maps(summary["map_name"]),
        ordered=True,
    )
    summary["crosstrack_rmse_m_mean"] = summary["crosstrack_rmse_m_mean"].astype(float)
    return (
        summary.sort_values(["map_name", "crosstrack_rmse_m_mean"], kind="stable")
        .groupby("map_name", observed=False, as_index=False)
        .first()
        .rename(columns={"crosstrack_rmse_m_mean": "oracle_rmse"})
    )


def best_overall_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the best overall row per map."""
    return (
        summary.sort_values(["map_name", "cloud_latency", "rank"], kind="stable")
        .groupby(["map_name", "cloud_latency"], observed=False, as_index=False)
        .first()
        .sort_values(["map_name", "cloud_latency"], kind="stable")
        .reset_index(drop=True)
    )


def best_target_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the best target-band row per map."""
    target = summary[summary["in_target_ccr_band"]].copy()
    return (
        target.sort_values(["map_name", "cloud_latency", "rank"], kind="stable")
        .groupby(["map_name", "cloud_latency"], observed=False, as_index=False)
        .first()
        .sort_values(["map_name", "cloud_latency"], kind="stable")
        .reset_index(drop=True)
    )


def strategy_win_counts(summary: pd.DataFrame, target_only: bool) -> pd.Series:
    """Count strategy wins overall or inside the target band."""
    winners = best_target_rows(summary) if target_only else best_overall_rows(summary)
    return winners["strategy"].value_counts().reindex(STRATEGY_DISPLAY, fill_value=0)


def best_multi_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the best multi-tier configuration per map."""
    return (
        summary.sort_values(
            [
                "map_name",
                "collision",
                "crosstrack_rmse_delta_pct_vs_large_always",
                "large_call_reduction_pct_vs_large_always",
            ],
            ascending=[True, True, True, False],
            kind="stable",
        )
        .groupby("map_name", observed=False, as_index=False)
        .first()
        .sort_values("map_name", kind="stable")
        .reset_index(drop=True)
    )


def attach_oracle_baseline(summary: pd.DataFrame, oracle_summary: pd.DataFrame) -> pd.DataFrame:
    """Attach oracle RMSE and competitive-ratio columns to the single-tier summary."""
    enriched = summary.merge(oracle_summary[["map_name", "oracle_rmse"]], on="map_name", how="left")
    if enriched["oracle_rmse"].isna().any():
        raise ValueError("oracle summary is missing one or more maps from the single-tier summary")

    denominator = enriched["always_rmse"] - enriched["oracle_rmse"]
    enriched["oracle_gap"] = enriched["crosstrack_rmse_m_mean"] - enriched["oracle_rmse"]
    enriched["always_gap"] = denominator
    enriched["oracle_regret_mm"] = 1000.0 * enriched["oracle_gap"]
    enriched["always_gap_mm"] = 1000.0 * enriched["always_gap"]
    enriched["oracle_competitive_ratio"] = enriched["crosstrack_rmse_m_mean"] / enriched["oracle_rmse"]
    enriched["always_oracle_ratio"] = enriched["always_rmse"] / enriched["oracle_rmse"]
    enriched["normalized_gap_vs_oracle"] = np.where(
        np.abs(enriched["always_gap"]) > 1e-12,
        enriched["oracle_gap"] / enriched["always_gap"],
        np.nan,
    )
    return enriched


def short_label(strategy: str) -> str:
    """Return a compact strategy label."""
    return STRATEGY_DISPLAY[strategy]


def short_experiment_label(record: pd.Series) -> str:
    """Return a compact experiment label for annotations."""
    strategy = record["strategy"]
    name = str(record["experiment"])
    if strategy == "always":
        return "Always"
    if strategy == "fixed_bernoulli":
        return f"p={record['cloud_call_rate_mean']:.2f}"
    if strategy == "bernoulli_max_miss" and "_p" in name and "_m" in name:
        tail = name.split("_p", maxsplit=1)[-1]
        probability, miss = tail.split("_m", maxsplit=1)
        return f"p={probability.replace('p', '.')} m={miss}"
    if strategy == "deterministic" and "_t" in name:
        return f"t={name.split('_t', maxsplit=1)[-1].replace('p', '.')}"
    if strategy == "piecewise_ramp":
        tail = name.split("piecewise_ramp_", maxsplit=1)[-1]
        low, high = tail.split("_", maxsplit=1)
        return f"{low.replace('p', '.')} to {high.replace('p', '.')}"
    if strategy == "logistic":
        tail = name.split("logistic_", maxsplit=1)[-1]
        center, slope = tail.split("_s", maxsplit=1)
        return f"{center.replace('c', '').replace('p', '.')} / {slope}"
    if strategy == "exponential":
        tail = name.split("exponential_", maxsplit=1)[-1]
        center, rate = tail.split("_r", maxsplit=1)
        return f"{center.replace('c', '').replace('p', '.')} / {rate}"
    return name


def add_title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Add a consistent title/subtitle block."""
    fig.text(0.06, 0.965, title, ha="left", va="top", fontsize=20, fontweight="bold")
    fig.text(0.06, 0.933, subtitle, ha="left", va="top", fontsize=11, color=MUTED)


def save_figure_formats(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Save a figure in the requested formats."""
    saved: list[Path] = []
    for file_format in formats:
        out_path = out_dir / f"{stem}.{file_format}"
        if file_format == "png":
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight")
        saved.append(out_path)
    plt.close(fig)
    return saved


def family_frontier_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the lower-envelope frontier rows for each map and strategy family."""
    metric = (
        "oracle_competitive_ratio"
        if "oracle_competitive_ratio" in summary.columns
        else "normalized_gap_vs_oracle"
    )
    frontier_rows: list[pd.DataFrame] = []
    for (map_name, strategy), group in summary.groupby(
        ["map_name", "strategy"], observed=False, sort=False
    ):
        del map_name
        ordered = group.sort_values("cloud_call_rate_mean", kind="stable")
        best_so_far = np.inf
        keep_indices: list[int] = []
        for row_index, row in ordered.iterrows():
            value = float(row[metric])
            if value < best_so_far - 1e-9:
                keep_indices.append(row_index)
                best_so_far = value
        frontier_rows.append(ordered.loc[keep_indices])
    return pd.concat(frontier_rows, axis=0).reset_index(drop=True)


def figure_pareto_rmse(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    target_low: float,
    target_high: float,
) -> list[Path]:
    """Plot the main accuracy-vs-communication frontier."""
    best = best_overall_rows(summary)
    target = best_target_rows(summary)
    map_order = list(summary["map_name"].cat.categories)
    fig, axes = plt.subplots(5, 2, figsize=(14.5, 16.5), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    y_min = summary["crosstrack_rmse_m_mean"].min() - 0.005
    y_max = summary["crosstrack_rmse_m_mean"].max() + 0.01
    x_min = max(0.5, summary["cloud_call_rate_mean"].min() - 0.02)
    x_max = min(1.01, summary["cloud_call_rate_mean"].max() + 0.01)

    add_title_block(
        fig,
        "Figure 1. Pareto Frontier: Accuracy vs Communication",
        "Every configuration is shown. Gold stars mark best overall; outlined squares mark the best 55-65% CCR pick.",
    )

    for axis, map_name in zip(axes_flat, map_order):
        subset = summary[summary["map_name"] == map_name]
        overall = best[best["map_name"] == map_name].iloc[0]
        band = target[target["map_name"] == map_name].iloc[0]
        always = subset[subset["strategy"] == "always"].iloc[0]

        axis.set_facecolor(PANEL)
        axis.axvspan(target_low, target_high, color=TARGET_BAND, alpha=0.85, zorder=0)
        axis.grid(alpha=0.35, linewidth=0.7)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_title(map_name, fontsize=11.5, fontweight="bold", pad=8)

        for _, row in subset.iterrows():
            color = STRATEGY_COLORS[row["strategy"]]
            axis.scatter(
                row["cloud_call_rate_mean"],
                row["crosstrack_rmse_m_mean"],
                s=46,
                marker=STRATEGY_MARKERS[row["strategy"]],
                c=[mpl.colors.to_rgba(color, 0.32)],
                edgecolors="none",
                zorder=2,
            )

        axis.scatter(
            always["cloud_call_rate_mean"],
            always["crosstrack_rmse_m_mean"],
            s=110,
            marker="o",
            c=[STRATEGY_COLORS["always"]],
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        axis.scatter(
            overall["cloud_call_rate_mean"],
            overall["crosstrack_rmse_m_mean"],
            s=210,
            marker="*",
            c=[ACCENT],
            edgecolors=TEXT,
            linewidths=0.9,
            zorder=7,
        )
        axis.scatter(
            band["cloud_call_rate_mean"],
            band["crosstrack_rmse_m_mean"],
            s=130,
            marker="s",
            facecolors="none",
            edgecolors=GOOD,
            linewidths=2.0,
            zorder=8,
        )

        axis.text(
            overall["cloud_call_rate_mean"] + 0.008,
            overall["crosstrack_rmse_m_mean"] - 0.0016,
            short_label(overall["strategy"]),
            fontsize=8.8,
            fontweight="bold",
            color=TEXT,
        )
        axis.text(
            band["cloud_call_rate_mean"] + 0.008,
            band["crosstrack_rmse_m_mean"] + 0.0012,
            f"Target: {short_label(band['strategy'])}",
            fontsize=8.1,
            color=GOOD,
        )

    for axis in axes_flat[:8]:
        axis.set_xlabel("")
    for axis in axes[:, 0]:
        axis.set_ylabel("Cross-Track RMSE (m)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Cloud Call Rate")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=STRATEGY_MARKERS[strategy],
            color="w",
            markerfacecolor=STRATEGY_COLORS[strategy],
            markeredgecolor="none",
            markersize=7.5,
            label=label,
        )
        for strategy, label in STRATEGY_DISPLAY.items()
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor=ACCENT,
                markeredgecolor=TEXT,
                markersize=11,
                label="Best overall",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color=GOOD,
                markerfacecolor="none",
                markeredgecolor=GOOD,
                markersize=8.5,
                linewidth=0,
                label="Best in target band",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=9.3,
    )
    fig.tight_layout(rect=(0.04, 0.05, 0.995, 0.92))
    return save_figure_formats(fig, out_dir, "figure1_pareto_rmse", formats, dpi)


def figure_normalized_family_frontiers(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    target_low: float,
    target_high: float,
) -> list[Path]:
    """Plot family-level frontiers using competitive ratio against the latency-0 baseline."""
    frontier = family_frontier_rows(summary)
    best = best_overall_rows(summary)
    target = best_target_rows(summary)
    map_order = list(summary["map_name"].cat.categories)
    fig, axes = plt.subplots(5, 2, figsize=(14.6, 16.5), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    y_min = min(0.97, float(summary["oracle_competitive_ratio"].min()) - 0.015)
    y_max = max(1.08, float(summary["oracle_competitive_ratio"].max()) + 0.02)

    add_title_block(
        fig,
        "Family Frontiers Relative To The Latency-0 Baseline",
        "Y-axis is the competitive ratio RMSE / oracle-RMSE. 1.0 is the latency-0 always-hit reference; values below 1 mean the latency-5 run beat that reference on that map.",
    )

    for axis, map_name in zip(axes_flat, map_order):
        subset = summary[summary["map_name"] == map_name]
        subset_frontier = frontier[frontier["map_name"] == map_name]
        overall = best[best["map_name"] == map_name].iloc[0]
        band = target[target["map_name"] == map_name].iloc[0]
        always_ratio = float(subset["always_oracle_ratio"].iloc[0])

        axis.set_facecolor(PANEL)
        axis.axvspan(target_low, target_high, color=TARGET_BAND, alpha=0.8, zorder=0)
        axis.axhline(1.0, color=GOOD, linewidth=1.0, linestyle="--", alpha=0.9)
        axis.axhline(always_ratio, color="#9b8d7b", linewidth=1.0, linestyle="--", alpha=0.9)
        axis.grid(alpha=0.3)
        axis.set_xlim(0.5, 1.01)
        axis.set_ylim(y_min, y_max)
        axis.set_title(map_name, fontsize=11.5, fontweight="bold", pad=8)

        for strategy, group in subset_frontier.groupby("strategy", sort=False):
            ordered = group.sort_values("cloud_call_rate_mean", kind="stable")
            axis.plot(
                ordered["cloud_call_rate_mean"],
                ordered["oracle_competitive_ratio"],
                color=STRATEGY_COLORS[strategy],
                linewidth=1.7,
                alpha=0.95,
                zorder=3,
            )
            axis.scatter(
                ordered["cloud_call_rate_mean"],
                ordered["oracle_competitive_ratio"],
                s=60,
                marker=STRATEGY_MARKERS[strategy],
                c=[STRATEGY_COLORS[strategy]],
                edgecolors=PANEL,
                linewidths=0.5,
                zorder=4,
            )

        axis.scatter(
            overall["cloud_call_rate_mean"],
            overall["oracle_competitive_ratio"],
            s=210,
            marker="*",
            c=[ACCENT],
            edgecolors=TEXT,
            linewidths=0.9,
            zorder=7,
        )
        axis.scatter(
            band["cloud_call_rate_mean"],
            band["oracle_competitive_ratio"],
            s=130,
            marker="s",
            facecolors="none",
            edgecolors=GOOD,
            linewidths=2.0,
            zorder=8,
        )
        axis.text(
            0.03,
            0.03,
            "oracle x1.00",
            fontsize=8.1,
            color=GOOD,
            ha="left",
            va="bottom",
            transform=axis.get_yaxis_transform(),
        )
        axis.text(
            0.03,
            0.93,
            f"always@5 x{always_ratio:.3f}",
            fontsize=8.1,
            color=MUTED,
            ha="left",
            va="top",
            transform=axis.get_yaxis_transform(),
        )
        axis.text(
            band["cloud_call_rate_mean"] + 0.01,
            band["oracle_competitive_ratio"] + 0.004,
            f"Target: {short_label(band['strategy'])}",
            fontsize=8.1,
            color=GOOD,
        )

    for axis in axes[:, 0]:
        axis.set_ylabel("Competitive Ratio vs Oracle")
    for axis in axes[-1, :]:
        axis.set_xlabel("Cloud Call Rate")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STRATEGY_COLORS[strategy],
            marker=STRATEGY_MARKERS[strategy],
            markerfacecolor=STRATEGY_COLORS[strategy],
            markeredgecolor=PANEL,
            linewidth=1.7,
            markersize=7.0,
            label=label,
        )
        for strategy, label in STRATEGY_DISPLAY.items()
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor=ACCENT,
                markeredgecolor=TEXT,
                markersize=11,
                label="Best overall",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color=GOOD,
                markerfacecolor="none",
                markeredgecolor=GOOD,
                markersize=8.5,
                linewidth=0,
                label="Best in target band",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0.04, 0.05, 0.995, 0.92))
    return save_figure_formats(fig, out_dir, "normalized_family_frontiers", formats, dpi)


def figure_pareto_safety(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    target_low: float,
    target_high: float,
) -> list[Path]:
    """Plot the safety-vs-communication tradeoff, including saturated panels."""
    map_order = list(summary["map_name"].cat.categories)
    fig, axes = plt.subplots(5, 2, figsize=(14.5, 15.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    add_title_block(
        fig,
        "Figure 2. Safety vs Communication",
        "On this corrected 10-map subset, safety is saturated: every map is collision-free across the full single-tier sweep.",
    )

    for axis, map_name in zip(axes_flat, map_order):
        subset = summary[summary["map_name"] == map_name]
        axis.set_facecolor(PANEL)
        axis.axvspan(target_low, target_high, color=TARGET_BAND, alpha=0.85, zorder=0)
        axis.grid(alpha=0.3, linewidth=0.7)
        axis.set_xlim(0.5, 1.01)
        axis.set_ylim(0.95, 1.01)
        axis.set_title(map_name, fontsize=11.5, fontweight="bold", pad=8)

        for _, row in subset.iterrows():
            axis.scatter(
                row["cloud_call_rate_mean"],
                row["collision_free_rate"],
                s=34,
                marker=STRATEGY_MARKERS[row["strategy"]],
                c=[mpl.colors.to_rgba(STRATEGY_COLORS[row["strategy"]], 0.3)],
                edgecolors="none",
                zorder=3,
            )

        axis.axhline(1.0, color=GOOD, linewidth=1.2, linestyle="--", alpha=0.9)
        axis.text(
            0.51,
            0.952,
            "All 24 configs collision-free",
            ha="left",
            va="bottom",
            fontsize=8.4,
            color=MUTED,
        )

    for axis in axes[:, 0]:
        axis.set_ylabel("Collision-Free Rate")
    for axis in axes[-1, :]:
        axis.set_xlabel("Cloud Call Rate")

    fig.tight_layout(rect=(0.04, 0.05, 0.995, 0.92))
    return save_figure_formats(fig, out_dir, "figure2_pareto_safety", formats, dpi)


def figure_target_band_matrix(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot the best target-band winner per map."""
    target = best_target_rows(summary).copy()
    target["display_label"] = target["strategy"].map(short_label)
    target = target.sort_values("map_name", kind="stable")

    fig, ax = plt.subplots(figsize=(13.5, 8.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    add_title_block(
        fig,
        "Figure 3. Best 55-65% CCR Configuration Per Map",
        "Bar length shows RMSE change vs always-hit. Negative values are better than the always-hit baseline.",
    )

    y_positions = np.arange(len(target))
    values = target["rmse_delta_vs_always_pct"].to_numpy()
    colors = [STRATEGY_COLORS[strategy] for strategy in target["strategy"]]

    ax.axvline(0.0, color=TEXT, linewidth=1.0, alpha=0.9)
    bars = ax.barh(
        y_positions,
        values,
        color=colors,
        height=0.7,
        edgecolor=TEXT,
        linewidth=0.6,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(target["map_name"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE Delta vs Always-Hit (%)")
    ax.grid(axis="x", alpha=0.3)

    for bar, (_, row) in zip(bars, target.iterrows()):
        x_text = bar.get_width() + (0.12 if bar.get_width() >= 0 else -0.12)
        alignment = "left" if bar.get_width() >= 0 else "right"
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2.0,
            (
                f"{short_label(row['strategy'])}  |  "
                f"CCR {row['cloud_call_rate_mean']:.3f}  |  "
                f"RMSE {row['crosstrack_rmse_m_mean']:.3f}"
            ),
            ha=alignment,
            va="center",
            fontsize=8.9,
            color=TEXT,
        )

    fig.tight_layout(rect=(0.055, 0.05, 0.99, 0.92))
    return save_figure_formats(fig, out_dir, "figure3_target_band_winners", formats, dpi)


def figure_target_band_family_matrix(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot the best in-band configuration for each family and map."""
    families = [
        "fixed_bernoulli",
        "bernoulli_max_miss",
        "logistic",
        "exponential",
        "piecewise_ramp",
        "deterministic",
    ]
    map_order = list(summary["map_name"].cat.categories)
    target = summary[summary["in_target_ccr_band"]].copy()
    family_best = (
        target.sort_values(["map_name", "strategy", "rank"], kind="stable")
        .groupby(["map_name", "strategy"], observed=False, as_index=False)
        .first()
    )

    fig, ax = plt.subplots(figsize=(13.4, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, len(families))
    ax.set_ylim(0, len(map_order))
    ax.axis("off")

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "oracle_gap",
        ["#2d8a72", "#efe7d7", "#b5524e"],
    )

    add_title_block(
        fig,
        "Target-Band Method Comparison By Family",
        "Cell color is the competitive ratio RMSE / oracle-RMSE. Blank cells mean that family never entered the 55-65% CCR band on that map.",
    )

    ratio_min = float(target["oracle_competitive_ratio"].min())
    ratio_max = float(target["oracle_competitive_ratio"].max())
    norm = mpl.colors.TwoSlopeNorm(vmin=min(ratio_min, 0.98), vcenter=1.0, vmax=max(ratio_max, 1.04))

    for col_idx, family in enumerate(families):
        ax.text(
            col_idx + 0.5,
            len(map_order) + 0.08,
            short_label(family),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    for row_idx, map_name in enumerate(map_order):
        ax.text(
            -0.08,
            len(map_order) - row_idx - 0.5,
            map_name,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        y_base = len(map_order) - row_idx - 1
        for col_idx, family in enumerate(families):
            match = family_best[
                (family_best["map_name"] == map_name)
                & (family_best["strategy"] == family)
            ]
            rect = mpl.patches.Rectangle(
                (col_idx + 0.05, y_base + 0.08),
                0.9,
                0.82,
                facecolor="#ebe2d4",
                edgecolor="#b9ab98",
                linewidth=0.9,
            )
            ax.add_patch(rect)
            if match.empty:
                ax.text(
                    col_idx + 0.5,
                    y_base + 0.49,
                    "No\nin-band\nconfig",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=MUTED,
                )
                continue
            row = match.iloc[0]
            value = float(row["oracle_competitive_ratio"])
            rect.set_facecolor(cmap(norm(value)))
            ax.text(
                col_idx + 0.5,
                y_base + 0.63,
                f"x{value:.3f}",
                ha="center",
                va="center",
                fontsize=9.1,
                fontweight="bold",
                color=TEXT,
            )
            ax.text(
                col_idx + 0.5,
                y_base + 0.45,
                f"CCR {row['cloud_call_rate_mean']:.3f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=TEXT,
            )
            ax.text(
                col_idx + 0.5,
                y_base + 0.28,
                short_experiment_label(row),
                ha="center",
                va="center",
                fontsize=7.8,
                color=MUTED,
            )

    fig.tight_layout(rect=(0.08, 0.05, 0.995, 0.92))
    return save_figure_formats(fig, out_dir, "target_band_family_matrix", formats, dpi)


def figure_overall_vs_target(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Compare best overall vs best target-band config for each map."""
    overall = best_overall_rows(summary).copy()
    target = best_target_rows(summary).copy()
    merged = overall.merge(
        target,
        on=["map_name", "cloud_latency"],
        suffixes=("_overall", "_target"),
    ).sort_values("map_name", kind="stable")

    fig, axes = plt.subplots(5, 2, figsize=(14.2, 15.8), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    add_title_block(
        fig,
        "Figure 4. Best Overall vs Best Target-Band",
        "Each panel connects the unconstrained accuracy winner to the best 55-65% CCR configuration.",
    )

    x_min = min(
        merged["cloud_call_rate_mean_overall"].min(),
        merged["cloud_call_rate_mean_target"].min(),
    ) - 0.03
    x_max = max(
        merged["cloud_call_rate_mean_overall"].max(),
        merged["cloud_call_rate_mean_target"].max(),
    ) + 0.03
    y_min = min(merged["crosstrack_rmse_m_mean_overall"].min(), merged["crosstrack_rmse_m_mean_target"].min()) - 0.004
    y_max = max(merged["crosstrack_rmse_m_mean_overall"].max(), merged["crosstrack_rmse_m_mean_target"].max()) + 0.008

    for axis, (_, row) in zip(axes_flat, merged.iterrows()):
        axis.set_facecolor(PANEL)
        axis.grid(alpha=0.3, linewidth=0.7)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_title(str(row["map_name"]), fontsize=11.5, fontweight="bold", pad=8)

        axis.plot(
            [row["cloud_call_rate_mean_overall"], row["cloud_call_rate_mean_target"]],
            [row["crosstrack_rmse_m_mean_overall"], row["crosstrack_rmse_m_mean_target"]],
            color="#9c8a76",
            linewidth=1.4,
            zorder=2,
        )
        axis.scatter(
            row["cloud_call_rate_mean_overall"],
            row["crosstrack_rmse_m_mean_overall"],
            s=130,
            marker=STRATEGY_MARKERS[row["strategy_overall"]],
            c=[STRATEGY_COLORS[row["strategy_overall"]]],
            edgecolors=TEXT,
            linewidths=0.8,
            zorder=4,
        )
        axis.scatter(
            row["cloud_call_rate_mean_target"],
            row["crosstrack_rmse_m_mean_target"],
            s=130,
            marker=STRATEGY_MARKERS[row["strategy_target"]],
            c=[mpl.colors.to_rgba(STRATEGY_COLORS[row["strategy_target"]], 0.3)],
            edgecolors=GOOD,
            linewidths=1.6,
            zorder=5,
        )
        if row["strategy_overall"] != row["strategy_target"]:
            axis.text(
                row["cloud_call_rate_mean_target"] + 0.008,
                row["crosstrack_rmse_m_mean_target"] + 0.001,
                f"{short_label(row['strategy_overall'])} → {short_label(row['strategy_target'])}",
                fontsize=8.3,
                color=MUTED,
            )

    for axis in axes[:, 0]:
        axis.set_ylabel("Cross-Track RMSE (m)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Cloud Call Rate")

    fig.tight_layout(rect=(0.04, 0.05, 0.995, 0.92))
    return save_figure_formats(fig, out_dir, "figure4_overall_vs_target", formats, dpi)


def figure_roofline_oracle(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot a roofline-style comparison using competitive ratio against the oracle."""
    overall = best_overall_rows(summary).copy()
    target = best_target_rows(summary).copy()
    merged = overall.merge(
        target,
        on=["map_name", "cloud_latency"],
        suffixes=("_overall", "_target"),
    ).sort_values("map_name", kind="stable")

    fig, ax = plt.subplots(figsize=(12.8, 8.7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    add_title_block(
        fig,
        "Roofline Comparison Against The Oracle Floor",
        "Each row starts at the latency-0 always-hit reference with ratio 1.0. The second anchor is realistic always-hit at latency 5, so the overlaid methods show how much of that gap they recover.",
    )

    y_positions = np.arange(len(merged))
    ax.axvline(1.0, color=GOOD, linewidth=1.1, linestyle="--")
    ax.grid(axis="x", alpha=0.28)

    for y_pos, (_, row) in enumerate(merged.iterrows()):
        always_ratio = float(row["always_oracle_ratio_overall"])
        line_start = min(1.0, always_ratio)
        line_end = max(1.0, always_ratio)
        ax.plot([line_start, line_end], [y_pos, y_pos], color="#cfbfab", linewidth=1.5, zorder=1)
        ax.scatter(1.0, y_pos, s=58, c=[GOOD], edgecolors=PANEL, linewidths=0.5, zorder=4)
        ax.scatter(always_ratio, y_pos, s=58, c=["#7d7167"], edgecolors=PANEL, linewidths=0.5, zorder=4)
        ax.scatter(
            row["oracle_competitive_ratio_overall"],
            y_pos,
            s=150,
            marker=STRATEGY_MARKERS[row["strategy_overall"]],
            c=[ACCENT],
            edgecolors=TEXT,
            linewidths=0.8,
            zorder=6,
        )
        ax.scatter(
            row["oracle_competitive_ratio_target"],
            y_pos,
            s=130,
            marker=STRATEGY_MARKERS[row["strategy_target"]],
            c=[mpl.colors.to_rgba(STRATEGY_COLORS[row["strategy_target"]], 0.28)],
            edgecolors=GOOD,
            linewidths=1.8,
            zorder=7,
        )
        ax.text(
            max(always_ratio, row["oracle_competitive_ratio_overall"], row["oracle_competitive_ratio_target"]) + 0.004,
            y_pos,
            (
                f"{short_label(row['strategy_overall'])} / "
                f"{short_label(row['strategy_target'])}"
            ),
            va="center",
            fontsize=8.5,
            color=TEXT,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(merged["map_name"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Competitive Ratio vs Latency-0 Always-Hit")
    x_min = min(
        0.97,
        float(merged["oracle_competitive_ratio_overall"].min()) - 0.01,
        float(merged["oracle_competitive_ratio_target"].min()) - 0.01,
        float(merged["always_oracle_ratio_overall"].min()) - 0.01,
    )
    x_max = max(
        1.03,
        float(merged["oracle_competitive_ratio_overall"].max()) + 0.015,
        float(merged["oracle_competitive_ratio_target"].max()) + 0.015,
        float(merged["always_oracle_ratio_overall"].max()) + 0.015,
    )
    ax.set_xlim(x_min, x_max)
    fig.tight_layout(rect=(0.07, 0.05, 0.995, 0.92))
    return save_figure_formats(fig, out_dir, "oracle_roofline_dumbbell", formats, dpi)


def figure_strategy_wins(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot strategy win counts overall and inside the target band."""
    overall = strategy_win_counts(summary, target_only=False)
    target = strategy_win_counts(summary, target_only=True)
    best = best_overall_rows(summary)
    band = best_target_rows(summary)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=False)
    fig.patch.set_facecolor(BG)
    add_title_block(
        fig,
        "Figure 5. Which Strategy Wins?",
        "Top panel counts best-overall wins. Bottom panel counts wins inside the 55-65% CCR target band.",
    )

    for axis, counts, rows, title in [
        (axes[0], overall, best, "Best Overall"),
        (axes[1], target, band, "Best In 55-65% CCR Band"),
    ]:
        axis.set_facecolor(PANEL)
        labels = [STRATEGY_DISPLAY[key] for key in counts.index]
        colors = [STRATEGY_COLORS[key] for key in counts.index]
        ypos = np.arange(len(labels))
        axis.barh(ypos, counts.to_numpy(), color=colors, edgecolor=TEXT, linewidth=0.6)
        axis.set_yticks(ypos)
        axis.set_yticklabels(labels, fontsize=10)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.28)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        for idx, key in enumerate(counts.index):
            winners = rows[rows["strategy"] == key]["map_name"].astype(str).tolist()
            axis.text(
                counts.iloc[idx] + 0.08,
                idx,
                ", ".join(winners) if winners else "—",
                va="center",
                fontsize=8.3,
                color=MUTED,
            )

    axes[1].set_xlabel("Number of Maps Won")
    fig.tight_layout(rect=(0.05, 0.05, 0.995, 0.92))
    return save_figure_formats(fig, out_dir, "figure5_strategy_wins", formats, dpi)


def figure_multi_tier_tradeoff(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot the corrected multi-tier tradeoff."""
    best = best_multi_rows(summary)

    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    add_title_block(
        fig,
        "Figure 6. Corrected Multi-Tier Trade-Off",
        "Each point is the best corrected multi-tier configuration per map. Lower is better on the y-axis; farther right is better on the x-axis.",
    )

    ax.axhline(0.0, color="#9b8d7b", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#9b8d7b", linewidth=1.0, linestyle="--")
    ax.grid(alpha=0.3)

    for _, row in best.iterrows():
        color = GOOD if row["collision"] == 0.0 else BAD
        ax.scatter(
            row["large_call_reduction_pct_vs_large_always"],
            row["crosstrack_rmse_delta_pct_vs_large_always"],
            s=90,
            c=[color],
            edgecolors=TEXT,
            linewidths=0.8,
            zorder=4,
        )
        ax.text(
            row["large_call_reduction_pct_vs_large_always"] + 0.55,
            row["crosstrack_rmse_delta_pct_vs_large_always"] + 0.18,
            f"{row['map_name']} · {row['experiment']}",
            fontsize=8.7,
            color=TEXT,
        )

    ax.set_xlabel("Large-Cloud Call Reduction vs Large-Always (%)")
    ax.set_ylabel("RMSE Delta vs Large-Always (%)")
    fig.tight_layout(rect=(0.05, 0.05, 0.995, 0.92))
    return save_figure_formats(fig, out_dir, "figure6_multi_tier_tradeoff", formats, dpi)


def run(
    single_summary_csv: str,
    single_target_csv: str,
    multi_summary_csv: str,
    oracle_summary_csv: str | None,
    output_dir: str,
    dpi: int,
    formats: list[str],
    target_low: float,
    target_high: float,
) -> list[Path]:
    """Generate the full paper figure set."""
    apply_theme()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    single_summary = load_single_summary(single_summary_csv, target_low, target_high)
    target_summary = load_target_band(single_target_csv)
    multi_summary = load_multi_summary(multi_summary_csv)

    target_rows = best_target_rows(single_summary)
    expected_target_rows = target_summary.sort_values(
        ["map_name", "cloud_latency", "rank"], kind="stable"
    ).reset_index(drop=True)
    if len(target_rows) != len(expected_target_rows):
        raise ValueError("target-band CSV does not match the recomputed target-band rows")

    outputs: list[Path] = []
    outputs.extend(
        figure_pareto_rmse(single_summary, out_dir, formats, dpi, target_low, target_high)
    )
    outputs.extend(
        figure_pareto_safety(single_summary, out_dir, formats, dpi, target_low, target_high)
    )
    outputs.extend(figure_target_band_matrix(single_summary, out_dir, formats, dpi))
    outputs.extend(figure_overall_vs_target(single_summary, out_dir, formats, dpi))
    outputs.extend(figure_strategy_wins(single_summary, out_dir, formats, dpi))
    outputs.extend(figure_multi_tier_tradeoff(multi_summary, out_dir, formats, dpi))
    if oracle_summary_csv:
        oracle_summary = load_oracle_summary(oracle_summary_csv)
        single_with_oracle = attach_oracle_baseline(single_summary, oracle_summary)
        outputs.extend(
            figure_normalized_family_frontiers(
                single_with_oracle,
                out_dir,
                formats,
                dpi,
                target_low,
                target_high,
            )
        )
        outputs.extend(figure_roofline_oracle(single_with_oracle, out_dir, formats, dpi))
        outputs.extend(figure_target_band_family_matrix(single_with_oracle, out_dir, formats, dpi))
    return outputs


def main() -> None:
    """Parse arguments and generate the figure set."""
    args = parse_args()
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    outputs = run(
        args.single_summary_csv,
        args.single_target_csv,
        args.multi_summary_csv,
        args.oracle_summary_csv or None,
        args.output_dir,
        args.dpi,
        formats,
        args.target_low,
        args.target_high,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
