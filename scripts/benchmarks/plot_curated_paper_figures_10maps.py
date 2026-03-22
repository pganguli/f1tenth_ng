#!/usr/bin/env python3
"""Generate a small, paper-focused figure set for the corrected 10-map results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_paper_results_10maps import (
    ACCENT,
    BG,
    GOOD,
    GRID,
    MUTED,
    PANEL,
    STRATEGY_COLORS,
    STRATEGY_DISPLAY,
    STRATEGY_MARKERS,
    TEXT,
    apply_theme,
    best_target_rows,
    load_single_summary,
    ordered_maps,
    save_figure_formats,
    strategy_win_counts,
)


TARGET_FAMILIES = ["fixed_bernoulli", "bernoulli_max_miss"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a curated three-figure set for the corrected 10-map results."
    )
    parser.add_argument(
        "--single-summary-csv",
        type=str,
        default="data/benchmarks/single_tier_paper_strategies_10maps_summary.csv",
        help="Path to the aggregated single-tier summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/paper_figures_10maps_curated",
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


def add_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Add a compact title block above the figure."""
    fig.text(
        0.03,
        0.965,
        title,
        fontsize=26,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.03,
        0.925,
        subtitle,
        fontsize=11.5,
        color=MUTED,
        ha="left",
        va="top",
    )


def best_target_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return best target-band rows with a stable display order."""
    best = best_target_rows(summary).copy()
    return (
        best.assign(sort_key=pd.Categorical(best["map_name"], ordered_maps(best["map_name"]), ordered=True))
        .sort_values("sort_key", kind="stable")
        .drop(columns=["sort_key"])
        .reset_index(drop=True)
    )


def best_by_family_in_band(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the best in-band Bernoulli-family result per map and family."""
    subset = summary[
        summary["in_target_ccr_band"] & summary["strategy"].isin(TARGET_FAMILIES)
    ].copy()
    return (
        subset.sort_values(["map_name", "strategy", "rank"], kind="stable")
        .groupby(["map_name", "strategy"], observed=False, as_index=False)
        .first()
    )


def figure_target_band_winners(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot the best target-band configuration per map with clean side labels."""
    winners = best_target_table(summary)
    y_pos = np.arange(len(winners))

    fig = plt.figure(figsize=(14.0, 9.4))
    fig.patch.set_facecolor(BG)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.6, 1.9],
        left=0.11,
        right=0.985,
        top=0.9,
        bottom=0.09,
        wspace=0.04,
    )
    ax = fig.add_subplot(grid[0, 0])
    note_ax = fig.add_subplot(grid[0, 1], sharey=ax)

    ax.set_facecolor(PANEL)
    note_ax.set_facecolor(BG)
    note_ax.axis("off")

    values = winners["rmse_delta_vs_always_pct"].astype(float)
    colors = [STRATEGY_COLORS[strategy] for strategy in winners["strategy"]]

    ax.axvline(0.0, color="#8d7e6d", linewidth=1.2)
    ax.grid(axis="x", alpha=0.35, color=GRID)
    ax.barh(
        y_pos,
        values,
        color=colors,
        edgecolor="#6b5d4f",
        height=0.66,
        linewidth=0.9,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(winners["map_name"], fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE Change vs Always-Hit (%)")
    ax.set_xlim(min(-1.25, float(values.min()) - 0.12), max(0.52, float(values.max()) + 0.1))

    for row_y, row in zip(y_pos, winners.itertuples(index=False)):
        note_ax.text(
            0.0,
            row_y,
            (
                f"{STRATEGY_DISPLAY[row.strategy]}  "
                f"|  CCR {row.cloud_call_rate_mean:.3f}  "
                f"|  RMSE {row.crosstrack_rmse_m_mean:.3f}"
            ),
            fontsize=11.3,
            color=TEXT,
            va="center",
        )

    add_title(
        fig,
        "Best 55-65% CCR Configuration Per Map",
        "Negative bars beat the always-hit baseline. The right-hand label reports the winning family, achieved cloud-call rate, and RMSE.",
    )
    return save_figure_formats(fig, out_dir, "curated_target_band_winners", formats, dpi)


def figure_strategy_wins_curated(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot a cleaned summary of overall and in-band strategy wins."""
    overall = strategy_win_counts(summary, target_only=False)
    target = strategy_win_counts(summary, target_only=True)
    display_order = [
        "deterministic",
        "fixed_bernoulli",
        "bernoulli_max_miss",
        "logistic",
        "exponential",
        "piecewise_ramp",
        "always",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 9.4), sharex=False)
    fig.patch.set_facecolor(BG)

    add_title(
        fig,
        "Which Strategy Actually Wins?",
        "Top panel counts the best overall configuration per map. Bottom panel counts the best configuration inside the 55-65% CCR target band.",
    )

    for axis, series, title in zip(
        axes,
        [overall.reindex(display_order), target.reindex(display_order)],
        ["Best Overall", "Best In 55-65% CCR Band"],
    ):
        axis.set_facecolor(PANEL)
        labels = [STRATEGY_DISPLAY[name] for name in display_order]
        values = series.fillna(0).astype(float).to_numpy()
        colors = [STRATEGY_COLORS[name] for name in display_order]
        y = np.arange(len(labels))
        axis.barh(y, values, color=colors, edgecolor="#6b5d4f", linewidth=0.8, height=0.74)
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=11.5)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.3, color=GRID)
        axis.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=8)
        axis.set_xlim(0, max(6.6, float(values.max()) + 0.6))
        for row_y, value in zip(y, values):
            axis.text(
                value + 0.14,
                row_y,
                f"{int(value)}",
                va="center",
                fontsize=11.5,
                color=TEXT,
                fontweight="bold",
            )

    axes[-1].set_xlabel("Number of Maps Won")
    fig.tight_layout(rect=(0.03, 0.05, 0.995, 0.91))
    return save_figure_formats(fig, out_dir, "curated_strategy_wins", formats, dpi)


def figure_target_band_head_to_head(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Compare Bernoulli and Bernoulli+Guard directly inside the target band."""
    best = best_by_family_in_band(summary)
    rows = []
    for map_name in ordered_maps(best["map_name"]):
        map_rows = best[best["map_name"] == map_name]
        bernoulli = map_rows[map_rows["strategy"] == "fixed_bernoulli"]
        guarded = map_rows[map_rows["strategy"] == "bernoulli_max_miss"]
        if bernoulli.empty or guarded.empty:
            continue
        rows.append(
            {
                "map_name": map_name,
                "bernoulli_delta": float(bernoulli.iloc[0]["rmse_delta_vs_always_pct"]),
                "bernoulli_ccr": float(bernoulli.iloc[0]["cloud_call_rate_mean"]),
                "guard_delta": float(guarded.iloc[0]["rmse_delta_vs_always_pct"]),
                "guard_ccr": float(guarded.iloc[0]["cloud_call_rate_mean"]),
            }
        )

    compare = pd.DataFrame(rows)
    y = np.arange(len(compare))

    fig = plt.figure(figsize=(13.8, 8.9))
    fig.patch.set_facecolor(BG)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.2, 1.7],
        left=0.11,
        right=0.985,
        top=0.9,
        bottom=0.1,
        wspace=0.04,
    )
    ax = fig.add_subplot(grid[0, 0])
    note_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    ax.set_facecolor(PANEL)
    note_ax.set_facecolor(BG)
    note_ax.axis("off")

    ax.axvline(0.0, color="#8d7e6d", linewidth=1.2)
    ax.grid(axis="x", alpha=0.35, color=GRID)
    ax.hlines(y, compare["guard_delta"], compare["bernoulli_delta"], color="#c9b9a5", linewidth=2.0, zorder=1)
    ax.scatter(
        compare["bernoulli_delta"],
        y,
        s=105,
        marker=STRATEGY_MARKERS["fixed_bernoulli"],
        c=STRATEGY_COLORS["fixed_bernoulli"],
        edgecolors="#3b332b",
        linewidths=0.8,
        zorder=3,
        label="Bernoulli",
    )
    ax.scatter(
        compare["guard_delta"],
        y,
        s=105,
        marker=STRATEGY_MARKERS["bernoulli_max_miss"],
        c=STRATEGY_COLORS["bernoulli_max_miss"],
        edgecolors="#3b332b",
        linewidths=0.8,
        zorder=3,
        label="Bernoulli+Guard",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(compare["map_name"], fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE Change vs Always-Hit (%)")
    ax.set_xlim(min(-1.25, float(compare[["bernoulli_delta", "guard_delta"]].min().min()) - 0.08), 0.62)
    ax.legend(loc="lower left", frameon=False, fontsize=11)

    for row_y, row in zip(y, compare.itertuples(index=False)):
        winner = "Bernoulli" if row.bernoulli_delta < row.guard_delta else "Bernoulli+Guard"
        note_ax.text(
            0.0,
            row_y,
            (
                f"{winner} wins  "
                f"|  B {row.bernoulli_ccr:.3f}  "
                f"|  G {row.guard_ccr:.3f}"
            ),
            fontsize=11.1,
            color=TEXT,
            va="center",
        )

    add_title(
        fig,
        "Target-Band Head-To-Head: Bernoulli vs Bernoulli+Guard",
        "Only these two families produced 55-65% CCR configurations on the corrected 10-map subset. Left is better.",
    )
    return save_figure_formats(fig, out_dir, "curated_target_band_head_to_head", formats, dpi)


def run(
    single_summary_csv: str,
    output_dir: str,
    dpi: int,
    formats: list[str],
    target_low: float,
    target_high: float,
) -> list[Path]:
    """Generate the curated three-figure set."""
    apply_theme()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_single_summary(single_summary_csv, target_low, target_high)

    outputs: list[Path] = []
    outputs.extend(figure_target_band_winners(summary, out_dir, formats, dpi))
    outputs.extend(figure_strategy_wins_curated(summary, out_dir, formats, dpi))
    outputs.extend(figure_target_band_head_to_head(summary, out_dir, formats, dpi))
    return outputs


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    outputs = run(
        single_summary_csv=args.single_summary_csv,
        output_dir=args.output_dir,
        dpi=args.dpi,
        formats=formats,
        target_low=args.target_low,
        target_high=args.target_high,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
