#!/usr/bin/env python3
"""Generate compact paper-facing figures for the age-decay sanity study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_paper_results_10maps import (
    BG,
    GRID,
    MUTED,
    PANEL,
    STRATEGY_COLORS,
    STRATEGY_DISPLAY,
    STRATEGY_MARKERS,
    TEXT,
    apply_theme,
    ordered_maps,
    save_figure_formats,
)

TARGET_CCR = 0.60
REPRESENTATIVE_DISPLAY = {
    "always_hit": "Always",
    "fixed_interval_k5": "Fixed Interval k=5",
    "fixed_bernoulli_p60": "Bernoulli p=0.60",
    "bernoulli_max_miss_p60_m5": "Bernoulli+Guard p=0.60, m=5",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate compact Pareto-style figures for the age-decay sanity study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lambda-raw-csv",
        type=str,
        default="data/benchmarks/lambda_sweep_sanity_5train.csv",
    )
    parser.add_argument(
        "--selected-lambda-json",
        type=str,
        default="data/benchmarks/lambda_sweep_sanity_optimal.json",
    )
    parser.add_argument(
        "--static-raw-csv",
        type=str,
        default="data/benchmarks/single_tier_sanity_3eval_static.csv",
    )
    parser.add_argument(
        "--lambda-raw-heldout-csv",
        type=str,
        default="data/benchmarks/single_tier_sanity_3eval_lambda.csv",
    )
    parser.add_argument(
        "--best-config-raw-csv",
        type=str,
        default="data/benchmarks/eval_best_configs_sanity_3eval_lambda.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/paper_figures_sanity_3eval",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", type=str, default="png,pdf")
    return parser.parse_args()


def add_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Add a compact title block."""
    fig.text(0.03, 0.965, title, fontsize=21, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.03, 0.922, subtitle, fontsize=10.8, color=MUTED, ha="left", va="top")


def load_selected_lambda(path: str) -> tuple[float, bool]:
    """Return the selected lambda and whether it hit the grid boundary."""
    payload = json.loads(Path(path).read_text())
    return float(payload["selected_lambda"]), bool(payload.get("boundary_hit", False))


def aggregate_runs(raw_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate raw run-level rows into means and uncertainty bars."""
    grouped = (
        raw_df.groupby(group_cols, as_index=False)
        .agg(
            runs=("run_idx", "count"),
            collision_rate=("collision", "mean"),
            step_cap_rate=("step_cap_hit", "mean"),
            crosstrack_max_m_mean=("crosstrack_max_m", "mean"),
            crosstrack_max_m_std=("crosstrack_max_m", "std"),
            cloud_call_rate_mean=("cloud_call_rate", "mean"),
            cloud_call_rate_std=("cloud_call_rate", "std"),
        )
        .fillna(0.0)
    )
    root_n = np.sqrt(grouped["runs"].clip(lower=1).astype(float))
    grouped["crosstrack_max_m_stderr"] = grouped["crosstrack_max_m_std"] / root_n
    grouped["crosstrack_max_m_ci95"] = 1.96 * grouped["crosstrack_max_m_stderr"]
    grouped["cloud_call_rate_stderr"] = grouped["cloud_call_rate_std"] / root_n
    grouped["cloud_call_rate_ci95"] = 1.96 * grouped["cloud_call_rate_stderr"]
    grouped["ccr_dist_target"] = (grouped["cloud_call_rate_mean"] - TARGET_CCR).abs()
    return grouped


def select_best_family(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Select the best config per strategy family using max CTE ranking."""
    winners: list[pd.Series] = []
    for strategy, family in aggregated.groupby("strategy", sort=False):
        del strategy
        ranked = family.sort_values(
            [
                "collision_rate",
                "step_cap_rate",
                "crosstrack_max_m_mean",
                "ccr_dist_target",
                "experiment",
            ],
            kind="stable",
        )
        winners.append(ranked.iloc[0])
    return pd.DataFrame(winners).reset_index(drop=True)


def non_dominated_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return a mask for the 2D Pareto frontier minimizing CCR and max CTE."""
    points = frame[["cloud_call_rate_mean", "crosstrack_max_m_mean"]].to_numpy(dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for idx, point in enumerate(points):
        dominated = np.any(
            np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        )
        keep[idx] = not dominated
    return keep


def figure_appendix_lambda_sweep(
    lambda_raw: pd.DataFrame,
    selected_lambda: float,
    boundary_hit: bool,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot the internal lambda sweep appendix figure."""
    per_config = aggregate_runs(lambda_raw, ["age_decay_lambda", "experiment", "strategy"])
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    add_title(
        fig,
        "Appendix: Lambda Sweep on the 5 Training Maps",
        (
            "Internal tuning artifact. Selected lambda is marked; boundary hit is noted in the subtitle."
            if boundary_hit
            else "Internal tuning artifact used to choose the stale-cloud decay scale."
        ),
    )
    ax.grid(True, alpha=0.28, color=GRID)
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xlabel("Age-Decay Lambda")
    ax.set_ylabel("Mean Max CTE (m)")
    for experiment, label in REPRESENTATIVE_DISPLAY.items():
        subset = per_config[per_config["experiment"] == experiment].sort_values("age_decay_lambda", kind="stable")
        if subset.empty:
            continue
        strategy = subset["strategy"].iloc[0]
        ax.errorbar(
            subset["age_decay_lambda"],
            subset["crosstrack_max_m_mean"],
            yerr=subset["crosstrack_max_m_ci95"],
            color=STRATEGY_COLORS[strategy],
            marker=STRATEGY_MARKERS[strategy],
            markerfacecolor=STRATEGY_COLORS[strategy],
            linewidth=2.0,
            capsize=3.0,
            label=label,
        )
    ax.axvline(selected_lambda, color="#7b4e14", linestyle="--", linewidth=1.4)
    ax.legend(frameon=False, loc="best", fontsize=9.8)
    return save_figure_formats(fig, out_dir, "appendix_lambda_sweep", formats, dpi)


def figure_aggregate_pareto(
    static_raw: pd.DataFrame,
    lambda_raw: pd.DataFrame,
    best_lambda_raw: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot the aggregate held-out strategy Pareto view."""
    static_family = select_best_family(aggregate_runs(static_raw, ["experiment", "strategy"]))
    lambda_family = select_best_family(aggregate_runs(best_lambda_raw, ["experiment", "strategy"]))
    background = aggregate_runs(lambda_raw, ["experiment", "strategy"])

    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    add_title(
        fig,
        "Held-Out Strategy Pareto",
        "Three non-train maps. Hollow markers: static fusion. Filled markers: anchored age decay. Error bars show 95% confidence intervals.",
    )
    ax.grid(True, alpha=0.26, color=GRID)
    ax.scatter(
        background["cloud_call_rate_mean"],
        background["crosstrack_max_m_mean"],
        s=26,
        color="#b8b1a6",
        alpha=0.35,
        linewidths=0.0,
        zorder=1,
    )

    strategy_order = sorted(lambda_family["strategy"].tolist())
    frontier = lambda_family[non_dominated_mask(lambda_family)].copy()

    for strategy in strategy_order:
        lambda_row = lambda_family[lambda_family["strategy"] == strategy]
        if lambda_row.empty:
            continue
        lambda_row = lambda_row.iloc[0]
        static_row = static_family[static_family["strategy"] == strategy]
        color = STRATEGY_COLORS[strategy]
        marker = STRATEGY_MARKERS[strategy]
        if not static_row.empty:
            static_row = static_row.iloc[0]
            ax.plot(
                [static_row["cloud_call_rate_mean"], lambda_row["cloud_call_rate_mean"]],
                [static_row["crosstrack_max_m_mean"], lambda_row["crosstrack_max_m_mean"]],
                color=color,
                alpha=0.45,
                linewidth=1.2,
                zorder=2,
            )
            ax.errorbar(
                static_row["cloud_call_rate_mean"],
                static_row["crosstrack_max_m_mean"],
                xerr=static_row["cloud_call_rate_ci95"],
                yerr=static_row["crosstrack_max_m_ci95"],
                fmt=marker,
                mfc="none",
                mec=color,
                ecolor=color,
                mew=1.6,
                ms=8.2,
                capsize=3.0,
                alpha=0.9,
                zorder=3,
            )
        ax.errorbar(
            lambda_row["cloud_call_rate_mean"],
            lambda_row["crosstrack_max_m_mean"],
            xerr=lambda_row["cloud_call_rate_ci95"],
            yerr=lambda_row["crosstrack_max_m_ci95"],
            fmt=marker,
            mfc=color,
            mec=color,
            ecolor=color,
            ms=8.4,
            capsize=3.0,
            linewidth=1.6,
            label=STRATEGY_DISPLAY[strategy],
            zorder=4,
        )

    if not frontier.empty:
        frontier = frontier.sort_values(["cloud_call_rate_mean", "crosstrack_max_m_mean"], kind="stable")
        ax.plot(
            frontier["cloud_call_rate_mean"],
            frontier["crosstrack_max_m_mean"],
            color="#6a4c1c",
            linewidth=1.4,
            linestyle="--",
            zorder=2,
        )
        for row in frontier.itertuples(index=False):
            ax.text(
                row.cloud_call_rate_mean + 0.005,
                row.crosstrack_max_m_mean + 0.0015,
                STRATEGY_DISPLAY[row.strategy],
                color=TEXT,
                fontsize=10.2,
            )

    ax.set_xlabel("Cloud Call Rate")
    ax.set_ylabel("Mean Max CTE (m)")
    return save_figure_formats(fig, out_dir, "aggregate_strategy_pareto", formats, dpi)


def figure_per_map_triptych(
    static_raw: pd.DataFrame,
    best_lambda_raw: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot one Pareto panel per held-out map."""
    static_map = aggregate_runs(static_raw, ["map_name", "experiment", "strategy"])
    lambda_map = aggregate_runs(best_lambda_raw, ["map_name", "experiment", "strategy"])
    eval_maps = ordered_maps(lambda_map["map_name"])

    fig, axes = plt.subplots(1, len(eval_maps), figsize=(5.2 * len(eval_maps), 5.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(BG)
    add_title(
        fig,
        "Per-Map Pareto Panels",
        "Held-out best-per-family points only. Filled markers use anchored age decay; hollow markers use static fusion.",
    )

    for ax, map_name in zip(axes, eval_maps):
        ax.set_facecolor(PANEL)
        ax.grid(True, alpha=0.24, color=GRID)
        static_best = select_best_family(static_map[static_map["map_name"] == map_name])
        lambda_best = select_best_family(lambda_map[lambda_map["map_name"] == map_name])
        frontier = lambda_best[non_dominated_mask(lambda_best)].copy() if not lambda_best.empty else lambda_best

        for row in lambda_best.itertuples(index=False):
            color = STRATEGY_COLORS[row.strategy]
            marker = STRATEGY_MARKERS[row.strategy]
            static_row = static_best[static_best["strategy"] == row.strategy]
            if not static_row.empty:
                static_row = static_row.iloc[0]
                ax.errorbar(
                    static_row["cloud_call_rate_mean"],
                    static_row["crosstrack_max_m_mean"],
                    xerr=static_row["cloud_call_rate_ci95"],
                    yerr=static_row["crosstrack_max_m_ci95"],
                    fmt=marker,
                    mfc="none",
                    mec=color,
                    ecolor=color,
                    ms=6.6,
                    capsize=2.5,
                    alpha=0.85,
                )
            ax.errorbar(
                row.cloud_call_rate_mean,
                row.crosstrack_max_m_mean,
                xerr=row.cloud_call_rate_ci95,
                yerr=row.crosstrack_max_m_ci95,
                fmt=marker,
                mfc=color,
                mec=color,
                ecolor=color,
                ms=6.8,
                capsize=2.5,
            )

        for row in frontier.itertuples(index=False):
            ax.text(
                row.cloud_call_rate_mean + 0.004,
                row.crosstrack_max_m_mean + 0.001,
                STRATEGY_DISPLAY[row.strategy],
                fontsize=8.8,
                color=TEXT,
            )
        ax.set_title(map_name, fontsize=12.0, color=TEXT, fontweight="bold")
        ax.set_xlabel("Cloud Call Rate")
    axes[0].set_ylabel("Mean Max CTE (m)")
    return save_figure_formats(fig, out_dir, "per_map_strategy_pareto", formats, dpi)


def figure_family_leaderboard(
    static_raw: pd.DataFrame,
    best_lambda_raw: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot a compact family-level comparison leaderboard."""
    static_family = select_best_family(aggregate_runs(static_raw, ["experiment", "strategy"]))
    lambda_family = select_best_family(aggregate_runs(best_lambda_raw, ["experiment", "strategy"]))
    combined = pd.concat(
        [
            static_family.assign(condition="Static"),
            lambda_family.assign(condition="Anchored"),
        ],
        ignore_index=True,
    )
    order = (
        lambda_family.sort_values("crosstrack_max_m_mean", kind="stable")["strategy"].tolist()
    )
    combined["strategy"] = pd.Categorical(combined["strategy"], order, ordered=True)
    combined = combined.sort_values(["strategy", "condition"], kind="stable")

    fig, ax = plt.subplots(figsize=(11.4, max(5.4, 0.7 * len(order) + 1.8)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    add_title(
        fig,
        "Held-Out Family Leaderboard",
        "Families ordered by anchored-decay mean max CTE. Error bars show 95% confidence intervals; labels include mean CCR.",
    )
    ax.grid(axis="x", alpha=0.26, color=GRID)

    y_base = np.arange(len(order))
    offsets = {"Static": -0.17, "Anchored": 0.17}
    styles = {
        "Static": {"markerfacecolor": "none", "alpha": 0.85},
        "Anchored": {"markerfacecolor": None, "alpha": 1.0},
    }

    for row in combined.itertuples(index=False):
        color = STRATEGY_COLORS[str(row.strategy)]
        marker = STRATEGY_MARKERS[str(row.strategy)]
        y = y_base[order.index(str(row.strategy))] + offsets[row.condition]
        style = styles[row.condition]
        mfc = color if style["markerfacecolor"] is None else style["markerfacecolor"]
        ax.errorbar(
            row.crosstrack_max_m_mean,
            y,
            xerr=row.crosstrack_max_m_ci95,
            fmt=marker,
            mfc=mfc,
            mec=color,
            ecolor=color,
            ms=7.4,
            capsize=3.0,
            alpha=style["alpha"],
        )
        ax.text(
            row.crosstrack_max_m_mean + row.crosstrack_max_m_ci95 + 0.0015,
            y,
            f"{row.condition}  CCR {row.cloud_call_rate_mean:.3f}",
            va="center",
            fontsize=9.3,
            color=MUTED,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels([STRATEGY_DISPLAY[name] for name in order], fontsize=10.6)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Max CTE (m)")
    return save_figure_formats(fig, out_dir, "family_comparison_leaderboard", formats, dpi)


def main() -> None:
    """Generate the sanity-study figures."""
    args = parse_args()
    apply_theme()
    formats = [value.strip() for value in args.formats.split(",") if value.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lambda_raw = pd.read_csv(args.lambda_raw_csv)
    static_raw = pd.read_csv(args.static_raw_csv)
    lambda_raw_heldout = pd.read_csv(args.lambda_raw_heldout_csv)
    best_lambda_raw = pd.read_csv(args.best_config_raw_csv)
    selected_lambda, boundary_hit = load_selected_lambda(args.selected_lambda_json)

    outputs: list[Path] = []
    outputs.extend(
        figure_appendix_lambda_sweep(
            lambda_raw,
            selected_lambda,
            boundary_hit,
            out_dir,
            formats,
            args.dpi,
        )
    )
    outputs.extend(
        figure_aggregate_pareto(
            static_raw,
            lambda_raw_heldout,
            best_lambda_raw,
            out_dir,
            formats,
            args.dpi,
        )
    )
    outputs.extend(
        figure_per_map_triptych(
            static_raw,
            best_lambda_raw,
            out_dir,
            formats,
            args.dpi,
        )
    )
    outputs.extend(
        figure_family_leaderboard(
            static_raw,
            best_lambda_raw,
            out_dir,
            formats,
            args.dpi,
        )
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
