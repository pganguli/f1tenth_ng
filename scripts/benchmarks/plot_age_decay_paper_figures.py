#!/usr/bin/env python3
"""Generate paper-facing figures for the anchored age-decay study."""

from __future__ import annotations

import argparse
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
    best_target_rows,
    load_single_summary,
    ordered_maps,
    save_figure_formats,
)


REPRESENTATIVE_DISPLAY = {
    "always_hit": "Always",
    "fixed_interval_k5": "Fixed Interval k=5",
    "fixed_bernoulli_p60": "Bernoulli p=0.60",
    "bernoulli_max_miss_p60_m5": "Bernoulli+Guard p=0.60, m=5",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-facing figures for the anchored age-decay study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lambda-summary-csv",
        type=str,
        default="data/benchmarks/lambda_sweep_train_13maps_summary.csv",
        help="Aggregated lambda sweep summary CSV.",
    )
    parser.add_argument(
        "--lambda-per-config-csv",
        type=str,
        default="data/benchmarks/lambda_sweep_train_13maps.csv",
        help="Raw lambda sweep experiment CSV.",
    )
    parser.add_argument(
        "--static-summary-csv",
        type=str,
        default="data/benchmarks/single_tier_paper_strategies_10maps_static_summary.csv",
        help="Held-out static full-suite summary CSV.",
    )
    parser.add_argument(
        "--lambda-summary-heldout-csv",
        type=str,
        default="data/benchmarks/single_tier_paper_strategies_10maps_lambda_summary.csv",
        help="Held-out anchored-decay full-suite summary CSV.",
    )
    parser.add_argument(
        "--best-config-summary-csv",
        type=str,
        default="data/benchmarks/eval_best_configs_10maps_lambda_summary.csv",
        help="Held-out best-config summary CSV.",
    )
    parser.add_argument(
        "--selected-lambda-json",
        type=str,
        default="data/benchmarks/lambda_sweep_optimal.json",
        help="JSON artifact containing the selected lambda.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/paper_figures_10maps_age_decay",
        help="Directory where figure files should be written.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG export DPI.")
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated export formats.",
    )
    return parser.parse_args()


def add_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Add a compact title block above the figure."""
    fig.text(0.03, 0.965, title, fontsize=22, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.03, 0.918, subtitle, fontsize=11.0, color=MUTED, ha="left", va="top")


def _trial_text(summary: pd.DataFrame) -> str:
    """Return a compact trial-count description."""
    trials = sorted({int(value) for value in summary["trials"].dropna().astype(int)})
    if not trials:
        return "trial count unavailable"
    if len(trials) == 1:
        return f"n={trials[0]} trials per configuration"
    return "trial counts vary"


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_augmented_summary(path: str, target_low: float, target_high: float) -> pd.DataFrame:
    """Load a held-out summary, tolerating pre-augmented synthetic fixtures."""
    try:
        return load_single_summary(path, target_low, target_high)
    except ValueError as exc:
        summary = pd.read_csv(path)
        if (
            "always-hit rows" not in str(exc)
            or "rmse_delta_vs_always_pct" not in summary.columns
        ):
            raise
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
        return summary.sort_values(["map_name", "cloud_latency", "rank"], kind="stable")


def load_selected_lambda(path: Path) -> float:
    """Load the selected lambda value from JSON."""
    import json

    data = json.loads(path.read_text())
    return float(data["selected_lambda"])


def figure_lambda_train_sweep(
    per_config_df: pd.DataFrame,
    lambda_summary_df: pd.DataFrame,
    selected_lambda: float,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot representative-strategy RMSE versus lambda with uncertainty."""
    required = ["age_decay_lambda", "experiment", "crosstrack_rmse_m_mean", "crosstrack_rmse_m_ci95"]
    _require_columns(per_config_df, required, "lambda per-config summary")

    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    add_title(
        fig,
        "Train-Set Lambda Sweep",
        "Representative strategies on the 13 training maps. Error bars show 95% confidence intervals across repeated trials.",
    )
    ax.grid(True, alpha=0.28, color=GRID)
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xlabel("Age-Decay Lambda")
    ax.set_ylabel("Mean Crosstrack RMSE (m)")

    plotted = []
    for experiment, label in REPRESENTATIVE_DISPLAY.items():
        subset = per_config_df[per_config_df["experiment"] == experiment].sort_values(
            "age_decay_lambda", kind="stable"
        )
        if subset.empty:
            continue
        strategy = subset["strategy"].iloc[0]
        color = STRATEGY_COLORS[strategy]
        marker = STRATEGY_MARKERS[strategy]
        ax.errorbar(
            subset["age_decay_lambda"],
            subset["crosstrack_rmse_m_mean"],
            yerr=subset["crosstrack_rmse_m_ci95"],
            color=color,
            marker=marker,
            markerfacecolor=color,
            linewidth=2.0,
            capsize=3.0,
            label=label,
        )
        plotted.append(label)

    ax.axvline(selected_lambda, color="#7b4e14", linestyle="--", linewidth=1.5)
    lambda_row = lambda_summary_df.loc[
        lambda_summary_df["age_decay_lambda"] == selected_lambda
    ].iloc[0]
    ax.text(
        selected_lambda,
        float(lambda_row["mean_crosstrack_rmse"]),
        f"  selected λ={selected_lambda:g}",
        color="#7b4e14",
        fontsize=10.5,
        va="bottom",
    )
    if plotted:
        ax.legend(frameon=False, loc="best", fontsize=10.2)
    return save_figure_formats(fig, out_dir, "lambda_train_sweep", formats, dpi)


def figure_static_vs_lambda_target_band(
    static_summary: pd.DataFrame,
    lambda_summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Compare best in-band static and anchored-decay winners per held-out map."""
    static_best = best_target_rows(static_summary).assign(condition="Static")
    lambda_best = best_target_rows(lambda_summary).assign(condition="Anchored Decay")
    combined = pd.concat([static_best, lambda_best], ignore_index=True)

    if combined.empty:
        raise ValueError("No target-band rows available for static-vs-lambda comparison.")

    combined["map_name"] = pd.Categorical(
        combined["map_name"],
        ordered_maps(combined["map_name"]),
        ordered=True,
    )
    combined = combined.sort_values(["map_name", "condition"], kind="stable")
    maps = list(combined["map_name"].cat.categories)
    y_base = np.arange(len(maps))
    offsets = {"Static": -0.18, "Anchored Decay": 0.18}
    colors = {"Static": "#8b6914", "Anchored Decay": "#157a6e"}

    fig, ax = plt.subplots(figsize=(13.0, max(6.2, 0.62 * len(maps) + 2.2)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    add_title(
        fig,
        "Held-Out Target-Band Comparison",
        "Best 55-65% CCR configuration per map under static fusion vs anchored age decay. Error bars show RMSE 95% confidence intervals.",
    )
    ax.axvline(0.0, color="#8d7e6d", linewidth=1.1)
    ax.grid(axis="x", alpha=0.28, color=GRID)
    for condition in ("Static", "Anchored Decay"):
        subset = combined[combined["condition"] == condition]
        positions = [y_base[maps.index(row.map_name)] + offsets[condition] for row in subset.itertuples()]
        ax.barh(
            positions,
            subset["rmse_delta_vs_always_pct"],
            xerr=subset.get("crosstrack_rmse_m_ci95", pd.Series(np.zeros(len(subset)))),
            color=colors[condition],
            edgecolor="#6b5d4f",
            height=0.32,
            linewidth=0.9,
            label=condition,
            capsize=3.0,
        )
    ax.set_yticks(y_base)
    ax.set_yticklabels(maps, fontsize=11.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE Change vs Always-Hit (%)")
    ax.legend(frameon=False, loc="best")
    return save_figure_formats(fig, out_dir, "static_vs_lambda_target_band", formats, dpi)


def figure_strategy_family_tradeoff(
    summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot all held-out anchored-decay strategies with RMSE and CCR uncertainty bars."""
    required = [
        "strategy",
        "experiment",
        "cloud_call_rate_mean",
        "cloud_call_rate_ci95",
        "crosstrack_rmse_m_mean",
        "crosstrack_rmse_m_ci95",
    ]
    _require_columns(summary, required, "held-out anchored summary")

    fig, ax = plt.subplots(figsize=(12.4, 8.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    add_title(
        fig,
        "Held-Out Communication vs Tracking Tradeoff",
        f"All 27 anchored-decay strategies on the 10 eval maps; {_trial_text(summary)}. Error bars show 95% confidence intervals.",
    )
    ax.grid(True, alpha=0.28, color=GRID)
    ax.set_xlabel("Cloud Call Rate")
    ax.set_ylabel("Crosstrack RMSE (m)")
    ax.axvspan(0.55, 0.65, color="#efe0c7", alpha=0.55, zorder=0)

    for row in summary.itertuples(index=False):
        color = STRATEGY_COLORS[row.strategy]
        marker = STRATEGY_MARKERS[row.strategy]
        ax.errorbar(
            row.cloud_call_rate_mean,
            row.crosstrack_rmse_m_mean,
            xerr=row.cloud_call_rate_ci95,
            yerr=row.crosstrack_rmse_m_ci95,
            fmt=marker,
            ms=7.5,
            mec=color,
            mfc=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.6,
            alpha=0.85,
        )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=STRATEGY_MARKERS[strategy],
            color=STRATEGY_COLORS[strategy],
            markerfacecolor=STRATEGY_COLORS[strategy],
            linestyle="None",
            markersize=7.0,
            label=label,
        )
        for strategy, label in STRATEGY_DISPLAY.items()
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best", fontsize=10.0)
    return save_figure_formats(fig, out_dir, "strategy_family_tradeoff", formats, dpi)


def figure_strategy_win_summary(
    static_summary: pd.DataFrame,
    lambda_summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Compare per-map in-band strategy wins under static and anchored decay."""
    static_wins = best_target_rows(static_summary)["strategy"].value_counts().reindex(
        STRATEGY_DISPLAY.keys(), fill_value=0
    )
    lambda_wins = best_target_rows(lambda_summary)["strategy"].value_counts().reindex(
        STRATEGY_DISPLAY.keys(), fill_value=0
    )
    labels = [STRATEGY_DISPLAY[key] for key in STRATEGY_DISPLAY]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.0), sharey=True)
    fig.patch.set_facecolor(BG)
    add_title(
        fig,
        "Target-Band Strategy Wins",
        "Number of eval maps won by each strategy family inside the 55-65% cloud-call band.",
    )
    for axis, series, title in zip(
        axes,
        (static_wins, lambda_wins),
        ("Static Fusion", "Anchored Age Decay"),
    ):
        axis.set_facecolor(PANEL)
        axis.grid(axis="x", alpha=0.28, color=GRID)
        values = series.to_numpy(dtype=float)
        colors = [STRATEGY_COLORS[key] for key in STRATEGY_DISPLAY]
        axis.barh(y, values, color=colors, edgecolor="#6b5d4f", linewidth=0.8, height=0.72)
        axis.set_title(title, loc="left", fontsize=15, fontweight="bold")
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=11.2)
        axis.invert_yaxis()
        for row_y, value in zip(y, values):
            axis.text(value + 0.12, row_y, f"{int(value)}", va="center", color=TEXT, fontsize=11.0)
    axes[1].set_xlabel("Maps Won")
    return save_figure_formats(fig, out_dir, "strategy_win_summary", formats, dpi)


def figure_alpha_decay_curves(
    selected_lambda: float,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot the per-feature anchored alpha decay curves for the selected lambda."""
    static_alphas = {
        "Left": 0.2,
        "Track Width": 0.2,
        "Heading": 0.7,
    }
    sigma2_edge = {
        "Left": 0.028020,
        "Track Width": 0.036518,
        "Heading": 0.019371,
    }
    sigma2_cloud = {
        "Left": 0.000518,
        "Track Width": 0.001539,
        "Heading": 0.001140,
    }
    sigma_proc = {
        "Left": 0.044961,
        "Track Width": 0.067937,
        "Heading": 0.033182,
    }
    colors = {"Left": "#b55d28", "Track Width": "#157a6e", "Heading": "#235fa4"}
    ages = np.arange(0, 16)

    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    add_title(
        fig,
        "Per-Feature Cloud Trust Decay",
        f"Anchored alphas at λ={selected_lambda:g}. Dashed lines show the static age-0 cloud weights.",
    )
    ax.grid(True, alpha=0.28, color=GRID)
    ax.set_xlabel("Cloud Age (steps since arrival)")
    ax.set_ylabel("Cloud Weight")
    for label in ("Left", "Track Width", "Heading"):
        base = sigma2_edge[label] + sigma2_cloud[label]
        values = static_alphas[label] * base / (
            base + selected_lambda * ages * (sigma_proc[label] ** 2)
        )
        ax.plot(ages, values, color=colors[label], linewidth=2.2, label=label)
        ax.axhline(static_alphas[label], color=colors[label], linestyle="--", linewidth=1.1, alpha=0.65)
    ax.set_ylim(0.0, 0.78)
    ax.legend(frameon=False, loc="best")
    return save_figure_formats(fig, out_dir, "alpha_decay_curves", formats, dpi)


def main() -> None:
    """Generate the anchored age-decay paper figure set."""
    args = parse_args()
    apply_theme()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    target_low = 0.55
    target_high = 0.65
    lambda_summary_df = pd.read_csv(args.lambda_summary_csv)
    per_config_df = pd.read_csv(args.lambda_per_config_csv)
    static_summary = load_augmented_summary(args.static_summary_csv, target_low, target_high)
    lambda_summary = load_augmented_summary(args.lambda_summary_heldout_csv, target_low, target_high)
    best_config_summary = pd.read_csv(args.best_config_summary_csv)
    selected_lambda = load_selected_lambda(Path(args.selected_lambda_json))

    # Aggregate the raw lambda sweep to per-lambda/per-config mean+CI for plotting.
    per_config_summary = (
        per_config_df.groupby(["age_decay_lambda", "experiment", "strategy"], as_index=False)
        .agg(
            trials=("run_idx", "count"),
            crosstrack_rmse_m_mean=("crosstrack_rmse_m", "mean"),
            crosstrack_rmse_m_std=("crosstrack_rmse_m", "std"),
        )
        .fillna(0.0)
    )
    per_config_summary["crosstrack_rmse_m_stderr"] = (
        per_config_summary["crosstrack_rmse_m_std"]
        / np.sqrt(per_config_summary["trials"].clip(lower=1))
    )
    per_config_summary["crosstrack_rmse_m_ci95"] = (
        1.96 * per_config_summary["crosstrack_rmse_m_stderr"]
    )

    outputs: list[Path] = []
    outputs.extend(
        figure_lambda_train_sweep(
            per_config_summary,
            lambda_summary_df,
            selected_lambda,
            out_dir,
            formats,
            args.dpi,
        )
    )
    outputs.extend(
        figure_static_vs_lambda_target_band(
            static_summary,
            lambda_summary,
            out_dir,
            formats,
            args.dpi,
        )
    )
    outputs.extend(
        figure_strategy_family_tradeoff(
            lambda_summary,
            out_dir,
            formats,
            args.dpi,
        )
    )
    outputs.extend(
        figure_strategy_win_summary(
            static_summary,
            lambda_summary,
            out_dir,
            formats,
            args.dpi,
        )
    )
    outputs.extend(figure_alpha_decay_curves(selected_lambda, out_dir, formats, args.dpi))

    # Use the best-config summary to ensure the file is validated in this plotting path.
    _require_columns(
        best_config_summary,
        ["strategy", "crosstrack_rmse_m_mean", "cloud_call_rate_mean"],
        "best-config held-out summary",
    )

    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
