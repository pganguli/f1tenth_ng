#!/usr/bin/env python3
"""Plot focused dual-signal periodic study outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot dual-signal periodic study outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cross-map-csv",
        type=str,
        default="data/benchmarks/dual_signal_periodic_eval_cross_map.csv",
    )
    parser.add_argument(
        "--comparison-csv",
        type=str,
        default="data/benchmarks/dual_signal_periodic_vs_previous.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="data/benchmarks/dual_signal_periodic_eval_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/paper_figures_dual_signal_periodic",
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
    """Load a CSV file."""
    return pd.read_csv(path)


def select_dual_winner(comparison_df: pd.DataFrame) -> pd.Series:
    """Return the best-ranked dual-signal finalist."""
    dual = comparison_df[comparison_df["strategy"] == "dual_signal_periodic"].copy()
    if dual.empty:
        raise ValueError("Comparison frame does not contain dual_signal_periodic rows.")
    dual = dual.sort_values(
        [
            "acceptable",
            "mean_cloud_call_rate",
            "mean_crosstrack_rmse",
            "mean_crosstrack_max",
            "experiment",
        ],
        ascending=[False, True, True, True, True],
        kind="stable",
    )
    return dual.iloc[0]


def _control_info(comparison_df: pd.DataFrame, label: str) -> tuple[str | None, str | None]:
    """Return control experiment/base names from comparison metadata when available."""
    experiment_col = f"{label}_experiment"
    base_col = f"{label}_base_experiment"
    if experiment_col in comparison_df.columns and base_col in comparison_df.columns:
        experiment = comparison_df[experiment_col].dropna()
        base = comparison_df[base_col].dropna()
        return (
            str(experiment.iloc[0]) if not experiment.empty else None,
            str(base.iloc[0]) if not base.empty else None,
        )
    legacy = {
        "baseline_k2": ("fixed_interval_k2__lambda_32p0", "fixed_interval_k2"),
        "interval_best": ("fixed_interval_k3__lambda_16p0", "fixed_interval_k3"),
        "bernoulli_best": (
            "bernoulli_max_miss_p15_m2__lambda_20p666667",
            "bernoulli_max_miss_p15_m2",
        ),
        "srp_best": ("self_norm_tau1p0_n3__lambda_4p0", "self_norm_tau1p0_n3"),
    }
    return legacy.get(label, (None, None))


def figure_tradeoff_overlay(
    cross_map_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Overlay prior findings with current dual-signal finalists."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.set_title("Dual-Signal vs Previous Winners")
    ax.set_xlabel("Mean Cloud Call Rate")
    ax.set_ylabel("Mean Max Cross-Track Error (m)")
    ax.grid(True, alpha=0.25)

    previous = comparison_df[comparison_df["stored_prev_mean_crosstrack_max"].notna()].copy()
    for row in previous.itertuples(index=False):
        ax.scatter(
            row.stored_prev_mean_cloud_call_rate,
            row.stored_prev_mean_crosstrack_max,
            color="#94a3b8",
            marker="o",
            s=42,
            alpha=0.8,
        )
        ax.text(
            row.stored_prev_mean_cloud_call_rate + 0.003,
            row.stored_prev_mean_crosstrack_max,
            row.base_experiment,
            fontsize=8,
            color="#64748b",
        )

    dual = cross_map_df[cross_map_df["strategy"] == "dual_signal_periodic"].copy()
    for row in dual.itertuples(index=False):
        ax.scatter(
            row.mean_cloud_call_rate,
            row.mean_crosstrack_max,
            color="#0f766e",
            marker="D",
            s=64,
        )
        ax.text(
            row.mean_cloud_call_rate + 0.003,
            row.mean_crosstrack_max,
            row.base_experiment,
            fontsize=8,
            color="#0f766e",
        )

    if "k2_threshold" in cross_map_df.columns:
        threshold = float(cross_map_df["k2_threshold"].iloc[0])
        ax.axhline(threshold, color="#b45309", linestyle="--", linewidth=1.2)
    save_figure(fig, output_dir, "tradeoff_overlay", formats, dpi)


def figure_direct_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Render the required direct comparison table."""
    dual_winner = select_dual_winner(comparison_df)
    control_labels = ["baseline_k2", "interval_best", "bernoulli_best", "srp_best"]
    control_experiments = {
        exp_name
        for exp_name, _base_name in (_control_info(comparison_df, label) for label in control_labels)
        if exp_name is not None
    }
    controls = comparison_df[comparison_df["experiment"].isin(control_experiments)].copy()
    controls = controls.sort_values("mean_cloud_call_rate", kind="stable")
    frame = pd.concat([controls, dual_winner.to_frame().T], ignore_index=True)
    comparison = pd.DataFrame(
        {
            "experiment": frame["base_experiment"],
            "mean_ccr": frame["mean_cloud_call_rate"].map(lambda v: round(float(v), 4)),
            "mean_max_cte": frame["mean_crosstrack_max"].map(lambda v: round(float(v), 4)),
            "mean_rmse_cte": frame["mean_crosstrack_rmse"].map(lambda v: round(float(v), 4)),
            "delta_ccr_vs_interval_best": frame["delta_ccr_vs_interval_best"].map(
                lambda v: round(float(v), 4)
            ),
            "delta_max_cte_vs_interval_best": frame["delta_max_cte_vs_interval_best"].map(
                lambda v: round(float(v), 4)
            ),
        }
    )

    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.axis("off")
    ax.set_title("Current Rerun Comparison")
    table = ax.table(
        cellText=comparison.values,
        colLabels=list(comparison.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.35)
    save_figure(fig, output_dir, "direct_comparison", formats, dpi)


def figure_per_map_heatmap(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Render per-map max-CTE deltas for the best dual config vs fixed_interval_k3."""
    dual_winner = select_dual_winner(comparison_df)
    dual_name = str(dual_winner["experiment"])
    baseline_name, baseline_base = _control_info(comparison_df, "interval_best")
    if baseline_name is None:
        raise ValueError("Comparison frame is missing interval_best metadata.")

    dual = summary_df[summary_df["experiment"] == dual_name][
        ["map_name", "crosstrack_max_m_mean"]
    ].rename(columns={"crosstrack_max_m_mean": "dual_max_cte"})
    baseline = summary_df[summary_df["experiment"] == baseline_name][
        ["map_name", "crosstrack_max_m_mean"]
    ].rename(columns={"crosstrack_max_m_mean": "k3_max_cte"})
    merged = dual.merge(baseline, on="map_name", how="inner")
    merged["delta"] = merged["dual_max_cte"] - merged["k3_max_cte"]
    merged = merged.sort_values("map_name", kind="stable").reset_index(drop=True)

    values = merged["delta"].to_numpy(dtype=float).reshape(1, -1)
    max_abs = max(abs(values.min()), abs(values.max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(merged)), 2.4))
    image = ax.imshow(values, cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax.set_title(
        f"Per-Map Max CTE Delta vs {baseline_base or 'interval_best'} ({dual_winner['base_experiment']})"
    )
    ax.set_xticks(range(len(merged)))
    ax.set_xticklabels(list(merged["map_name"]), rotation=30, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels(["delta"])
    for idx, value in enumerate(merged["delta"]):
        ax.text(idx, 0, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.75)
    save_figure(fig, output_dir, "per_map_delta_heatmap", formats, dpi)


def figure_call_reason_stacked_bar(
    cross_map_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Render dual-signal call reasons as a stacked bar chart."""
    dual = cross_map_df[cross_map_df["strategy"] == "dual_signal_periodic"].copy()
    if dual.empty:
        raise ValueError("Cross-map frame does not contain dual_signal_periodic rows.")
    dual = dual.sort_values(
        ["acceptable", "mean_cloud_call_rate", "mean_crosstrack_max"],
        ascending=[False, True, True],
        kind="stable",
    )

    columns = [
        ("mean_scheduler_calls_bootstrap", "bootstrap", "#94a3b8"),
        ("mean_scheduler_calls_backbone", "backbone", "#1d4ed8"),
        ("mean_scheduler_calls_burst", "burst", "#0f766e"),
        ("mean_scheduler_calls_force_age", "force_age", "#b45309"),
    ]
    fig, ax = plt.subplots(figsize=(10.0, max(4.5, 0.55 * len(dual) + 1.5)))
    cumulative = [0.0] * len(dual)
    y_labels = list(dual["base_experiment"])
    for column, label, color in columns:
        values = dual[column].fillna(0.0).tolist() if column in dual.columns else [0.0] * len(dual)
        ax.barh(y_labels, values, left=cumulative, label=label, color=color)
        cumulative = [left + value for left, value in zip(cumulative, values)]
    ax.set_title("Dual-Signal Call Mix")
    ax.set_xlabel("Mean Calls per Episode")
    ax.set_ylabel("Experiment")
    ax.legend(frameon=False)
    ax.invert_yaxis()
    save_figure(fig, output_dir, "call_reason_stacked", formats, dpi)


def main() -> None:
    """Render the focused dual-signal figure set."""
    args = parse_args()
    cross_map_df = load_frame(args.cross_map_csv)
    comparison_df = load_frame(args.comparison_csv)
    summary_df = load_frame(args.summary_csv)
    output_dir = Path(args.output_dir)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    figure_tradeoff_overlay(cross_map_df, comparison_df, output_dir, formats, args.dpi)
    figure_direct_comparison(comparison_df, output_dir, formats, args.dpi)
    figure_per_map_heatmap(summary_df, comparison_df, output_dir, formats, args.dpi)
    figure_call_reason_stacked_bar(cross_map_df, output_dir, formats, args.dpi)


if __name__ == "__main__":
    main()
