#!/usr/bin/env python3
"""Focused plotting for the latency-10 SRP/SRPv2 comprehensive comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "Fixed Interval": "#c68b17",
    "SRP (Ours)": "#14837d",
    "SRPv2": "#d2693c",
    "Never query": "#9ca3af",
}
MARKERS = {
    "Fixed Interval": "s",
    "SRP (Ours)": "D",
    "SRPv2": "v",
    "Never query": "+",
}
BANDS = ["0-10% CCR", "10-20% CCR", "20-30% CCR", "30-40% CCR", "40-50% CCR", "50-60% CCR"]
FAMILY_LABELS = {
    "fixed_interval": "Fixed Interval",
    "self_normalizing_momentum": "SRP (Ours)",
    "srpv2": "SRPv2",
}
EDGE_COLORS = {
    "Fixed Interval": "#5b3b0d",
    "SRP (Ours)": "#27424c",
    "SRPv2": "#5d3626",
    "Never query": "#4b5563",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot the latency-10 SRPv2 comparison figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cross-map-csv",
        type=str,
        default="data/benchmarks/srpv2_comparison_L10_eval_cross_map.csv",
    )
    parser.add_argument(
        "--budget-matches-csv",
        type=str,
        default="data/benchmarks/srpv2_comparison_L10_eval_budget_matches.csv",
    )
    parser.add_argument(
        "--paired-delta-csv",
        type=str,
        default="data/benchmarks/srpv2_comparison_L10_eval_paired_delta.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/paper_figures_srpv2_L10",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--formats", type=str, default="png,pdf")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    """Save a figure in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def apply_plot_style() -> None:
    """Set a restrained paper-facing matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#94a3b8",
            "axes.facecolor": "#fffdf8",
            "figure.facecolor": "#fffdf8",
            "savefig.facecolor": "#fffdf8",
            "grid.color": "#cbd5e1",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "axes.titleweight": "regular",
        }
    )


def _y_positions() -> dict[str, int]:
    return {label: idx for idx, label in enumerate(BANDS)}


def _band_label_for_rate(rate: float) -> str:
    percent = float(rate) * 100.0
    for low, high in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)):
        if low <= percent < high:
            return f"{low}-{high}% CCR"
    return "Out of Band"


def figure_budget_matched(
    cross_map_df: pd.DataFrame,
    budget_matches_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Render the equal-CCR budget-matched scatter comparison."""
    fig, ax = plt.subplots(figsize=(11.5, 6.3))
    ax.set_title("Budget-Matched Comparison at Equal Cloud Call Rates", fontsize=20, pad=12)
    ax.set_xlabel("Max cross-track error (m, mean across eval maps)", fontsize=13)
    ax.set_ylabel("")
    ax.grid(True)

    y_positions = _y_positions()
    interval = cross_map_df[cross_map_df["strategy"] == "fixed_interval"].copy()
    interval["plot_band"] = interval["mean_cloud_call_rate"].map(_band_label_for_rate)
    interval = interval[interval["plot_band"].isin(BANDS)].copy()
    interval = interval.sort_values("mean_cloud_call_rate", kind="stable")
    ax.scatter(
        interval["mean_crosstrack_max"],
        interval["plot_band"].map(y_positions),
        s=170,
        marker=MARKERS["Fixed Interval"],
        color=COLORS["Fixed Interval"],
        edgecolor=EDGE_COLORS["Fixed Interval"],
        linewidth=1.2,
        label="Fixed Interval",
        zorder=2,
    )

    for label in ("SRP (Ours)", "SRPv2"):
        family = budget_matches_df[budget_matches_df["display_strategy"] == label].copy()
        if family.empty:
            continue
        family["plot_band"] = family["mean_cloud_call_rate"].map(_band_label_for_rate)
        family = family[family["plot_band"].isin(BANDS)].sort_values(
            "mean_cloud_call_rate",
            kind="stable",
        )
        ax.plot(
            family["mean_crosstrack_max"],
            family["plot_band"].map(y_positions),
            color=COLORS[label],
            linewidth=1.6,
            alpha=0.35,
            zorder=1,
        )
        ax.scatter(
            family["mean_crosstrack_max"],
            family["plot_band"].map(y_positions),
            s=170,
            marker=MARKERS[label],
            color=COLORS[label],
            edgecolor=EDGE_COLORS[label],
            linewidth=1.2,
            label=label,
            zorder=3,
        )
        for row in family.itertuples(index=False):
            ax.annotate(
                f"{label.split()[0]} Δ {row.delta_max_cte_vs_interval:+.3f} m",
                xy=(row.mean_crosstrack_max, y_positions[row.plot_band]),
                xytext=(12, -10 if label == "SRP (Ours)" else 10),
                textcoords="offset points",
                fontsize=9.5,
                color=COLORS[label],
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": COLORS[label],
                    "alpha": 0.75,
                },
            )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(BANDS, fontsize=12)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    save_figure(fig, output_dir, "budget_matched_equal_ccr", formats, dpi)


def figure_headline_tradeoff(
    cross_map_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Render the main Max CTE vs cloud-call tradeoff figure."""
    fig, ax = plt.subplots(figsize=(10.6, 6.1))
    ax.set_title("Max CTE vs Cloud Call Rate", fontsize=20, pad=12)
    ax.set_xlabel("Cloud call rate (%)", fontsize=13)
    ax.set_ylabel("Max cross-track error (cm, mean across eval maps)", fontsize=13)
    ax.grid(True)

    frame = cross_map_df[cross_map_df["strategy"].isin(FAMILY_LABELS)].copy()
    frame["mean_crosstrack_max_cm"] = frame["mean_crosstrack_max"] * 100.0
    for strategy, label in FAMILY_LABELS.items():
        family = frame[frame["strategy"] == strategy].copy().sort_values(
            "mean_cloud_call_rate",
            kind="stable",
        )
        if family.empty:
            continue
        ax.plot(
            family["mean_cloud_call_rate"] * 100.0,
            family["mean_crosstrack_max_cm"],
            color=COLORS[label],
            linewidth=1.6,
            alpha=0.5,
            zorder=1,
        )
        ax.scatter(
            family["mean_cloud_call_rate"] * 100.0,
            family["mean_crosstrack_max_cm"],
            s=118,
            marker=MARKERS[label],
            color=COLORS[label],
            edgecolor=EDGE_COLORS[label],
            linewidth=0.95,
            label=label,
            zorder=2,
        )

    annotations = [
        ("self_normalizing_momentum", "SRP best"),
        ("srpv2", "SRPv2 best"),
        ("fixed_interval", "Interval best"),
    ]
    for strategy, tag in annotations:
        family = frame[frame["strategy"] == strategy].copy()
        if family.empty:
            continue
        sort_keys = ["mean_crosstrack_max", "mean_cloud_call_rate"]
        if "mean_crosstrack_rmse" in family.columns:
            sort_keys.append("mean_crosstrack_rmse")
        sort_keys.append("experiment")
        best = family.sort_values(sort_keys, kind="stable").iloc[0]
        label = FAMILY_LABELS[strategy]
        ax.scatter(
            [best["mean_cloud_call_rate"] * 100.0],
            [best["mean_crosstrack_max_cm"]],
            s=190,
            facecolors="none",
            edgecolors=EDGE_COLORS[label],
            linewidths=1.25,
            zorder=3,
        )
        ax.annotate(
            f"{tag}\n{best['mean_crosstrack_max_cm']:.1f} cm, {best['mean_cloud_call_rate'] * 100.0:.1f}%",
            xy=(best["mean_cloud_call_rate"] * 100.0, best["mean_crosstrack_max_cm"]),
            xytext=(10, -16 if strategy != "fixed_interval" else 10),
            textcoords="offset points",
            fontsize=9.2,
            color=EDGE_COLORS[label],
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "#fffdf8",
                "edgecolor": EDGE_COLORS[label],
                "alpha": 0.9,
            },
        )

    ax.set_xlim(left=0.0)
    ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    save_figure(fig, output_dir, "max_cte_vs_cloud_call_rate", formats, dpi)


def figure_delta_vs_interval(
    budget_matches_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Render the improvement over nearest interval at matched budgets."""
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_title("Δ Max CTE vs Nearest Interval at Matched Budgets", fontsize=20, pad=12)
    ax.set_xlabel("Cloud call rate (%)", fontsize=13)
    ax.set_ylabel("Δ Max CTE vs nearest interval (m)", fontsize=13)
    ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.4)
    ax.grid(True)

    for label in ("SRP (Ours)", "SRPv2"):
        family = budget_matches_df[budget_matches_df["display_strategy"] == label].copy()
        if family.empty:
            continue
        family = family.sort_values("mean_cloud_call_rate", kind="stable")
        ax.plot(
            family["mean_cloud_call_rate"] * 100.0,
            family["delta_max_cte_vs_interval"],
            color=COLORS[label],
            linewidth=1.6,
            alpha=0.45,
            zorder=1,
        )
        ax.scatter(
            family["mean_cloud_call_rate"] * 100.0,
            family["delta_max_cte_vs_interval"],
            s=165,
            marker=MARKERS[label],
            color=COLORS[label],
            edgecolor=EDGE_COLORS[label],
            linewidth=1.1,
            label=label,
            zorder=2,
        )

    ax.legend(frameon=False, loc="upper right", fontsize=11)
    save_figure(fig, output_dir, "delta_vs_nearest_interval", formats, dpi)


def figure_pareto_frontier(
    cross_map_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Render the low-call Pareto frontier for max CTE vs cloud calls."""
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_title("Pareto Frontier: Minimize Calls and Max CTE", fontsize=20, pad=12)
    ax.set_xlabel("Cloud call rate (%)", fontsize=13)
    ax.set_ylabel("Max cross-track error (m, mean across eval maps)", fontsize=13)
    ax.grid(True)

    frame = cross_map_df[cross_map_df["strategy"].isin(FAMILY_LABELS)].copy()
    rows = frame.to_dict("records")
    pareto_names: set[str] = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                float(other["mean_cloud_call_rate"]) <= float(row["mean_cloud_call_rate"])
                and float(other["mean_crosstrack_max"]) <= float(row["mean_crosstrack_max"])
            )
            strictly_better = (
                float(other["mean_cloud_call_rate"]) < float(row["mean_cloud_call_rate"])
                or float(other["mean_crosstrack_max"]) < float(row["mean_crosstrack_max"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_names.add(str(row["experiment"]))

    for strategy, label in FAMILY_LABELS.items():
        family = frame[frame["strategy"] == strategy].copy().sort_values(
            "mean_cloud_call_rate",
            kind="stable",
        )
        if family.empty:
            continue
        ax.scatter(
            family["mean_cloud_call_rate"] * 100.0,
            family["mean_crosstrack_max"],
            s=155,
            marker=MARKERS[label],
            color=COLORS[label],
            edgecolor=EDGE_COLORS[label],
            linewidth=1.0,
            alpha=0.72,
            label=label,
            zorder=2,
        )

    frontier = frame[frame["experiment"].isin(pareto_names)].copy().sort_values(
        "mean_cloud_call_rate",
        kind="stable",
    )
    ax.plot(
        frontier["mean_cloud_call_rate"] * 100.0,
        frontier["mean_crosstrack_max"],
        color="#111827",
        linewidth=1.6,
        linestyle="--",
        alpha=0.75,
        zorder=1,
    )
    ax.scatter(
        frontier["mean_cloud_call_rate"] * 100.0,
        frontier["mean_crosstrack_max"],
        s=210,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.4,
        zorder=3,
    )

    for row in frontier.itertuples(index=False):
        label = FAMILY_LABELS[row.strategy]
        if label not in {"SRP (Ours)", "SRPv2"}:
            continue
        ax.annotate(
            label.replace(" (Ours)", ""),
            xy=(row.mean_cloud_call_rate * 100.0, row.mean_crosstrack_max),
            xytext=(8, -14 if label == "SRP (Ours)" else 10),
            textcoords="offset points",
            fontsize=10,
            color=EDGE_COLORS[label],
        )

    ax.set_xlim(left=0.0)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    save_figure(fig, output_dir, "pareto_frontier_max_cte", formats, dpi)

def figure_paired_delta(
    paired_delta_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Render the paired delta vs always-query figure."""
    order = ["Never query", "Fixed Interval", "SRP (Ours)", "SRPv2"]
    panels = ["Sochi", "Spa", "Aggregate"]
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.0), sharey=True)
    fig.suptitle("Latency-10 Paired Δ Max CTE vs Always-query", fontsize=22, y=1.02)
    fig.text(
        0.5,
        0.96,
        "Representative held-out controls with paired bootstrap 95% confidence intervals.",
        ha="center",
        fontsize=11.5,
        color="#6b7280",
    )

    for ax, panel in zip(axes, panels, strict=True):
        frame = paired_delta_df[paired_delta_df["panel"] == panel].copy()
        frame["label"] = pd.Categorical(frame["label"], order, ordered=True)
        frame = frame.sort_values("label", kind="stable")
        ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.4)
        ax.grid(True, axis="y")
        for idx, row in enumerate(frame.itertuples(index=False)):
            low = float(row.mean_delta_max_cte_vs_always - row.ci95_low)
            high = float(row.ci95_high - row.mean_delta_max_cte_vs_always)
            marker_kwargs = {}
            if row.label != "Never query":
                marker_kwargs = {
                    "edgecolor": EDGE_COLORS.get(row.label, "#475569"),
                    "linewidth": 1.0,
                }
            ax.errorbar(
                idx,
                row.mean_delta_max_cte_vs_always,
                yerr=[[low], [high]],
                fmt=MARKERS[row.label],
                color=COLORS[row.label],
                markersize=10,
                elinewidth=1.8,
                capsize=4,
            )
            ax.scatter(
                [idx],
                [row.mean_delta_max_cte_vs_always],
                s=90,
                marker=MARKERS[row.label],
                color=COLORS[row.label],
                zorder=3,
                **marker_kwargs,
            )
        ax.set_title(panel, fontsize=16)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=24, ha="right", fontsize=11)
    axes[0].set_ylabel("Δ Max CTE vs always", fontsize=13)
    save_figure(fig, output_dir, "paired_delta_vs_always", formats, dpi)


def write_tables(
    cross_map_df: pd.DataFrame,
    budget_matches_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write paper-facing CSV and LaTeX tables for the headline results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def pick(experiment: str) -> pd.Series:
        return cross_map_df[cross_map_df["experiment"] == experiment].iloc[0]

    def pick_optional(experiment: str) -> pd.Series:
        rows = cross_map_df[cross_map_df["experiment"] == experiment]
        if not rows.empty:
            return rows.iloc[0]
        return pd.Series(
            {
                "experiment": experiment,
                "mean_cloud_call_rate": float("nan"),
                "mean_crosstrack_max": float("nan"),
                "mean_crosstrack_rmse": float("nan"),
                "mean_collision_rate": float("nan"),
                "acceptable": False,
            }
        )

    def pick_interval_fallback(preferred: str, order: str = "low") -> pd.Series:
        rows = cross_map_df[cross_map_df["strategy"] == "fixed_interval"].copy()
        preferred_rows = rows[rows["experiment"] == preferred]
        if not preferred_rows.empty:
            return preferred_rows.iloc[0]
        rows = rows.sort_values(["mean_cloud_call_rate", "mean_crosstrack_max"], kind="stable")
        if order == "high":
            return rows.iloc[-1]
        return rows.iloc[0]

    def pick_family_best(strategy: str, label: str) -> pd.Series:
        family = cross_map_df[cross_map_df["strategy"] == strategy].copy()
        if not family.empty:
            sort_keys = ["mean_crosstrack_max", "mean_cloud_call_rate"]
            if "mean_crosstrack_rmse" in family.columns:
                sort_keys.append("mean_crosstrack_rmse")
            sort_keys.append("experiment")
            return family.sort_values(sort_keys, kind="stable").iloc[0]
        budget_family = budget_matches_df[budget_matches_df["display_strategy"] == label].copy()
        if budget_family.empty:
            raise ValueError(f"Missing rows for {label} in cross-map and budget tables.")
        budget_family = budget_family.sort_values(
            ["mean_crosstrack_max", "mean_cloud_call_rate", "experiment"],
            kind="stable",
        )
        best_budget = budget_family.iloc[0]
        return pd.Series(
            {
                "experiment": best_budget["experiment"],
                "mean_cloud_call_rate": best_budget["mean_cloud_call_rate"],
                "mean_crosstrack_max": best_budget["mean_crosstrack_max"],
                "mean_crosstrack_rmse": float("nan"),
                "mean_collision_rate": float("nan"),
                "acceptable": False,
            }
        )

    srp_best = pick_family_best("self_normalizing_momentum", "SRP (Ours)")
    srpv2_best = pick_family_best("srpv2", "SRPv2")
    interval_low_budget = pick_interval_fallback("fixed_interval_k10__lambda_4p0", order="low")
    interval_strong = pick_interval_fallback("fixed_interval_k3__lambda_16p0", order="high")
    never = pick_optional("never_query__lambda_0p0")
    always = pick_optional("always_hit__lambda_0p0")

    headline_rows = [
        ("Never query", never),
        ("Fixed interval (k=10)", interval_low_budget),
        ("SRP best", srp_best),
        ("SRPv2 best", srpv2_best),
        ("Fixed interval (k=3)", interval_strong),
        ("Always query", always),
    ]
    headline_df = pd.DataFrame(
        [
            {
                "Method": label,
                "Experiment": row["experiment"],
                "CCR (%)": float(row["mean_cloud_call_rate"]) * 100.0,
                "Max CTE (m)": float(row["mean_crosstrack_max"]),
                "RMSE (m)": float(row["mean_crosstrack_rmse"])
                if "mean_crosstrack_rmse" in row.index
                else float("nan"),
                "Collision rate": float(row["mean_collision_rate"])
                if "mean_collision_rate" in row.index
                else float("nan"),
                "Acceptable": bool(row["acceptable"]) if "acceptable" in row.index else False,
            }
            for label, row in headline_rows
        ]
    )
    headline_df.to_csv(output_dir / "headline_performance_table.csv", index=False)
    headline_df.to_latex(
        output_dir / "headline_performance_table.tex",
        index=False,
        float_format=lambda x: f"{x:.3f}",
        escape=False,
    )

    budget_table_df = budget_matches_df.copy()
    budget_table_df["CCR (%)"] = budget_table_df["mean_cloud_call_rate"] * 100.0
    selected_columns = ["display_strategy", "experiment", "CCR (%)", "mean_crosstrack_max"]
    rename_columns = {
        "display_strategy": "Method",
        "experiment": "Experiment",
        "mean_crosstrack_max": "Max CTE (m)",
    }
    if "matched_interval_experiment" in budget_table_df.columns:
        selected_columns.append("matched_interval_experiment")
        rename_columns["matched_interval_experiment"] = "Nearest interval"
    if "matched_interval_crosstrack_max" in budget_table_df.columns:
        selected_columns.append("matched_interval_crosstrack_max")
        rename_columns["matched_interval_crosstrack_max"] = "Interval Max CTE (m)"
    if "delta_max_cte_vs_interval" in budget_table_df.columns:
        selected_columns.append("delta_max_cte_vs_interval")
        rename_columns["delta_max_cte_vs_interval"] = "$\\Delta$ Max CTE (m)"
    budget_table_df = budget_table_df[selected_columns].rename(columns=rename_columns)
    budget_table_df.to_csv(output_dir / "budget_match_table.csv", index=False)
    budget_table_df.to_latex(
        output_dir / "budget_match_table.tex",
        index=False,
        float_format=lambda x: f"{x:.3f}",
        escape=False,
    )


def main() -> None:
    """Render the focused SRP/SRPv2 figure set."""
    args = parse_args()
    apply_plot_style()
    cross_map_df = pd.read_csv(args.cross_map_csv)
    budget_matches_df = pd.read_csv(args.budget_matches_csv)
    paired_delta_df = pd.read_csv(args.paired_delta_csv)
    output_dir = Path(args.output_dir)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    figure_headline_tradeoff(cross_map_df, output_dir, formats, args.dpi)
    figure_budget_matched(cross_map_df, budget_matches_df, output_dir, formats, args.dpi)
    figure_delta_vs_interval(budget_matches_df, output_dir, formats, args.dpi)
    figure_pareto_frontier(cross_map_df, output_dir, formats, args.dpi)
    figure_paired_delta(paired_delta_df, output_dir, formats, args.dpi)
    write_tables(cross_map_df, budget_matches_df, output_dir)


if __name__ == "__main__":
    main()
