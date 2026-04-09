#!/usr/bin/env python3
"""Paper-facing plots for the focused SRP/SRPv2 policy ladder study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "Fixed Interval": "#c68b17",
    "SRP (Ours)": "#14837d",
    "SRPv2": "#d2693c",
    "Never query": "#9ca3af",
    "Always query": "#6b7280",
}
MARKERS = {
    "Fixed Interval": "s",
    "SRP (Ours)": "D",
    "SRPv2": "v",
    "Never query": "+",
    "Always query": "o",
}
EDGE_COLORS = {
    "Fixed Interval": "#5b3b0d",
    "SRP (Ours)": "#27424c",
    "SRPv2": "#5d3626",
    "Never query": "#4b5563",
    "Always query": "#374151",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render figures for the SRP/SRPv2 policy ladder study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cross-map-csv",
        type=str,
        default="data/benchmarks/srp_policy_ladder_L10_eval_cross_map.csv",
    )
    parser.add_argument(
        "--budget-matches-csv",
        type=str,
        default="data/benchmarks/srp_policy_ladder_L10_eval_budget_matches.csv",
    )
    parser.add_argument(
        "--paired-delta-csv",
        type=str,
        default="data/benchmarks/srp_policy_ladder_L10_eval_paired_delta.csv",
    )
    parser.add_argument(
        "--per-map-csv",
        type=str,
        default="data/benchmarks/srp_policy_ladder_L10_eval_per_map.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/paper_figures_srp_policy_ladder_L10",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--formats", type=str, default="png,pdf")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def apply_plot_style() -> None:
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
        }
    )


def _best_by_strategy(frame: pd.DataFrame, strategy: str) -> pd.Series:
    family = frame[frame["strategy"] == strategy].copy()
    return family.sort_values(
        ["acceptable", "mean_crosstrack_max", "mean_cloud_call_rate", "experiment"],
        ascending=[False, True, True, True],
        kind="stable",
    ).iloc[0]


def figure_tradeoff(
    cross_map_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_title("Max CTE vs Cloud Call Rate", fontsize=20, pad=12)
    ax.set_xlabel("Cloud call rate (%)", fontsize=13)
    ax.set_ylabel("Max cross-track error (cm, mean across eval maps)", fontsize=13)
    ax.grid(True)

    interval = cross_map_df[cross_map_df["strategy"] == "fixed_interval"].copy().sort_values(
        "mean_cloud_call_rate",
        kind="stable",
    )
    ax.plot(
        interval["mean_cloud_call_rate"] * 100.0,
        interval["mean_crosstrack_max_cm"],
        color=COLORS["Fixed Interval"],
        linewidth=1.7,
        alpha=0.7,
        zorder=1,
    )
    ax.scatter(
        interval["mean_cloud_call_rate"] * 100.0,
        interval["mean_crosstrack_max_cm"],
        s=110,
        marker=MARKERS["Fixed Interval"],
        color=COLORS["Fixed Interval"],
        edgecolor=EDGE_COLORS["Fixed Interval"],
        linewidth=0.95,
        label="Fixed Interval",
        zorder=2,
    )

    for strategy, label in (
        ("self_normalizing_momentum", "SRP (Ours)"),
        ("srpv2", "SRPv2"),
    ):
        family = cross_map_df[cross_map_df["strategy"] == strategy].copy().sort_values(
            "mean_cloud_call_rate",
            kind="stable",
        )
        ax.scatter(
            family["mean_cloud_call_rate"] * 100.0,
            family["mean_crosstrack_max_cm"],
            s=90,
            marker=MARKERS[label],
            color=COLORS[label],
            edgecolor=EDGE_COLORS[label],
            linewidth=0.9,
            label=label,
            zorder=3,
        )

    annotations = [
        ("self_normalizing_momentum", "SRP best"),
        ("srpv2", "SRPv2 best"),
    ]
    for strategy, tag in annotations:
        best = _best_by_strategy(cross_map_df, strategy)
        label = "SRP (Ours)" if strategy == "self_normalizing_momentum" else "SRPv2"
        ax.scatter(
            [best["mean_cloud_call_rate"] * 100.0],
            [best["mean_crosstrack_max_cm"]],
            s=180,
            facecolors="none",
            edgecolors=EDGE_COLORS[label],
            linewidths=1.2,
            zorder=4,
        )
        ax.annotate(
            f"{tag}\n{best['mean_crosstrack_max_cm']:.1f} cm, {best['mean_cloud_call_rate'] * 100.0:.1f}%",
            xy=(best["mean_cloud_call_rate"] * 100.0, best["mean_crosstrack_max_cm"]),
            xytext=(10, -15 if strategy == "self_normalizing_momentum" else 8),
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

    for base_name in ("fixed_interval_k2", "fixed_interval_k4", "fixed_interval_k10", "fixed_interval_k15"):
        row = interval[interval["base_experiment"] == base_name]
        if row.empty:
            continue
        chosen = row.iloc[0]
        k_value = int(base_name.rsplit("k", maxsplit=1)[1])
        ax.annotate(
            f"k={k_value}",
            xy=(chosen["mean_cloud_call_rate"] * 100.0, chosen["mean_crosstrack_max_cm"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=EDGE_COLORS["Fixed Interval"],
        )

    ax.set_xlim(left=0.0)
    ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    save_figure(fig, output_dir, "max_cte_vs_cloud_call_rate", formats, dpi)


def figure_pareto_frontier(
    cross_map_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.1))
    ax.set_title("Pareto Frontier: Minimize Calls and Max CTE", fontsize=19, pad=12)
    ax.set_xlabel("Cloud call rate (%)", fontsize=13)
    ax.set_ylabel("Max cross-track error (cm, mean across eval maps)", fontsize=13)
    ax.grid(True)

    frame = cross_map_df[
        cross_map_df["strategy"].isin(["fixed_interval", "self_normalizing_momentum", "srpv2"])
    ].copy()
    for strategy, label in (
        ("fixed_interval", "Fixed Interval"),
        ("self_normalizing_momentum", "SRP (Ours)"),
        ("srpv2", "SRPv2"),
    ):
        family = frame[frame["strategy"] == strategy].copy().sort_values(
            "mean_cloud_call_rate",
            kind="stable",
        )
        ax.scatter(
            family["mean_cloud_call_rate"] * 100.0,
            family["mean_crosstrack_max_cm"],
            s=95,
            marker=MARKERS[label],
            color=COLORS[label],
            edgecolor=EDGE_COLORS[label],
            linewidth=0.9,
            alpha=0.8,
            label=label,
        )

    frontier = frame[frame["pareto_frontier"]].copy().sort_values(
        "mean_cloud_call_rate",
        kind="stable",
    )
    ax.plot(
        frontier["mean_cloud_call_rate"] * 100.0,
        frontier["mean_crosstrack_max_cm"],
        color="#111827",
        linewidth=1.5,
        linestyle="--",
        alpha=0.75,
    )
    ax.scatter(
        frontier["mean_cloud_call_rate"] * 100.0,
        frontier["mean_crosstrack_max_cm"],
        s=165,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.2,
    )
    for row in frontier.itertuples(index=False):
        if row.strategy not in {"self_normalizing_momentum", "srpv2"}:
            continue
        label = "SRP" if row.strategy == "self_normalizing_momentum" else "SRPv2"
        ax.annotate(
            label,
            xy=(row.mean_cloud_call_rate * 100.0, row.mean_crosstrack_max_cm),
            xytext=(8, 8 if label == "SRPv2" else -14),
            textcoords="offset points",
            fontsize=9.2,
            color=EDGE_COLORS["SRP (Ours)" if label == "SRP" else "SRPv2"],
        )
    ax.set_xlim(left=0.0)
    ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    save_figure(fig, output_dir, "pareto_frontier_max_cte", formats, dpi)


def figure_budget_matched(
    budget_matches_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    ax.set_title("Budget-Matched Δ Max CTE vs Interval", fontsize=19, pad=12)
    ax.set_xlabel("Cloud call rate (%)", fontsize=13)
    ax.set_ylabel("Δ Max CTE vs matched interval (cm)", fontsize=13)
    ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.3)
    ax.grid(True)

    for label in ("SRP (Ours)", "SRPv2"):
        family = budget_matches_df[budget_matches_df["display_strategy"] == label].copy().sort_values(
            "mean_cloud_call_rate",
            kind="stable",
        )
        ax.plot(
            family["mean_cloud_call_rate"] * 100.0,
            family["delta_max_cte_vs_interval_cm"],
            color=COLORS[label],
            linewidth=1.5,
            alpha=0.5,
        )
        ax.scatter(
            family["mean_cloud_call_rate"] * 100.0,
            family["delta_max_cte_vs_interval_cm"],
            s=100,
            marker=MARKERS[label],
            color=COLORS[label],
            edgecolor=EDGE_COLORS[label],
            linewidth=0.9,
            label=label,
        )
    ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    save_figure(fig, output_dir, "budget_matched_vs_interval", formats, dpi)


def figure_interval_ladder_only(
    cross_map_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.set_title("Fixed-Interval Ladder (k=2..15)", fontsize=19, pad=12)
    ax.set_xlabel("Cloud call rate (%)", fontsize=13)
    ax.set_ylabel("Max cross-track error (cm, mean across eval maps)", fontsize=13)
    ax.grid(True)

    interval = cross_map_df[cross_map_df["strategy"] == "fixed_interval"].copy().sort_values(
        "mean_cloud_call_rate",
        kind="stable",
    )
    ax.plot(
        interval["mean_cloud_call_rate"] * 100.0,
        interval["mean_crosstrack_max_cm"],
        color=COLORS["Fixed Interval"],
        linewidth=1.8,
    )
    ax.scatter(
        interval["mean_cloud_call_rate"] * 100.0,
        interval["mean_crosstrack_max_cm"],
        s=105,
        marker=MARKERS["Fixed Interval"],
        color=COLORS["Fixed Interval"],
        edgecolor=EDGE_COLORS["Fixed Interval"],
        linewidth=0.95,
    )
    for row in interval.itertuples(index=False):
        params = json.loads(row.params_json or "{}") if isinstance(row.params_json, str) else {}
        interval_k = params.get("interval")
        if interval_k is None:
            continue
        ax.annotate(
            f"k={interval_k}",
            xy=(row.mean_cloud_call_rate * 100.0, row.mean_crosstrack_max_cm),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.8,
            color=EDGE_COLORS["Fixed Interval"],
        )
    save_figure(fig, output_dir, "interval_ladder_only", formats, dpi)


def figure_per_map_comparison(
    per_map_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), sharex=True, sharey=True)
    fig.suptitle("Per-Map Max CTE vs Matched Interval", fontsize=19, y=1.01)
    max_value = max(
        float(per_map_df["target_max_cte_cm"].max()),
        float(per_map_df["interval_max_cte_cm"].max()),
    )
    min_value = min(
        float(per_map_df["target_max_cte_cm"].min()),
        float(per_map_df["interval_max_cte_cm"].min()),
    )
    padding = 0.7
    for ax, label in zip(axes, ("SRP (Ours)", "SRPv2"), strict=True):
        family = per_map_df[per_map_df["family"] == label].copy()
        ax.grid(True)
        ax.plot(
            [min_value - padding, max_value + padding],
            [min_value - padding, max_value + padding],
            linestyle="--",
            color="#64748b",
            linewidth=1.2,
        )
        ax.scatter(
            family["interval_max_cte_cm"],
            family["target_max_cte_cm"],
            s=90,
            marker=MARKERS[label],
            color=COLORS[label],
            edgecolor=EDGE_COLORS[label],
            linewidth=0.9,
        )
        for row in family.itertuples(index=False):
            ax.annotate(
                row.map_name,
                xy=(row.interval_max_cte_cm, row.target_max_cte_cm),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8.3,
            )
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Matched interval Max CTE (cm)", fontsize=12)
    axes[0].set_ylabel("Policy Max CTE (cm)", fontsize=12)
    save_figure(fig, output_dir, "per_map_srpx_vs_interval", formats, dpi)


def figure_paired_delta(
    paired_delta_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    order = ["Never query", "Fixed Interval", "SRP (Ours)", "SRPv2"]
    panels = ["Sochi", "Spa", "Aggregate"]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8), sharey=True)
    fig.suptitle("Latency-10 Paired Δ Max CTE vs Always-query", fontsize=20, y=1.02)
    for ax, panel in zip(axes, panels, strict=True):
        frame = paired_delta_df[paired_delta_df["panel"] == panel].copy()
        frame["label"] = pd.Categorical(frame["label"], order, ordered=True)
        frame = frame.sort_values("label", kind="stable")
        ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.3)
        ax.grid(True, axis="y")
        for idx, row in enumerate(frame.itertuples(index=False)):
            low = float(row.mean_delta_max_cte_vs_always_cm - row.ci95_low_cm)
            high = float(row.ci95_high_cm - row.mean_delta_max_cte_vs_always_cm)
            marker_kwargs = {}
            if row.label != "Never query":
                marker_kwargs = {
                    "edgecolor": EDGE_COLORS.get(row.label, "#475569"),
                    "linewidth": 0.9,
                }
            ax.errorbar(
                idx,
                row.mean_delta_max_cte_vs_always_cm,
                yerr=[[low], [high]],
                fmt=MARKERS[row.label],
                color=COLORS[row.label],
                markersize=9,
                elinewidth=1.6,
                capsize=4,
            )
            ax.scatter(
                [idx],
                [row.mean_delta_max_cte_vs_always_cm],
                s=75,
                marker=MARKERS[row.label],
                color=COLORS[row.label],
                **marker_kwargs,
            )
        ax.set_title(panel, fontsize=15)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=24, ha="right", fontsize=10.5)
    axes[0].set_ylabel("Δ Max CTE vs always (cm)", fontsize=12)
    save_figure(fig, output_dir, "paired_delta_vs_always", formats, dpi)


def write_tables(
    cross_map_df: pd.DataFrame,
    budget_matches_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for label, strategy in (
        ("Always query", "always"),
        ("Never query", "never_query"),
        ("SRP (Ours)", "self_normalizing_momentum"),
        ("SRPv2", "srpv2"),
    ):
        best = _best_by_strategy(cross_map_df, strategy)
        rows.append(
            {
                "Method": label,
                "Experiment": str(best["experiment"]),
                "CCR (%)": float(best["mean_cloud_call_rate"]) * 100.0,
                "Max CTE (cm)": float(best["mean_crosstrack_max_cm"]),
                "RMSE (cm)": float(best["mean_crosstrack_rmse_cm"]),
                "Collision rate": float(best["mean_collision_rate"]),
                "Acceptable": bool(best["acceptable"]),
            }
        )
    for base_name in ("fixed_interval_k2", "fixed_interval_k4", "fixed_interval_k10", "fixed_interval_k15"):
        subset = cross_map_df[cross_map_df["base_experiment"] == base_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "Method": f"Fixed interval (k={int(base_name.rsplit('k', maxsplit=1)[1])})",
                "Experiment": str(row["experiment"]),
                "CCR (%)": float(row["mean_cloud_call_rate"]) * 100.0,
                "Max CTE (cm)": float(row["mean_crosstrack_max_cm"]),
                "RMSE (cm)": float(row["mean_crosstrack_rmse_cm"]),
                "Collision rate": float(row["mean_collision_rate"]),
                "Acceptable": bool(row["acceptable"]),
            }
        )
    headline_df = pd.DataFrame(rows)
    headline_df.to_csv(output_dir / "headline_performance_table.csv", index=False)
    headline_df.to_latex(
        output_dir / "headline_performance_table.tex",
        index=False,
        float_format=lambda x: f"{x:.3f}",
        escape=False,
    )

    budget_df = budget_matches_df[
        [
            "display_strategy",
            "experiment",
            "matched_interval_k",
            "mean_cloud_call_rate",
            "mean_crosstrack_max_cm",
            "delta_max_cte_vs_interval_cm",
        ]
    ].rename(
        columns={
            "display_strategy": "Method",
            "experiment": "Experiment",
            "matched_interval_k": "Matched interval k",
            "mean_cloud_call_rate": "CCR",
            "mean_crosstrack_max_cm": "Max CTE (cm)",
            "delta_max_cte_vs_interval_cm": "$\\Delta$ Max CTE (cm)",
        }
    )
    budget_df["CCR"] = budget_df["CCR"] * 100.0
    budget_df.to_csv(output_dir / "budget_match_table.csv", index=False)
    budget_df.to_latex(
        output_dir / "budget_match_table.tex",
        index=False,
        float_format=lambda x: f"{x:.3f}",
        escape=False,
    )


def main() -> None:
    args = parse_args()
    apply_plot_style()
    cross_map_df = pd.read_csv(args.cross_map_csv)
    budget_matches_df = pd.read_csv(args.budget_matches_csv)
    paired_delta_df = pd.read_csv(args.paired_delta_csv)
    per_map_df = pd.read_csv(args.per_map_csv)
    output_dir = Path(args.output_dir)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    figure_tradeoff(cross_map_df, output_dir, formats, args.dpi)
    figure_pareto_frontier(cross_map_df, output_dir, formats, args.dpi)
    figure_budget_matched(budget_matches_df, output_dir, formats, args.dpi)
    figure_interval_ladder_only(cross_map_df, output_dir, formats, args.dpi)
    figure_per_map_comparison(per_map_df, output_dir, formats, args.dpi)
    figure_paired_delta(paired_delta_df, output_dir, formats, args.dpi)
    write_tables(cross_map_df, budget_matches_df, output_dir)


if __name__ == "__main__":
    main()
