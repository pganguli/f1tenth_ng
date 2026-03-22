#!/usr/bin/env python3
"""Exploratory plot: legacy single-tier analysis views for internal use.

Reads: single-tier benchmark summaries from ``data/benchmarks``.
Writes: non-canonical diagnostic plots under an internal output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


STRATEGY_COLORS = {
    "always": "#24221d",
    "fixed_bernoulli": "#1f8a70",
    "bernoulli_max_miss": "#5b8c2a",
    "logistic": "#6e59a5",
    "exponential": "#d4538c",
    "piecewise_ramp": "#cf6a32",
    "deterministic": "#2d6fb7",
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
    "fixed_bernoulli": "Fixed Bernoulli",
    "bernoulli_max_miss": "Bernoulli + Guard",
    "logistic": "Logistic",
    "exponential": "Exponential",
    "piecewise_ramp": "Piecewise Ramp",
    "deterministic": "Threshold",
}

PREFERRED_MAP_ORDER = [
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "Hockenheim",
    "IMS",
    "Melbourne",
    "MexicoCity",
    "Montreal",
    "Monza",
    "MoscowRaceway",
    "Nuerburgring",
    "Oschersleben",
    "Sakhir",
    "SaoPaulo",
    "Sepang",
    "Shanghai",
    "Silverstone",
    "Sochi",
    "Spa",
    "Spielberg",
    "YasMarina",
    "Zandvoort",
]

BG = "#f5efe6"
PANEL = "#fbf7f2"
TEXT = "#1e1915"
MUTED = "#76695d"
GRID = "#d9cbbb"
BAND = "#efe2cc"
GOOD = "#2b8a78"
MID = "#d6a84a"
BAD = "#d46a4c"

SUCCESS_CMAP = LinearSegmentedColormap.from_list(
    "success_map",
    ["#bc4b51", "#e6d5a8", "#2b8a78"],
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate legacy exploratory single-tier analysis plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Status: exploratory/internal only. Example: "
            "python scripts/benchmarks/plot_single_tier_paper_strategies.py "
            "--output-dir data/benchmarks/internal_single_tier_plots"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="data/benchmarks/single_tier_paper_strategies_summary.csv",
        help="Path to the aggregated benchmark summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/internal_single_tier_plots",
        help="Directory for the generated plot files.",
    )
    parser.add_argument(
        "--target-low",
        type=float,
        default=0.55,
        help="Lower bound of the target cloud-call-rate band.",
    )
    parser.add_argument(
        "--target-high",
        type=float,
        default=0.65,
        help="Upper bound of the target cloud-call-rate band.",
    )
    return parser.parse_args()


def apply_theme() -> None:
    """Apply a restrained poster-style plotting theme."""
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.edgecolor": "#b8aa97",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_summary(path: str) -> pd.DataFrame:
    """Load and order the summary data."""
    summary = pd.read_csv(path)
    present = set(summary["map_name"].dropna().astype(str))
    ordered_maps = [name for name in PREFERRED_MAP_ORDER if name in present]
    extras = sorted(present.difference(ordered_maps))
    summary["map_name"] = pd.Categorical(
        summary["map_name"],
        ordered_maps + extras,
        ordered=True,
    )
    summary["cloud_latency"] = summary["cloud_latency"].astype(float)
    return summary.sort_values(["map_name", "cloud_latency", "rank"], kind="stable")


def latency_order(summary: pd.DataFrame) -> list[float]:
    """Return the latencies present in the summary, in ascending order."""
    return sorted(float(value) for value in summary["cloud_latency"].dropna().unique())


def ensure_output_dir(path: str) -> Path:
    """Create the plot directory if needed."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def blend_with_white(color: str, amount: float) -> tuple[float, float, float]:
    """Lighten a hex color by blending it with white."""
    red, green, blue = mpl.colors.to_rgb(color)
    return (
        red + (1.0 - red) * amount,
        green + (1.0 - green) * amount,
        blue + (1.0 - blue) * amount,
    )


def short_experiment_label(record: pd.Series) -> str:
    """Return a compact label for an experiment configuration."""
    strategy = record["strategy"]
    name = str(record["experiment"])

    if strategy == "always":
        return "always-hit"
    if strategy == "fixed_bernoulli":
        return f"fixed p={record['cloud_call_rate_mean']:.2f}"
    if strategy == "bernoulli_max_miss":
        tail = name.split("_p", maxsplit=1)[-1]
        if "_m" in tail:
            probability, miss = tail.split("_m", maxsplit=1)
            probability = probability.replace("p", ".")
            return f"guard p={probability} m={miss}"
        return "guard"
    if strategy == "deterministic" and "_t" in name:
        return f"threshold {name.split('_t', maxsplit=1)[-1].replace('p', '.')}"
    if strategy == "piecewise_ramp":
        tail = name.split("piecewise_ramp_", maxsplit=1)[-1]
        low, high = tail.split("_", maxsplit=1)
        return f"ramp {low.replace('p', '.')} to {high.replace('p', '.')}"
    if strategy == "logistic":
        tail = name.split("logistic_", maxsplit=1)[-1]
        center, slope = tail.split("_s", maxsplit=1)
        return f"logistic {center.replace('c', '').replace('p', '.')} / {slope}"
    if strategy == "exponential":
        tail = name.split("exponential_", maxsplit=1)[-1]
        center, rate = tail.split("_r", maxsplit=1)
        return f"expo {center.replace('c', '').replace('p', '.')} / {rate}"
    return name


def collision_badge_color(value: float) -> str:
    """Return a status color for collision-free rate."""
    if value >= 0.99:
        return GOOD
    if value > 0.0:
        return MID
    return BAD


def add_title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Draw a consistent poster-like title block."""
    fig.text(0.06, 0.965, title, fontsize=22, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.928, subtitle, fontsize=11.5, color=MUTED, ha="left", va="top")


def best_overall_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the rank-1 row per map and latency."""
    return (
        summary.sort_values(["map_name", "cloud_latency", "rank"], kind="stable")
        .groupby(["map_name", "cloud_latency"], observed=False)
        .head(1)
        .copy()
    )


def best_band_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the best in-band row per map and latency when available."""
    target = summary[summary["in_target_ccr_band"]].copy()
    if target.empty:
        return target
    return (
        target.sort_values(["map_name", "cloud_latency", "rank"], kind="stable")
        .groupby(["map_name", "cloud_latency"], observed=False)
        .head(1)
        .copy()
    )


def plot_success_heatmap(
    summary: pd.DataFrame,
    out_dir: Path,
    latencies: list[float],
) -> Path:
    """Plot viability and best-in-cell recommendations as a styled matrix."""
    map_order = list(summary["map_name"].cat.categories)
    best = best_overall_rows(summary)
    best_band = best_band_rows(summary)
    collision_free_counts = (
        summary.groupby("map_name", observed=False)["collision_rate"]
        .apply(lambda values: int((values == 0.0).sum()))
        .to_dict()
    )

    fig, ax = plt.subplots(figsize=(13.5, max(8.2, 0.46 * len(map_order) + 2.6)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, len(latencies))
    ax.set_ylim(0, len(map_order))
    ax.axis("off")

    add_title_block(
        fig,
        "Where The Sweep Actually Works",
        "Each cell shows the best overall config for a map/latency pair, with the best 55-65% CCR option underneath.",
    )

    for col_idx, latency in enumerate(latencies):
        ax.text(
            col_idx + 0.5,
            len(map_order) + 0.08,
            f"Latency {int(latency)}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=TEXT,
        )

    for row_idx, map_name in enumerate(map_order):
        ax.text(
            -0.18,
            len(map_order) - row_idx - 0.5,
            map_name,
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=TEXT,
        )
        ax.text(
            -0.18,
            len(map_order) - row_idx - 0.74,
            f"{collision_free_counts.get(map_name, 0)} collision-free configs",
            ha="right",
            va="center",
            fontsize=9.5,
            color=MUTED,
        )

    for row_idx, map_name in enumerate(map_order):
        y_base = len(map_order) - row_idx - 1
        for col_idx, latency in enumerate(latencies):
            cell = best[
                (best["map_name"] == map_name)
                & (best["cloud_latency"].astype(float) == latency)
            ].iloc[0]
            band_match = best_band[
                (best_band["map_name"] == map_name)
                & (best_band["cloud_latency"].astype(float) == latency)
            ]
            band = band_match.iloc[0] if not band_match.empty else None

            fill = SUCCESS_CMAP(float(cell["collision_free_rate"]))
            rect = Rectangle(
                (col_idx + 0.04, y_base + 0.06),
                0.92,
                0.88,
                facecolor=fill,
                edgecolor="#a2917d",
                linewidth=1.2,
            )
            ax.add_patch(rect)

            badge = FancyBboxPatch(
                (col_idx + 0.08, y_base + 0.77),
                0.18,
                0.11,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor=collision_badge_color(float(cell["collision_free_rate"])),
                edgecolor="none",
            )
            ax.add_patch(badge)
            ax.text(
                col_idx + 0.17,
                y_base + 0.825,
                f"CF {cell['collision_free_rate']:.2f}",
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="bold",
                color="white",
            )

            body_color = TEXT if float(cell["collision_free_rate"]) >= 0.34 else "white"
            muted_color = MUTED if body_color == TEXT else "#f4ebe2"
            ax.text(
                col_idx + 0.08,
                y_base + 0.66,
                "BEST OVERALL",
                fontsize=7.8,
                fontweight="bold",
                color=muted_color,
                ha="left",
                va="top",
            )
            ax.text(
                col_idx + 0.08,
                y_base + 0.57,
                short_experiment_label(cell),
                fontsize=10.5,
                fontweight="bold",
                color=body_color,
                ha="left",
                va="top",
            )
            ax.text(
                col_idx + 0.08,
                y_base + 0.47,
                f"RMSE {cell['crosstrack_rmse_m_mean']:.3f}   CCR {cell['cloud_call_rate_mean']:.3f}",
                fontsize=8.8,
                color=body_color,
                ha="left",
                va="top",
            )

            divider_y = y_base + 0.39
            ax.plot(
                [col_idx + 0.08, col_idx + 0.88],
                [divider_y, divider_y],
                color=blend_with_white(body_color, 0.5) if body_color != TEXT else "#c9b9a8",
                linewidth=0.8,
                alpha=0.9,
            )

            if band is not None:
                ax.text(
                    col_idx + 0.08,
                    y_base + 0.31,
                    "55-65% CCR PICK",
                    fontsize=7.8,
                    fontweight="bold",
                    color=muted_color,
                    ha="left",
                    va="top",
                )
                ax.text(
                    col_idx + 0.08,
                    y_base + 0.23,
                    short_experiment_label(band),
                    fontsize=9.7,
                    fontweight="bold",
                    color=body_color,
                    ha="left",
                    va="top",
                )
                ax.text(
                    col_idx + 0.08,
                    y_base + 0.14,
                    (
                        f"RMSE {band['crosstrack_rmse_m_mean']:.3f}   "
                        f"CCR {band['cloud_call_rate_mean']:.3f}   "
                        f"CF {band['collision_free_rate']:.2f}"
                    ),
                    fontsize=8.5,
                    color=body_color,
                    ha="left",
                    va="top",
                )
            else:
                ax.text(
                    col_idx + 0.08,
                    y_base + 0.22,
                    "No config entered the target CCR band.",
                    fontsize=8.5,
                    color=body_color,
                    ha="left",
                    va="top",
                )

    fig.tight_layout(rect=(0.08, 0.06, 0.98, 0.9))
    out_path = out_dir / "single_tier_paper_strategies_success_heatmap.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tradeoff_grid(
    summary: pd.DataFrame,
    out_dir: Path,
    target_low: float,
    target_high: float,
    latencies: list[float],
) -> Path:
    """Plot the full tradeoff surface with restrained hierarchy."""
    map_order = list(summary["map_name"].cat.categories)
    fig, axes = plt.subplots(
        len(map_order),
        len(latencies),
        figsize=(15.5, max(11.5, 1.65 * len(map_order) + 1.5)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(BG)

    y_min = float(summary["crosstrack_rmse_m_mean"].min()) - 0.006
    y_max = float(summary["crosstrack_rmse_m_mean"].max()) + 0.01

    add_title_block(
        fig,
        "Accuracy vs Communication Tradeoff",
        "Shaded band marks the 55-65% CCR target. Filled points are collision-free; faded points are not. Stars mark best overall.",
    )

    for row_idx, map_name in enumerate(map_order):
        for col_idx, latency in enumerate(latencies):
            ax = axes[row_idx, col_idx]
            subset = summary[
                (summary["map_name"] == map_name)
                & (summary["cloud_latency"].astype(float) == latency)
            ].copy()
            ax.set_facecolor(PANEL)
            ax.axvspan(target_low, target_high, color=BAND, alpha=0.9, zorder=0)
            ax.grid(alpha=0.35, linewidth=0.7)
            ax.set_xlim(0.54, 1.01)
            ax.set_ylim(y_min, y_max)

            collision_free_configs = int((subset["collision_rate"] == 0.0).sum())
            ax.text(
                0.02,
                0.96,
                f"CF configs {collision_free_configs}/{len(subset)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.3,
                color=MUTED,
                bbox={
                    "boxstyle": "round,pad=0.28,rounding_size=0.12",
                    "facecolor": BG,
                    "edgecolor": "#d8cab9",
                },
            )

            for _, record in subset.iterrows():
                base_color = STRATEGY_COLORS[record["strategy"]]
                is_safe = float(record["collision_free_rate"]) == 1.0
                face = base_color if is_safe else blend_with_white(base_color, 0.55)
                edge = "#181512" if is_safe else blend_with_white(base_color, 0.22)
                alpha = 0.95 if is_safe else 0.48
                size = 76 if is_safe else 56
                ax.scatter(
                    record["cloud_call_rate_mean"],
                    record["crosstrack_rmse_m_mean"],
                    s=size,
                    c=[face],
                    marker=STRATEGY_MARKERS[record["strategy"]],
                    alpha=alpha,
                    edgecolors=edge,
                    linewidths=0.9,
                    zorder=3,
                )

            best_overall = subset.nsmallest(1, "rank").iloc[0]
            ax.scatter(
                best_overall["cloud_call_rate_mean"],
                best_overall["crosstrack_rmse_m_mean"],
                s=220,
                marker="*",
                c="#f2c14e",
                edgecolors="#181512",
                linewidths=1.0,
                zorder=6,
            )

            in_band = subset[subset["in_target_ccr_band"]]
            if not in_band.empty:
                best_band = in_band.nsmallest(1, "rank").iloc[0]
                ax.scatter(
                    best_band["cloud_call_rate_mean"],
                    best_band["crosstrack_rmse_m_mean"],
                    s=150,
                    marker="s",
                    facecolors="none",
                    edgecolors="#8f5d12",
                    linewidths=1.8,
                    zorder=7,
                )

            if row_idx == 0:
                ax.set_title(f"Latency {int(latency)}", fontsize=12.5, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"{map_name}\nCross-Track RMSE (m)")
            if row_idx == len(map_order) - 1:
                ax.set_xlabel("Cloud Call Rate")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=STRATEGY_MARKERS[strategy],
            color="w",
            markerfacecolor=color,
            markeredgecolor="#181512",
            markersize=8.5,
            label=label,
        )
        for strategy, label in STRATEGY_DISPLAY.items()
        for color in [STRATEGY_COLORS[strategy]]
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="#f2c14e",
                markeredgecolor="#181512",
                markersize=12,
                label="Best overall",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="#8f5d12",
                markerfacecolor="none",
                markeredgecolor="#8f5d12",
                linewidth=0,
                markersize=10,
                label="Best in 55-65% CCR band",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.06, 1, 0.9))

    out_path = out_dir / "single_tier_paper_strategies_tradeoff_grid.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_target_band_focus(summary: pd.DataFrame, out_dir: Path) -> Path:
    """Plot only the best target-band config per map/latency."""
    target = summary[summary["in_target_ccr_band"]].copy()
    best_band = best_band_rows(summary)
    best_band["label"] = best_band.apply(
        lambda row: f"{row['map_name']} @ {int(float(row['cloud_latency']))}",
        axis=1,
    )
    best_band = best_band.sort_values(["map_name", "cloud_latency"], kind="stable")

    full_success_count = int((best_band["collision_free_rate"] == 1.0).sum())

    fig, ax = plt.subplots(figsize=(12, max(7, 0.34 * len(best_band) + 2.8)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    add_title_block(
        fig,
        "Target-Band Decision View",
        f"Best config inside the 55-65% CCR band for each map/latency pair. Full collision-free wins: {full_success_count}/{len(best_band)}.",
    )

    y_positions = np.arange(len(best_band))
    bar_colors = [
        blend_with_white(STRATEGY_COLORS[strategy], 0.22)
        for strategy in best_band["strategy"]
    ]
    bars = ax.barh(
        y_positions,
        best_band["crosstrack_rmse_m_mean"],
        color=bar_colors,
        edgecolor="#181512",
        linewidth=0.8,
        height=0.78,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(best_band["label"], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-Track RMSE (m)")
    ax.grid(axis="x", alpha=0.28, linewidth=0.7)
    ax.set_axisbelow(True)

    for index in range(1, len(best_band)):
        if best_band.iloc[index]["map_name"] != best_band.iloc[index - 1]["map_name"]:
            ax.axhline(index - 0.5, color=GRID, linewidth=1.1)

    for bar, (_, row) in zip(bars, best_band.iterrows()):
        status_color = collision_badge_color(float(row["collision_free_rate"]))
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2.0,
            (
                f"{short_experiment_label(row)}  |  "
                f"{STRATEGY_DISPLAY[row['strategy']]}  |  "
                f"CCR {row['cloud_call_rate_mean']:.3f}"
            ),
            va="center",
            fontsize=9.6,
            color=TEXT,
        )
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2.0 + 0.23,
            f"CF {row['collision_free_rate']:.2f}",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.24,rounding_size=0.12",
                "facecolor": status_color,
                "edgecolor": "none",
            },
        )

    fig.tight_layout(rect=(0.04, 0.06, 1, 0.9))
    out_path = out_dir / "single_tier_paper_strategies_target_band_focus.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Generate the plot set."""
    args = parse_args()
    apply_theme()
    summary = load_summary(args.summary_csv)
    latencies = latency_order(summary)
    out_dir = ensure_output_dir(args.output_dir)

    heatmap_path = plot_success_heatmap(summary, out_dir, latencies)
    tradeoff_path = plot_tradeoff_grid(
        summary,
        out_dir,
        args.target_low,
        args.target_high,
        latencies,
    )
    target_band_path = plot_target_band_focus(summary, out_dir)

    print(f"Wrote {heatmap_path}")
    print(f"Wrote {tradeoff_path}")
    print(f"Wrote {target_band_path}")


if __name__ == "__main__":
    main()
