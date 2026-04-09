#!/usr/bin/env python3
"""Render concentration-colored spatial trace figures for SRP, SRPv2, and k=15 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
import yaml
from PIL import Image

from benchmark_single_tier_paper_strategies import (
    DEFAULT_MAX_STEPS,
    Experiment,
    MAP_EXT,
    build_planner,
    make_env,
    start_pose_for_track,
    track_config,
)
from f110_planning.utils import load_waypoints
from f110_planning.visualization.svg_trace import (
    SimTrace,
    _pool_wall_mask,
    collect_step,
    render_svg,
)
from low_ccr_per_method_lambda import SUPERVISOR_SETTINGS


METHODS = ("SRP", "SRPv2", "Interval (k=15)")
DISPLAY_METHODS = ("SRPv2", "Interval (k=15)")
MAPS = ("Nuerburgring", "Sochi")
METHOD_COLORS = {
    "SRP": "#14837d",
    "SRPv2": "#d2693c",
    "Interval (k=15)": "#c68b17",
}
REPORT_METHOD_KEYS = {
    "SRP": "SRP (Ours)",
    "SRPv2": "SRPv2",
}
WALL_COLOR = "#4b5563"
CENTERLINE_COLOR = "#cbd5e1"
TRAJECTORY_SHADOW = "#0f172a"
FIG_FACE = "#fffdf8"
TABLE_BLUE = "#4b9bd6"
DENSITY_THRESHOLD = 0.035
DENSITY_SIGMA = 8.5
DENSITY_GAMMA = 0.62


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render trace figures for the best SRP and SRPv2 runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="data/benchmarks/srp_policy_ladder_L10_report.json",
    )
    parser.add_argument(
        "--selected-configs-json",
        type=str,
        default="data/benchmarks/srp_policy_ladder_L10_train_selected_configs.json",
    )
    parser.add_argument(
        "--eval-csv",
        type=str,
        default="data/benchmarks/srp_policy_ladder_L10_eval.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/paper_figures_srp_policy_ladder_L10/traces",
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=",".join(MAPS),
    )
    parser.add_argument("--cloud-latency", type=int, default=10)
    parser.add_argument("--max-laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("(", "").replace(")", "")


def load_best_experiments(report_json: Path, selected_configs_json: Path) -> dict[str, str]:
    payload = json.loads(report_json.read_text())
    selected = payload["selected_representatives"]
    selected_configs = json.loads(selected_configs_json.read_text())
    interval_k15 = next(
        control
        for control in selected_configs["interval_controls"]
        if control["base_experiment"] == "fixed_interval_k15"
    )
    experiments = {
        method: str(selected[REPORT_METHOD_KEYS[method]]["experiment"])
        for method in REPORT_METHOD_KEYS
    }
    experiments["Interval (k=15)"] = str(interval_k15["experiment_name"])
    return experiments


def choose_best_run(eval_df: pd.DataFrame, experiment: str, map_name: str) -> pd.Series:
    rows = eval_df[
        (eval_df["experiment"] == experiment)
        & (eval_df["map_name"] == map_name)
    ].copy()
    if rows.empty:
        raise ValueError(f"No eval rows found for experiment={experiment} map={map_name}")
    return rows.sort_values(
        ["crosstrack_max_m", "crosstrack_rmse_m", "cloud_call_rate", "run_idx"],
        ascending=[True, True, True, True],
        kind="stable",
    ).iloc[0]


def experiment_from_row(row: pd.Series) -> Experiment:
    return Experiment(
        name=str(row["experiment"]),
        strategy=str(row["strategy"]),
        params=json.loads(row["params_json"] or "{}"),
    )


def run_trace_episode(
    row: pd.Series,
    cloud_latency: int,
    max_laps: int,
    max_steps: int,
) -> tuple[SimTrace, np.ndarray]:
    exp = experiment_from_row(row)
    track = track_config(str(row["map_name"]))
    waypoints = load_waypoints(track.waypoints_path)
    planner = build_planner(exp, cloud_latency, SUPERVISOR_SETTINGS, run_idx=int(row["run_idx"]))
    env = make_env(track, max_laps=max_laps)
    obs, _ = env.reset(options={"poses": start_pose_for_track(track)})
    trace = SimTrace()

    done = False
    step_idx = 0
    while not done:
        action = planner.plan(obs, ego_idx=0)
        obs, _, terminated, truncated, _ = env.step(np.array([[action.steer, action.speed]]))
        collect_step(trace, obs, planner, step_idx=step_idx)
        done = bool(terminated or truncated)
        step_idx += 1
        if not done and step_idx >= max_steps:
            break

    env.close()
    trace.total_steps = max(trace.total_steps, step_idx)
    return trace, waypoints


def _load_map_geometry(map_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(map_path + ".yaml", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle)
    resolution = float(meta["resolution"])
    origin_x = float(meta["origin"][0])
    origin_y = float(meta["origin"][1])

    img_raw = np.array(
        Image.open(map_path + MAP_EXT)
        .convert("L")
        .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    )
    h, w = img_raw.shape
    wall_pooled = _pool_wall_mask(img_raw)
    ph, pw = wall_pooled.shape
    x_pool = np.linspace(origin_x, origin_x + w * resolution, pw)
    y_pool = np.linspace(origin_y, origin_y + h * resolution, ph)
    xx, yy = np.meshgrid(x_pool, y_pool)
    return xx, yy, wall_pooled


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def density_cmap(color_hex: str) -> LinearSegmentedColormap:
    r, g, b = _hex_to_rgb(color_hex)
    stops = [
        (0.0, (r, g, b, 0.0)),
        (0.16, (r, g, b, 0.05)),
        (0.38, (r, g, b, 0.16)),
        (0.62, (r, g, b, 0.34)),
        (0.84, (r, g, b, 0.56)),
        (1.0, (r, g, b, 0.80)),
    ]
    cmap_key = color_hex.lstrip("#")
    return LinearSegmentedColormap.from_list(f"density_{cmap_key}", stops)


def concentration_line_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "cloud_concentration_line",
        [
            (0.0, "#22c55e"),
            (0.62, "#65d56e"),
            (0.8, "#d9ef4f"),
            (0.9, "#facc15"),
            (1.0, "#dc2626"),
        ],
    )


def wall_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "wall_mask",
        [
            (0.0, (0.0, 0.0, 0.0, 0.0)),
            (1.0, (*_hex_to_rgb(WALL_COLOR), 1.0)),
        ],
    )


def build_density_field(
    trace: SimTrace,
    xx: np.ndarray,
    yy: np.ndarray,
    wall_pooled: np.ndarray,
) -> np.ma.MaskedArray | None:
    if not trace.cloud_events or not trace.positions:
        return None

    call_xs = np.asarray([event[0] for event in trace.cloud_events], dtype=float)
    call_ys = np.asarray([event[1] for event in trace.cloud_events], dtype=float)
    pos = np.asarray(trace.positions, dtype=float)
    pos_xs = pos[:, 0]
    pos_ys = pos[:, 1]
    x_coords = xx[0, :]
    y_coords = yy[:, 0]
    if x_coords.size < 2 or y_coords.size < 2:
        return None

    x_step = float(np.mean(np.diff(x_coords)))
    y_step = float(np.mean(np.diff(y_coords)))
    x_edges = np.concatenate(([x_coords[0] - x_step / 2.0], x_coords + x_step / 2.0))
    y_edges = np.concatenate(([y_coords[0] - y_step / 2.0], y_coords + y_step / 2.0))

    call_counts, _, _ = np.histogram2d(call_ys, call_xs, bins=[y_edges, x_edges])
    pos_counts, _, _ = np.histogram2d(pos_ys, pos_xs, bins=[y_edges, x_edges])
    call_smooth = gaussian_filter(call_counts, sigma=DENSITY_SIGMA, mode="nearest")
    pos_smooth = gaussian_filter(pos_counts, sigma=DENSITY_SIGMA, mode="nearest")
    concentration = np.divide(
        call_smooth,
        pos_smooth + 1e-6,
        out=np.zeros_like(call_smooth),
        where=pos_smooth > 1e-5,
    )
    valid = (wall_pooled < 0.5) & (pos_smooth > 1e-3)
    if not np.any(valid):
        return None
    peak = float(np.nanmax(concentration[valid]))
    if peak <= 0.0:
        return None
    normalized = np.power(concentration / peak, DENSITY_GAMMA)
    normalized[normalized < DENSITY_THRESHOLD] = np.nan
    normalized[~valid] = np.nan
    return np.ma.masked_invalid(normalized)


def principal_horizontal_transform(
    xx: np.ndarray,
    yy: np.ndarray,
    waypoints: np.ndarray,
    positions: np.ndarray | None,
    cloud_events: np.ndarray | None = None,
) -> tuple[Affine2D, tuple[float, float, float, float]]:
    """Rotate the scene so the dominant track direction reads horizontally."""
    if waypoints.size > 0:
        ref = waypoints[:, :2]
    elif positions is not None and positions.size > 0:
        ref = positions[:, :2]
    else:
        ref = np.column_stack([xx.ravel(), yy.ravel()])

    center = ref.mean(axis=0)
    centered = ref - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    if principal[0] < 0:
        principal = -principal
    angle = float(np.arctan2(principal[1], principal[0]))
    transform = Affine2D().rotate_around(center[0], center[1], -angle)

    bounds_sources = [ref]
    if positions is not None and positions.size > 0:
        bounds_sources.append(positions[:, :2])
    if cloud_events is not None and cloud_events.size > 0:
        bounds_sources.append(cloud_events[:, :2])
    all_source_pts = np.vstack(bounds_sources)
    all_pts = transform.transform(all_source_pts)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    span = max_xy - min_xy
    pad = np.maximum(span * 0.045, 0.12)
    bounds = (
        float(min_xy[0] - pad[0]),
        float(max_xy[0] + pad[0]),
        float(min_xy[1] - pad[1]),
        float(max_xy[1] + pad[1]),
    )
    return transform, bounds


def trajectory_concentration_values(
    positions: np.ndarray,
    density: np.ma.MaskedArray | None,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    if positions.shape[0] < 2:
        return np.asarray([], dtype=float)
    if density is None:
        return np.zeros(positions.shape[0] - 1, dtype=float)
    field = density.filled(0.0)
    interpolator = RegularGridInterpolator(
        (yy[:, 0], xx[0, :]),
        field,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    sampled = interpolator(np.column_stack([positions[:, 1], positions[:, 0]]))
    segment_values = 0.5 * (sampled[:-1] + sampled[1:])
    smoothed = gaussian_filter1d(segment_values, sigma=10.0, mode="nearest")
    scale = max(float(np.percentile(smoothed, 97.0)), 1e-6)
    normalized = np.clip(smoothed / scale, 0.0, 1.0)
    return np.power(normalized, 1.45)


def draw_concentration_trajectory(
    ax: plt.Axes,
    positions: np.ndarray,
    concentration_values: np.ndarray,
    transform: Affine2D,
) -> None:
    if positions.shape[0] < 2:
        return
    segments = np.stack([positions[:-1], positions[1:]], axis=1)
    ax.add_collection(
        LineCollection(
            segments,
            colors=TRAJECTORY_SHADOW,
            linewidths=4.35,
            alpha=0.16,
            zorder=3,
            capstyle="round",
            joinstyle="round",
            transform=transform + ax.transData,
        )
    )
    ax.add_collection(
        LineCollection(
            segments,
            array=concentration_values,
            cmap=concentration_line_cmap(),
            linewidths=2.18,
            alpha=0.98,
            zorder=4,
            capstyle="round",
            joinstyle="round",
            transform=transform + ax.transData,
        )
    )


def draw_density_panel(
    ax: plt.Axes,
    trace: SimTrace,
    map_path: str,
    waypoints: np.ndarray,
    path_color: str,
    metrics_row: pd.Series,
) -> None:
    xx, yy, wall_pooled = _load_map_geometry(map_path)
    ax.set_facecolor(FIG_FACE)
    positions = np.asarray(trace.positions) if trace.positions else np.empty((0, 2))
    cloud_events = (
        np.asarray([(event[0], event[1]) for event in trace.cloud_events], dtype=float)
        if trace.cloud_events
        else np.empty((0, 2))
    )
    scene_transform, bounds = principal_horizontal_transform(
        xx,
        yy,
        waypoints,
        positions,
        cloud_events,
    )
    wall_mask = np.ma.masked_where(wall_pooled < 0.5, wall_pooled)
    ax.imshow(
        wall_mask,
        origin="lower",
        extent=(float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max())),
        cmap=wall_cmap(),
        interpolation="nearest",
        zorder=0,
        transform=scene_transform + ax.transData,
    )
    density = build_density_field(trace, xx, yy, wall_pooled)
    if waypoints.size > 0:
        ax.plot(
            waypoints[:, 0],
            waypoints[:, 1],
            color=CENTERLINE_COLOR,
            linewidth=0.85,
            alpha=0.7,
            zorder=2,
            transform=scene_transform + ax.transData,
        )
    if trace.positions:
        pos = positions
        concentration_values = trajectory_concentration_values(pos, density, xx, yy)
        draw_concentration_trajectory(ax, pos, concentration_values, scene_transform)
    ax.text(
        0.50,
        0.035,
        (
            f"CCR {float(metrics_row['cloud_call_rate']) * 100.0:.1f}% | "
            f"Calls {int(metrics_row['total_cloud_calls'])}"
        ),
        transform=ax.transAxes,
        fontsize=9.2,
        color="#334155",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "#fffdf8",
            "edgecolor": "#cbd5e1",
            "alpha": 0.92,
        },
    )
    ax.set_aspect("equal")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.axis("off")


def save_individual_panel(
    output_dir: Path,
    stem: str,
    trace: SimTrace,
    map_path: str,
    waypoints: np.ndarray,
    path_color: str,
    metrics_row: pd.Series,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(6.3, 4.6))
    fig.patch.set_facecolor(FIG_FACE)
    ax = fig.add_subplot(1, 1, 1)
    draw_density_panel(
        ax=ax,
        trace=trace,
        map_path=map_path,
        waypoints=waypoints,
        path_color=path_color,
        metrics_row=metrics_row,
    )
    for fmt in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_combined_figure(
    payloads: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
        }
    )
    fig = plt.figure(figsize=(13.3, 7.6))
    fig.patch.set_facecolor(FIG_FACE)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.22, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.045,
        right=0.985,
        top=0.89,
        bottom=0.055,
        wspace=0.03,
        hspace=0.055,
    )

    row_label_axes = [fig.add_subplot(grid[row, 0]) for row in range(2)]
    panel_axes = [
        [fig.add_subplot(grid[row, col]) for col in range(1, 3)]
        for row in range(2)
    ]

    for ax in row_label_axes:
        ax.axis("off")

    payload_lookup = {(p["map_name"], p["method"]): p for p in payloads}
    for row_idx, map_name in enumerate(MAPS):
        for col_idx, method in enumerate(DISPLAY_METHODS):
            payload = payload_lookup[(map_name, method)]
            ax = panel_axes[row_idx][col_idx]
            draw_density_panel(
                ax=ax,
                trace=payload["trace"],
                map_path=payload["track"].map_path,
                waypoints=payload["waypoints"],
                path_color=payload["path_color"],
                metrics_row=payload["metrics_row"],
            )

    for row_idx, map_name in enumerate(MAPS):
        bbox = row_label_axes[row_idx].get_position()
        fig.text(
            bbox.x0 + 0.035,
            (bbox.y0 + bbox.y1) / 2.0,
            map_name,
            ha="center",
            va="center",
            fontsize=19,
            color="#2b2b2b",
        )

    top_axes = [panel_axes[0][i] for i in range(2)]
    for method, ax in zip(DISPLAY_METHODS, top_axes):
        bbox = ax.get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2.0,
            0.922,
            method,
            ha="center",
            va="bottom",
            fontsize=18,
            color="#2b2b2b",
        )

    left_bbox = row_label_axes[0].get_position()
    first_panel_bbox = panel_axes[0][0].get_position()
    second_panel_bbox = panel_axes[0][1].get_position()
    lower_row_bbox = row_label_axes[1].get_position()

    line_kwargs = dict(transform=fig.transFigure, color=TABLE_BLUE, linewidth=1.8, alpha=0.85)
    y_mid = (row_label_axes[0].get_position().y0 + row_label_axes[1].get_position().y1) / 2.0
    x_sep1 = left_bbox.x1 + 0.008
    x_sep2 = first_panel_bbox.x1 + 0.008
    x_right = second_panel_bbox.x1
    y_bottom = lower_row_bbox.y0
    header_gap = 0.018
    y_header = top_axes[0].get_position().y1 + header_gap
    corner_x = x_sep1
    corner_y = y_header

    fig.text(
        left_bbox.x0 + 0.030,
        y_header + 0.006,
        "Track",
        ha="center",
        va="bottom",
        fontsize=17,
        color=TABLE_BLUE,
        fontweight="bold",
    )
    fig.text(
        left_bbox.x0 + 0.145,
        y_header + 0.028,
        "Strategy",
        ha="center",
        va="bottom",
        fontsize=17,
        color=TABLE_BLUE,
        fontweight="bold",
    )

    fig.add_artist(Line2D([x_sep1, x_sep1], [y_bottom, y_header], **line_kwargs))
    fig.add_artist(Line2D([x_sep2, x_sep2], [y_bottom, y_header], **line_kwargs))
    fig.add_artist(Line2D([x_sep1, x_right], [y_header, y_header], **line_kwargs))
    fig.add_artist(Line2D([left_bbox.x0 + 0.036, corner_x], [y_header + 0.052, corner_y], **line_kwargs))
    fig.add_artist(Line2D([left_bbox.x0 + 0.03, x_right], [y_mid, y_mid], **line_kwargs))

    stem = output_dir / "cloud_calls_density_panels"
    for fmt in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = load_best_experiments(
        Path(args.report_json),
        Path(args.selected_configs_json),
    )
    eval_df = pd.read_csv(args.eval_csv)
    maps = [value.strip() for value in args.maps.split(",") if value.strip()]

    payloads: list[dict[str, Any]] = []
    for map_name in maps:
        track = track_config(map_name)
        for method in DISPLAY_METHODS:
            experiment = best[method]
            metrics_row = choose_best_run(eval_df, experiment, map_name)
            trace, waypoints = run_trace_episode(
                metrics_row,
                cloud_latency=args.cloud_latency,
                max_laps=args.max_laps,
                max_steps=args.max_steps,
            )
            method_slug = _slug(method.replace(" ", "_"))
            map_slug = _slug(map_name)

            render_svg(
                trace,
                track.map_path,
                MAP_EXT,
                waypoints,
                output_path=str(output_dir / f"{map_slug}__{method_slug}__raw_trace.svg"),
                path_color=METHOD_COLORS[method],
                cloud_alpha=0.22,
                path_linewidth=1.8,
            )

            stem = f"{map_slug}__{method_slug}"
            save_individual_panel(
                output_dir=output_dir,
                stem=stem,
                trace=trace,
                map_path=track.map_path,
                waypoints=waypoints,
                path_color=METHOD_COLORS[method],
                metrics_row=metrics_row,
                dpi=args.dpi,
            )
            payloads.append(
                {
                    "map_name": map_name,
                    "method": method,
                    "trace": trace,
                    "track": track,
                    "waypoints": waypoints,
                    "path_color": METHOD_COLORS[method],
                    "metrics_row": metrics_row,
                }
            )

    render_combined_figure(payloads, output_dir, dpi=args.dpi)
    print(f"Trace figures written to {output_dir}")


if __name__ == "__main__":
    main()
