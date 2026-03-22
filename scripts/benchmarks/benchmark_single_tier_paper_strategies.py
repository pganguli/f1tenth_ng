#!/usr/bin/env python3
"""Canonical benchmark: corrected 10-map single-tier Strategy 1-5 sweep.

Reads: F1 map assets, per-map start poses, and the edge/cloud model checkpoints.
Writes: canonical single-tier benchmark artifacts under ``data/benchmarks``.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from f110_gym.envs.base_classes import Integrator
from f110_planning.metrics import MetricAggregator
from f110_planning.reactive import EdgeCloudPlanner
from f110_planning.schedulers import (
    AlwaysCloudScheduler,
    BernoulliMaxMissConfig,
    BernoulliMaxMissPolicy,
    DeterministicDeviationConfig,
    DeterministicDeviationPolicy,
    ExponentialRiskConfig,
    ExponentialRiskPolicy,
    FixedBernoulliConfig,
    FixedBernoulliPolicy,
    LogisticRiskConfig,
    LogisticRiskPolicy,
    PiecewiseLinearRampConfig,
    PiecewiseLinearRampPolicy,
    PolicyDrivenScheduler,
)
from f110_planning.utils import load_waypoints
from f110_planning.utils.sim_utils import (
    DEFAULT_START_THETA,
    DEFAULT_START_X,
    DEFAULT_START_Y,
    load_start_pose_from_yaml,
)


def available_map_names() -> list[str]:
    """Return the available F1 track directory names."""
    root = Path("data/maps/F1")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


DEFAULT_MAPS = [
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
DEFAULT_CLOUD_LATENCY = 5
DEFAULT_TRIALS = 3
DEFAULT_MAX_STEPS = 15000
DEFAULT_ALPHA_LEFT = 0.2
DEFAULT_ALPHA_TRACK = 0.2
DEFAULT_ALPHA_HEADING = 0.7
DEFAULT_DEVIATION_STEER_WEIGHT = 1.0
DEFAULT_DEVIATION_SPEED_WEIGHT = 1.0
TARGET_CCR_LOW = 0.55
TARGET_CCR_HIGH = 0.65
MAP_EXT = ".png"


@dataclass(frozen=True)
class TrackConfig:
    """Map and waypoint paths for a benchmark track."""

    name: str
    map_path: str
    waypoints_path: str


@dataclass(frozen=True)
class PlannerSettings:
    """Fixed planner settings shared across the benchmark sweep."""

    alpha_left: float = DEFAULT_ALPHA_LEFT
    alpha_track: float = DEFAULT_ALPHA_TRACK
    alpha_heading: float = DEFAULT_ALPHA_HEADING
    deviation_steer_weight: float = DEFAULT_DEVIATION_STEER_WEIGHT
    deviation_speed_weight: float = DEFAULT_DEVIATION_SPEED_WEIGHT


@dataclass(frozen=True)
class Experiment:
    """Benchmark configuration for one scheduler variant."""

    name: str
    strategy: str
    params: dict[str, Any]


def _json_default(value: Any) -> Any:
    """Convert numpy and NaN payload values into JSON-safe primitives."""
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value):
            return None
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Canonical single-tier Strategy 1-5 benchmark for the corrected 10-map workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Status: canonical. Example: "
            "python scripts/benchmarks/benchmark_single_tier_paper_strategies.py"
        ),
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=",".join(DEFAULT_MAPS),
        help="Comma-separated list of F1 map names under data/maps/F1.",
    )
    parser.add_argument(
        "--cloud-latency",
        type=int,
        default=DEFAULT_CLOUD_LATENCY,
        help="Single-tier cloud latency in simulation steps.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of repeated trials per configuration.",
    )
    parser.add_argument(
        "--max-laps",
        type=int,
        default=1,
        help="Maximum laps per episode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Safety cap on simulator steps per episode.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="single_tier_paper_strategies_10maps",
        help="Base filename stem for outputs under data/benchmarks.",
    )
    parser.add_argument(
        "--alpha-left",
        type=float,
        default=DEFAULT_ALPHA_LEFT,
        help="Cloud blend weight for the left-distance feature.",
    )
    parser.add_argument(
        "--alpha-track",
        type=float,
        default=DEFAULT_ALPHA_TRACK,
        help="Cloud blend weight for the track-width feature.",
    )
    parser.add_argument(
        "--alpha-heading",
        type=float,
        default=DEFAULT_ALPHA_HEADING,
        help="Cloud blend weight for the heading-error feature.",
    )
    parser.add_argument(
        "--deviation-steer-weight",
        type=float,
        default=DEFAULT_DEVIATION_STEER_WEIGHT,
        help="Weight on steering disagreement when forming scheduler deviation.",
    )
    parser.add_argument(
        "--deviation-speed-weight",
        type=float,
        default=DEFAULT_DEVIATION_SPEED_WEIGHT,
        help="Weight on speed disagreement when forming scheduler deviation.",
    )
    return parser.parse_args()


def track_config(map_name: str) -> TrackConfig:
    """Return map and waypoint paths for a named F1 track."""
    track_dir = Path("data/maps/F1") / map_name
    stem = "MexicoCity" if map_name == "Mexico City" else map_name
    return TrackConfig(
        name=map_name,
        map_path=str(track_dir / f"{stem}_map"),
        waypoints_path=str(track_dir / f"{stem}_centerline.tsv"),
    )


def base_model_paths(arch_id: int) -> dict[str, str]:
    """Return the three model paths for a single DNN architecture."""
    return {
        "left": f"data/models/left_wall_dist_arch{arch_id}.pt",
        "track_width": f"data/models/track_width_arch{arch_id}.pt",
        "heading": f"data/models/heading_error_arch{arch_id}.pt",
    }


def start_pose_for_track(track: TrackConfig) -> np.ndarray:
    """Return the simulator reset pose for a track using its YAML start pose."""
    pose = load_start_pose_from_yaml(track.map_path)
    if pose is None:
        pose = (DEFAULT_START_X, DEFAULT_START_Y, DEFAULT_START_THETA)
    return np.array([list(pose)], dtype=float)


def default_experiments() -> list[Experiment]:
    """Return the default paper-strategy sweep."""
    experiments = [Experiment("always_hit", "always", {})]

    for probability in (0.55, 0.60, 0.65):
        suffix = int(round(probability * 100))
        experiments.append(
            Experiment(
                f"fixed_bernoulli_p{suffix}",
                "fixed_bernoulli",
                {"p": probability, "seed": 7},
            )
        )

    for probability in (0.55, 0.60, 0.65):
        for max_miss in (3, 5):
            suffix = int(round(probability * 100))
            experiments.append(
                Experiment(
                    f"bernoulli_max_miss_p{suffix}_m{max_miss}",
                    "bernoulli_max_miss",
                    {"p": probability, "max_miss": max_miss, "seed": 7},
                )
            )

    for center in (0.02, 0.03):
        for slope in (30.0, 50.0):
            experiments.append(
                Experiment(
                    f"logistic_c{str(center).replace('.', 'p')}_s{int(slope)}",
                    "logistic",
                    {"center": center, "slope": slope, "seed": 7},
                )
            )

    for center in (0.02, 0.03):
        for rate in (15.0, 30.0):
            experiments.append(
                Experiment(
                    f"exponential_c{str(center).replace('.', 'p')}_r{int(rate)}",
                    "exponential",
                    {"center": center, "rate": rate, "seed": 7},
                )
            )

    for d_low, d_high in ((0.01, 0.04), (0.02, 0.05), (0.03, 0.06)):
        experiments.append(
            Experiment(
                f"piecewise_ramp_{str(d_low).replace('.', 'p')}_{str(d_high).replace('.', 'p')}",
                "piecewise_ramp",
                {"d_low": d_low, "d_high": d_high, "seed": 7},
            )
        )

    for threshold in (0.02, 0.03, 0.05):
        experiments.append(
            Experiment(
                f"deterministic_t{str(threshold).replace('.', 'p')}",
                "deterministic",
                {"threshold": threshold},
            )
        )

    return experiments


def make_env(track: TrackConfig, max_laps: int) -> Any:
    """Create a headless single-agent environment."""
    return gym.make(
        "f110_gym:f110-v0",
        map=track.map_path,
        map_ext=MAP_EXT,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode=None,
        max_laps=max_laps,
    )


def trial_seed(exp: Experiment, run_idx: int) -> int | None:
    """Return the effective random seed for a trial, if the strategy uses one."""
    base_seed = exp.params.get("seed")
    if base_seed is None:
        return None
    return int(base_seed) + int(run_idx)


def build_planner(
    exp: Experiment,
    cloud_latency: int,
    settings: PlannerSettings,
    run_idx: int = 0,
) -> EdgeCloudPlanner:
    """Instantiate the single-tier planner for one benchmark configuration."""
    edge = base_model_paths(2)
    cloud = base_model_paths(6)
    seed = trial_seed(exp, run_idx)

    if exp.strategy == "always":
        scheduler = AlwaysCloudScheduler()
    elif exp.strategy == "fixed_bernoulli":
        scheduler = PolicyDrivenScheduler(
            FixedBernoulliPolicy(
                FixedBernoulliConfig(
                    p=float(exp.params["p"]),
                    min_interval=1,
                    warmup_steps=0,
                    fallback_interval=0,
                    seed=seed,
                )
            )
        )
    elif exp.strategy == "bernoulli_max_miss":
        scheduler = PolicyDrivenScheduler(
            BernoulliMaxMissPolicy(
                BernoulliMaxMissConfig(
                    p=float(exp.params["p"]),
                    min_interval=1,
                    warmup_steps=0,
                    fallback_interval=0,
                    max_miss=int(exp.params["max_miss"]),
                    seed=seed,
                )
            )
        )
    elif exp.strategy == "logistic":
        scheduler = PolicyDrivenScheduler(
            LogisticRiskPolicy(
                LogisticRiskConfig(
                    center=float(exp.params["center"]),
                    slope=float(exp.params["slope"]),
                    p_min=0.0,
                    p_max=1.0,
                    min_interval=1,
                    warmup_steps=0,
                    fallback_interval=0,
                    seed=seed,
                )
            )
        )
    elif exp.strategy == "exponential":
        scheduler = PolicyDrivenScheduler(
            ExponentialRiskPolicy(
                ExponentialRiskConfig(
                    center=float(exp.params["center"]),
                    rate=float(exp.params["rate"]),
                    p_min=0.0,
                    p_max=1.0,
                    min_interval=1,
                    warmup_steps=0,
                    fallback_interval=0,
                    seed=seed,
                )
            )
        )
    elif exp.strategy == "piecewise_ramp":
        scheduler = PolicyDrivenScheduler(
            PiecewiseLinearRampPolicy(
                PiecewiseLinearRampConfig(
                    d_low=float(exp.params["d_low"]),
                    d_high=float(exp.params["d_high"]),
                    min_interval=1,
                    warmup_steps=0,
                    fallback_interval=0,
                    seed=seed,
                )
            )
        )
    elif exp.strategy == "deterministic":
        scheduler = PolicyDrivenScheduler(
            DeterministicDeviationPolicy(
                DeterministicDeviationConfig(
                    threshold=float(exp.params["threshold"]),
                    min_interval=1,
                    warmup_steps=0,
                    fallback_interval=0,
                )
            )
        )
    else:
        raise ValueError(f"Unsupported strategy: {exp.strategy}")

    return EdgeCloudPlanner(
        cloud_latency=cloud_latency,
        scheduler=scheduler,
        alpha_left=settings.alpha_left,
        alpha_track=settings.alpha_track,
        alpha_heading=settings.alpha_heading,
        deviation_steer_weight=settings.deviation_steer_weight,
        deviation_speed_weight=settings.deviation_speed_weight,
        lookahead_distance=1.5,
        lateral_gain=1.0,
        edge_left_wall_model_path=edge["left"],
        edge_track_width_model_path=edge["track_width"],
        edge_heading_model_path=edge["heading"],
        cloud_left_wall_model_path=cloud["left"],
        cloud_track_width_model_path=cloud["track_width"],
        cloud_heading_model_path=cloud["heading"],
    )


def call_gap_stats(call_steps: list[int]) -> tuple[float, float]:
    """Return mean and max cloud-call gaps in simulation steps."""
    if len(call_steps) < 2:
        return 0.0, 0.0
    gaps = np.diff(np.array(call_steps, dtype=float))
    return float(np.mean(gaps)), float(np.max(gaps))


def run_episode(
    exp: Experiment,
    track: TrackConfig,
    cloud_latency: int,
    max_laps: int,
    settings: PlannerSettings,
    run_idx: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Run one benchmark episode and collect metrics plus scheduler statistics."""
    waypoints = load_waypoints(track.waypoints_path)
    planner = build_planner(exp, cloud_latency, settings, run_idx=run_idx)
    env = make_env(track, max_laps=max_laps)
    obs, _ = env.reset(options={"poses": start_pose_for_track(track)})
    metrics = MetricAggregator.create_default(waypoints=waypoints)
    metrics.on_reset(obs, waypoints=waypoints)

    call_steps: list[int] = []
    collision_steps = 0
    done = False
    step_cap_hit = False
    step_idx = 0
    start_wall = time.perf_counter()

    while not done:
        action = planner.plan(obs, ego_idx=0)
        if getattr(planner, "last_cloud_call", False):
            call_steps.append(step_idx)

        obs, reward, terminated, truncated, _ = env.step(
            np.array([[action.steer, action.speed]])
        )
        metrics.on_step(obs, action, float(reward), ego_idx=0)
        collision_steps += int(bool(obs["collisions"][0] > 0))
        done = bool(terminated or truncated)
        step_idx += 1
        if not done and step_idx >= max_steps:
            done = True
            step_cap_hit = True

    wall_time_s = time.perf_counter() - start_wall
    env.close()
    with contextlib.redirect_stdout(io.StringIO()):
        report = metrics.report()

    mean_gap, max_gap = call_gap_stats(call_steps)
    steps = max(1.0, float(report.get("steps", float(step_idx))))
    total_cloud_calls = float(len(call_steps))
    cloud_call_rate = total_cloud_calls / steps
    collision_count = int(report.get("collision", 0.0))
    effective_seed = trial_seed(exp, run_idx)

    report.update(
        {
            "experiment": exp.name,
            "strategy": exp.strategy,
            "map_name": track.name,
            "map_path": track.map_path,
            "waypoints_path": track.waypoints_path,
            "cloud_latency": float(cloud_latency),
            "run_idx": float(run_idx),
            "seed": float(effective_seed) if effective_seed is not None else np.nan,
            "total_cloud_calls": total_cloud_calls,
            "cloud_call_rate": round(float(cloud_call_rate), 4),
            "mean_call_gap_steps": round(float(mean_gap), 4),
            "max_call_gap_steps": round(float(max_gap), 4),
            "collision_count": float(collision_count),
            "collision_steps": float(collision_steps),
            "step_cap_hit": float(step_cap_hit),
            "wall_time_s": round(float(wall_time_s), 4),
            "rt_factor": (
                round(float(report["lap_time_s"]) / wall_time_s, 4)
                if wall_time_s > 0.0
                else 0.0
            ),
            "params_json": json.dumps(exp.params, sort_keys=True),
        }
    )
    return report


def summarize(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate per-trial results and select top-ranked configurations."""
    experiments_df = pd.DataFrame(results)
    group_cols = ["map_name", "cloud_latency", "experiment", "strategy"]

    summary_df = (
        experiments_df.groupby(group_cols, as_index=False)
        .agg(
            trials=("run_idx", "count"),
            collision_rate=("collision", "mean"),
            step_cap_rate=("step_cap_hit", "mean"),
            collision_count_mean=("collision_count", "mean"),
            collision_steps_mean=("collision_steps", "mean"),
            laps_completed_mean=("laps_completed", "mean"),
            lap_time_s_mean=("lap_time_s", "mean"),
            crosstrack_rmse_m_mean=("crosstrack_rmse_m", "mean"),
            crosstrack_rmse_m_std=("crosstrack_rmse_m", "std"),
            crosstrack_mean_m_mean=("crosstrack_mean_m", "mean"),
            crosstrack_max_m_mean=("crosstrack_max_m", "mean"),
            heading_error_rmse_deg_mean=("heading_error_rmse_deg", "mean"),
            wall_min_distance_m_mean=("wall_min_distance_m", "mean"),
            speed_mean_m_s_mean=("speed_mean_m_s", "mean"),
            steering_rate_mean_rad_s_mean=("steering_rate_mean_rad_s", "mean"),
            total_cloud_calls_mean=("total_cloud_calls", "mean"),
            cloud_call_rate_mean=("cloud_call_rate", "mean"),
            cloud_call_rate_std=("cloud_call_rate", "std"),
            mean_call_gap_steps_mean=("mean_call_gap_steps", "mean"),
            max_call_gap_steps_max=("max_call_gap_steps", "max"),
            rt_factor_mean=("rt_factor", "mean"),
        )
        .fillna(0.0)
    )

    summary_df["collision_free_rate"] = 1.0 - summary_df["collision_rate"]
    summary_df["in_target_ccr_band"] = summary_df["cloud_call_rate_mean"].between(
        TARGET_CCR_LOW,
        TARGET_CCR_HIGH,
    )
    summary_df = summary_df.sort_values(
        [
            "map_name",
            "cloud_latency",
            "collision_rate",
            "step_cap_rate",
            "crosstrack_rmse_m_mean",
            "cloud_call_rate_mean",
            "experiment",
        ],
        kind="stable",
    ).reset_index(drop=True)
    summary_df["rank"] = (
        summary_df.groupby(["map_name", "cloud_latency"]).cumcount() + 1
    )

    best_overall_df = (
        summary_df.groupby(["map_name", "cloud_latency"], as_index=False)
        .first()
        .sort_values(["map_name", "cloud_latency"], kind="stable")
        .reset_index(drop=True)
    )

    near_target_rows: list[pd.Series] = []
    for (_, _), group in summary_df.groupby(["map_name", "cloud_latency"], sort=False):
        candidates = group[group["in_target_ccr_band"]]
        if not candidates.empty:
            near_target_rows.append(candidates.iloc[0])
    near_target_df = (
        pd.DataFrame(near_target_rows).reset_index(drop=True)
        if near_target_rows
        else pd.DataFrame(columns=summary_df.columns)
    )

    return experiments_df, summary_df, near_target_df


def maybe_write_xlsx(frames: dict[str, pd.DataFrame], out_path: Path) -> str | None:
    """Write an Excel workbook if a supported engine is available."""
    engine: str | None = None
    try:
        import openpyxl  # pylint: disable=unused-import,import-outside-toplevel

        engine = "openpyxl"
    except ModuleNotFoundError:
        try:
            import xlsxwriter  # pylint: disable=unused-import,import-outside-toplevel

            engine = "xlsxwriter"
        except ModuleNotFoundError:
            return None

    with pd.ExcelWriter(out_path, engine=engine) as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return engine


def main() -> None:
    """Run the benchmark sweep and save JSON/CSV/XLSX reports."""
    args = parse_args()
    tracks = [track_config(name.strip()) for name in args.maps.split(",") if name.strip()]
    cloud_latency = int(args.cloud_latency)
    settings = PlannerSettings(
        alpha_left=args.alpha_left,
        alpha_track=args.alpha_track,
        alpha_heading=args.alpha_heading,
        deviation_steer_weight=args.deviation_steer_weight,
        deviation_speed_weight=args.deviation_speed_weight,
    )
    experiments = default_experiments()

    results = [
        run_episode(
            exp,
            track,
            cloud_latency,
            max_laps=args.max_laps,
            settings=settings,
            run_idx=run_idx,
            max_steps=args.max_steps,
        )
        for track in tracks
        for exp in experiments
        for run_idx in range(args.trials)
    ]

    experiments_df, summary_df, near_target_df = summarize(results)
    best_overall_df = (
        summary_df.groupby(["map_name", "cloud_latency"], as_index=False)
        .first()
        .sort_values(["map_name", "cloud_latency"], kind="stable")
        .reset_index(drop=True)
    )

    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    best_csv_path = out_dir / f"{stem}_best_target_band.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tracks": [track.__dict__ for track in tracks],
        "cloud_latency": cloud_latency,
        "cloud_latencies": [cloud_latency],
        "settings": settings.__dict__,
        "experiments": results,
        "summary": summary_df.to_dict(orient="records"),
        "best_overall": best_overall_df.to_dict(orient="records"),
        "best_near_target_band": near_target_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    experiments_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    near_target_df.to_csv(best_csv_path, index=False)
    engine = maybe_write_xlsx(
        {
            "experiments": experiments_df,
            "summary": summary_df,
            "best_overall": best_overall_df,
            "best_target_band": near_target_df,
        },
        xlsx_path,
    )

    print(summary_df.to_string(index=False))
    print(f"\nWrote JSON report to {json_path}")
    print(f"Wrote experiment CSV to {csv_path}")
    print(f"Wrote summary CSV to {summary_csv_path}")
    print(f"Wrote target-band CSV to {best_csv_path}")
    if engine is None:
        print("Skipped XLSX export because neither openpyxl nor xlsxwriter is installed.")
    else:
        print(f"Wrote XLSX workbook to {xlsx_path} using {engine}.")


if __name__ == "__main__":
    main()
