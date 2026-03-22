#!/usr/bin/env python3
"""Benchmark multi-tier cloud routing with fixed default delay settings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from f110_gym.envs.base_classes import Integrator
from f110_planning.base import Action, CloudTier, TieredCloudScheduler
from f110_planning.metrics import MetricAggregator
from f110_planning.reactive.edge_cloud_planner import EdgeCloudPlanner
from f110_planning.reactive.lidar_dnn_planner import LidarDNNPlanner
from f110_planning.reactive.multi_tier_edge_cloud_planner import MultiTierEdgeCloudPlanner
from f110_planning.schedulers import AlwaysCloudScheduler
from f110_planning.schedulers.safety_policies import (
    TieredProbabilisticRiskConfig,
    TieredProbabilisticRiskPolicy,
    TieredThresholdConfig,
    TieredThresholdPolicy,
)
from f110_planning.schedulers.tiered_policy_driven_scheduler import (
    TieredPolicyDrivenScheduler,
)
from f110_planning.utils import load_waypoints
from f110_planning.utils.sim_utils import (
    DEFAULT_START_THETA,
    DEFAULT_START_X,
    DEFAULT_START_Y,
    load_start_pose_from_yaml,
)


DEFAULT_DELAY_PAIRS = [(3, 5)]
DEFAULT_MAX_STEPS = 15000
MAP_EXT = ".png"


@dataclass(frozen=True)
class TrackConfig:
    """Map and waypoint paths for a benchmark track."""

    name: str
    map_path: str
    waypoints_path: str


@dataclass(frozen=True)
class DelayConfig:
    """Cloud latency configuration."""

    medium_latency: int
    large_latency: int


@dataclass(frozen=True)
class Experiment:
    """Benchmark configuration."""

    name: str
    kind: str
    params: dict[str, Any]


def available_map_names() -> list[str]:
    """Return all F1 track directories present in the repo."""
    maps_root = Path("data/maps/F1")
    return sorted(
        entry.name for entry in maps_root.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )


DEFAULT_MAPS = available_map_names()


class FixedTierScheduler(TieredCloudScheduler):  # pylint: disable=too-few-public-methods
    """Always request the same cloud tier."""

    def __init__(self, tier: CloudTier | None) -> None:
        self.tier = tier

    def choose_cloud_tier(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_actions: dict[CloudTier, Action | None],
        context: dict[str, Any] | None = None,
    ) -> CloudTier | None:
        del step, obs, latest_cloud_actions, context
        return self.tier


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark threshold and probabilistic multi-tier cloud routing."
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=",".join(DEFAULT_MAPS),
        help="Comma-separated list of map names under data/maps/F1.",
    )
    parser.add_argument(
        "--delay-pairs",
        type=str,
        default=",".join(f"{m}:{l}" for m, l in DEFAULT_DELAY_PAIRS),
        help="Comma-separated medium:large latency pairs, defaulting to a fixed 3:5 setup.",
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
        default="multi_tier_cloud_benchmark_feature_fusion",
        help="Base filename stem for benchmark outputs in data/benchmarks.",
    )
    return parser.parse_args()


def track_config(map_name: str) -> TrackConfig:
    """Return map and waypoint paths for a named F1 track."""
    track_dir = Path("data/maps/F1") / map_name
    if map_name == "Mexico City":
        stem = "MexicoCity"
    else:
        stem = map_name
    return TrackConfig(
        name=map_name,
        map_path=str(track_dir / f"{stem}_map"),
        waypoints_path=str(track_dir / f"{stem}_centerline.tsv"),
    )


def start_pose_for_track(track: TrackConfig) -> np.ndarray:
    """Return the simulator reset pose for a track using its YAML start pose."""
    pose = load_start_pose_from_yaml(track.map_path)
    if pose is None:
        pose = (DEFAULT_START_X, DEFAULT_START_Y, DEFAULT_START_THETA)
    return np.array([list(pose)], dtype=float)


def parse_delay_pairs(raw: str) -> list[DelayConfig]:
    """Parse CLI delay pairs."""
    pairs: list[DelayConfig] = []
    for item in raw.split(","):
        medium_raw, large_raw = item.strip().split(":")
        pairs.append(
            DelayConfig(
                medium_latency=int(medium_raw),
                large_latency=int(large_raw),
            )
        )
    return pairs


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


def base_model_paths(arch_id: int) -> dict[str, str]:
    """Return the three model paths for a single architecture."""
    return {
        "left": f"data/models/left_wall_dist_arch{arch_id}.pt",
        "track_width": f"data/models/track_width_arch{arch_id}.pt",
        "heading": f"data/models/heading_error_arch{arch_id}.pt",
    }


def default_experiments() -> list[Experiment]:
    """Return the benchmark sweep."""
    return [
        Experiment("edge_only_arch2", "edge_only", {}),
        Experiment(
            "large_always_arch6",
            "large_always",
            {
                "alpha_left": 0.2,
                "alpha_track": 0.2,
                "alpha_heading": 0.7,
            },
        ),
        Experiment(
            "threshold_conservative",
            "tiered_threshold",
            {
                "medium_threshold": 0.03,
                "large_threshold": 0.09,
                "use_relative_thresholds": True,
                "history_window": 256,
                "min_history": 64,
                "medium_quantile": 0.65,
                "large_quantile": 0.9,
                "min_interval": 1,
                "warmup_steps": 0,
                "fallback_interval": 0,
                "max_miss": 20,
                "alpha_left_medium": 0.1,
                "alpha_track_medium": 0.1,
                "alpha_heading_medium": 0.5,
                "alpha_left_large": 0.2,
                "alpha_track_large": 0.2,
                "alpha_heading_large": 0.7,
            },
        ),
        Experiment(
            "threshold_large_sparing",
            "tiered_threshold",
            {
                "medium_threshold": 0.05,
                "large_threshold": 0.14,
                "use_relative_thresholds": True,
                "history_window": 256,
                "min_history": 64,
                "medium_quantile": 0.75,
                "large_quantile": 0.95,
                "min_interval": 1,
                "warmup_steps": 0,
                "fallback_interval": 0,
                "max_miss": 20,
                "alpha_left_medium": 0.1,
                "alpha_track_medium": 0.1,
                "alpha_heading_medium": 0.5,
                "alpha_left_large": 0.2,
                "alpha_track_large": 0.2,
                "alpha_heading_large": 0.7,
            },
        ),
        Experiment(
            "prob_medium_favoring",
            "tiered_probabilistic",
            {
                "p_medium": 0.3,
                "p_large": 0.2,
                "min_interval": 1,
                "warmup_steps": 0,
                "fallback_interval": 0,
                "max_miss": 20,
                "seed": 7,
                "alpha_left_medium": 0.1,
                "alpha_track_medium": 0.1,
                "alpha_heading_medium": 0.5,
                "alpha_left_large": 0.2,
                "alpha_track_large": 0.2,
                "alpha_heading_large": 0.7,
            },
        ),
        Experiment(
            "prob_balanced",
            "tiered_probabilistic",
            {
                "p_medium": 0.4,
                "p_large": 0.2,
                "min_interval": 1,
                "warmup_steps": 0,
                "fallback_interval": 0,
                "max_miss": 20,
                "seed": 13,
                "alpha_left_medium": 0.1,
                "alpha_track_medium": 0.1,
                "alpha_heading_medium": 0.5,
                "alpha_left_large": 0.2,
                "alpha_track_large": 0.2,
                "alpha_heading_large": 0.7,
            },
        ),
    ]


def build_planner(exp: Experiment, delays: DelayConfig) -> Any:
    """Instantiate the planner for a benchmark experiment."""
    edge = base_model_paths(2)
    medium = base_model_paths(5)
    large = base_model_paths(6)

    if exp.kind == "edge_only":
        return LidarDNNPlanner(
            left_model_path=edge["left"],
            track_width_model_path=edge["track_width"],
            heading_model_path=edge["heading"],
            lookahead_distance=1.5,
            lateral_gain=1.0,
        )

    if exp.kind == "large_always":
        return EdgeCloudPlanner(
            cloud_latency=delays.large_latency,
            scheduler=AlwaysCloudScheduler(),
            alpha_left=float(exp.params.get("alpha_left", 0.2)),
            alpha_track=float(exp.params.get("alpha_track", 0.2)),
            alpha_heading=float(exp.params.get("alpha_heading", 0.7)),
            lookahead_distance=1.5,
            lateral_gain=1.0,
            edge_left_wall_model_path=edge["left"],
            edge_track_width_model_path=edge["track_width"],
            edge_heading_model_path=edge["heading"],
            cloud_left_wall_model_path=large["left"],
            cloud_track_width_model_path=large["track_width"],
            cloud_heading_model_path=large["heading"],
        )

    common_kwargs = {
        "medium_cloud_latency": delays.medium_latency,
        "large_cloud_latency": delays.large_latency,
        "alpha_left_medium": float(exp.params.get("alpha_left_medium", 0.1)),
        "alpha_track_medium": float(exp.params.get("alpha_track_medium", 0.1)),
        "alpha_heading_medium": float(exp.params.get("alpha_heading_medium", 0.5)),
        "alpha_left_large": float(exp.params.get("alpha_left_large", 0.2)),
        "alpha_track_large": float(exp.params.get("alpha_track_large", 0.2)),
        "alpha_heading_large": float(exp.params.get("alpha_heading_large", 0.7)),
        "lookahead_distance": 1.5,
        "lateral_gain": 1.0,
        "edge_left_wall_model_path": edge["left"],
        "edge_track_width_model_path": edge["track_width"],
        "edge_heading_model_path": edge["heading"],
        "medium_left_wall_model_path": medium["left"],
        "medium_track_width_model_path": medium["track_width"],
        "medium_heading_model_path": medium["heading"],
        "large_left_wall_model_path": large["left"],
        "large_track_width_model_path": large["track_width"],
        "large_heading_model_path": large["heading"],
    }

    if exp.kind == "tiered_threshold":
        scheduler = TieredPolicyDrivenScheduler(
            TieredThresholdPolicy(
                TieredThresholdConfig(
                    medium_threshold=float(exp.params["medium_threshold"]),
                    large_threshold=float(exp.params["large_threshold"]),
                    use_relative_thresholds=bool(exp.params.get("use_relative_thresholds", False)),
                    history_window=int(exp.params.get("history_window", 256)),
                    min_history=int(exp.params.get("min_history", 64)),
                    medium_quantile=float(exp.params.get("medium_quantile", 0.6)),
                    large_quantile=float(exp.params.get("large_quantile", 0.9)),
                    min_interval=int(exp.params["min_interval"]),
                    warmup_steps=int(exp.params["warmup_steps"]),
                    fallback_interval=int(exp.params["fallback_interval"]),
                    max_miss=int(exp.params["max_miss"]),
                )
            )
        )
        return MultiTierEdgeCloudPlanner(
            scheduler=scheduler,
            oracle_disagreement_routing=True,
            tier_steer_weight=1.0,
            tier_speed_weight=0.0,
            **common_kwargs,
        )

    if exp.kind == "tiered_probabilistic":
        scheduler = TieredPolicyDrivenScheduler(
            TieredProbabilisticRiskPolicy(
                TieredProbabilisticRiskConfig(
                    p_medium=float(exp.params["p_medium"]),
                    p_large=float(exp.params["p_large"]),
                    min_interval=int(exp.params["min_interval"]),
                    warmup_steps=int(exp.params["warmup_steps"]),
                    fallback_interval=int(exp.params["fallback_interval"]),
                    max_miss=int(exp.params["max_miss"]),
                    seed=int(exp.params["seed"]),
                )
            )
        )
        return MultiTierEdgeCloudPlanner(
            scheduler=scheduler,
            oracle_disagreement_routing=False,
            tier_steer_weight=1.0,
            tier_speed_weight=0.0,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported experiment kind: {exp.kind}")


def run_episode(
    exp: Experiment,
    track: TrackConfig,
    delays: DelayConfig,
    max_laps: int,
    run_idx: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Run one benchmark episode and collect planner and control statistics."""
    waypoints = load_waypoints(track.waypoints_path)
    planner = build_planner(exp, delays)
    env = make_env(track, max_laps=max_laps)
    obs, _ = env.reset(options={"poses": start_pose_for_track(track)})
    metrics = MetricAggregator.create_default(waypoints=waypoints)
    metrics.on_reset(obs, waypoints=waypoints)

    done = False
    step_idx = 0
    medium_calls = 0
    large_calls = 0
    total_cloud_calls = 0
    start_wall = time.perf_counter()

    while not done:
        action = planner.plan(obs, ego_idx=0)
        if getattr(planner, "last_cloud_call", False):
            total_cloud_calls += 1
            tier = getattr(planner, "last_cloud_tier_called", None)
            if tier == CloudTier.MEDIUM:
                medium_calls += 1
            elif tier == CloudTier.LARGE:
                large_calls += 1
            elif isinstance(planner, EdgeCloudPlanner):
                large_calls += 1

        obs, reward, terminated, truncated, _ = env.step(
            np.array([[action.steer, action.speed]])
        )
        metrics.on_step(obs, action, float(reward), ego_idx=0)
        done = bool(terminated or truncated)
        step_idx += 1
        if not done and step_idx >= max_steps:
            done = True

    wall_time_s = time.perf_counter() - start_wall
    env.close()
    report = metrics.report()
    report.update(
        {
            "experiment": exp.name,
            "kind": exp.kind,
            "map_name": track.name,
            "map_path": track.map_path,
            "waypoints_path": track.waypoints_path,
            "medium_latency": float(delays.medium_latency),
            "large_latency": float(delays.large_latency),
            "run_idx": float(run_idx),
            "medium_calls": float(medium_calls),
            "large_calls": float(large_calls),
            "total_cloud_calls": float(total_cloud_calls),
            "large_call_fraction": (
                float(large_calls) / float(total_cloud_calls) if total_cloud_calls else 0.0
            ),
            "step_cap_hit": float(step_idx >= max_steps),
            "wall_time_s": round(float(wall_time_s), 4),
            "rt_factor": round(float(report["lap_time_s"]) / wall_time_s, 4),
            "params_json": json.dumps(exp.params, sort_keys=True),
        }
    )
    return report


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce a compact summary per map/delay/experiment."""
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in results:
        key = (row["map_name"], row["medium_latency"], row["large_latency"])
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (map_name, medium_latency, large_latency), rows in grouped.items():
        baseline = next(
            row for row in rows if row["experiment"] == "large_always_arch6"
        )
        baseline_large_calls = max(1.0, baseline["large_calls"])
        baseline_lap_time = baseline["lap_time_s"]
        baseline_rmse = max(1e-9, baseline["crosstrack_rmse_m"])
        for row in rows:
            summary_rows.append(
                {
                    "map_name": map_name,
                    "medium_latency": medium_latency,
                    "large_latency": large_latency,
                    "experiment": row["experiment"],
                    "collision": row["collision"],
                    "lap_time_s": row["lap_time_s"],
                    "crosstrack_rmse_m": row["crosstrack_rmse_m"],
                    "medium_calls": row["medium_calls"],
                    "large_calls": row["large_calls"],
                    "large_call_reduction_pct_vs_large_always": round(
                        100.0 * (1.0 - (row["large_calls"] / baseline_large_calls)),
                        2,
                    ),
                    "lap_time_delta_pct_vs_large_always": round(
                        100.0 * ((row["lap_time_s"] - baseline_lap_time) / baseline_lap_time),
                        2,
                    ),
                    "crosstrack_rmse_delta_pct_vs_large_always": round(
                        100.0
                        * ((row["crosstrack_rmse_m"] - baseline_rmse) / baseline_rmse),
                        2,
                    ),
                    "rt_factor": row["rt_factor"],
                }
            )
    return summary_rows


def maybe_write_xlsx(
    experiments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
) -> str | None:
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
        experiments_df.to_excel(writer, sheet_name="experiments", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
    return engine


def main() -> None:
    """Run the benchmark sweep and save JSON/CSV/XLSX reports."""
    args = parse_args()
    tracks = [track_config(name.strip()) for name in args.maps.split(",") if name.strip()]
    delays = parse_delay_pairs(args.delay_pairs)
    exps = default_experiments()

    results = [
        run_episode(exp, track, delay, max_laps=args.max_laps, max_steps=args.max_steps)
        for track in tracks
        for delay in delays
        for exp in exps
    ]
    summary_rows = summarize(results)

    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tracks": [track.__dict__ for track in tracks],
        "delay_pairs": [delay.__dict__ for delay in delays],
        "experiments": results,
        "summary": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    experiments_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summary_rows)
    experiments_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    engine = maybe_write_xlsx(experiments_df, summary_df, xlsx_path)

    print(summary_df.to_string(index=False))
    print(f"\nWrote JSON report to {json_path}")
    print(f"Wrote experiment CSV to {csv_path}")
    print(f"Wrote summary CSV to {summary_csv_path}")
    if engine is None:
        print("Skipped XLSX export because neither openpyxl nor xlsxwriter is installed.")
    else:
        print(f"Wrote XLSX workbook to {xlsx_path} using {engine}.")


if __name__ == "__main__":
    main()
