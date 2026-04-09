#!/usr/bin/env python3
"""Focused SRP benchmark on the 10-train/7-held-out split with static fusion."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark_single_tier_paper_strategies import (
    DEFAULT_CLOUD_LATENCY,
    DEFAULT_MAX_STEPS,
    Experiment,
    PlannerSettings,
    _json_default,
    run_episode,
    summarize,
    track_config,
)
from low_ccr_per_method_lambda import SCOUT_TRAIN_MAPS
from map_split import NON_TRAIN_EVAL_MAPS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run a focused static-SRP search on the 10-train/7-held-out split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cloud-latency", type=int, default=DEFAULT_CLOUD_LATENCY)
    parser.add_argument("--coarse-trials", type=int, default=1)
    parser.add_argument("--refine-trials", type=int, default=2)
    parser.add_argument("--eval-trials", type=int, default=5)
    parser.add_argument("--max-laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--tau-grid",
        type=str,
        default="0.35,0.5,0.75,1.0,1.25,1.5,2.0,2.25,3.0",
    )
    parser.add_argument(
        "--nmax-grid",
        type=str,
        default="1,2,3,4,5",
    )
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument(
        "--train-output-stem",
        type=str,
        default="srp_10train7_static_train",
    )
    parser.add_argument(
        "--eval-output-stem",
        type=str,
        default="srp_10train7_static_eval",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="data/benchmarks/srp_10train7_static_report.json",
    )
    return parser.parse_args()


def parse_tau_grid(raw: str) -> list[float]:
    """Parse a comma-separated tau grid."""
    return [float(value) for value in raw.split(",") if value.strip()]


def parse_nmax_grid(raw: str) -> list[int]:
    """Parse a comma-separated nmax grid."""
    return [int(value) for value in raw.split(",") if value.strip()]


def static_settings() -> PlannerSettings:
    """Return the static SRP fusion settings required by the spec."""
    return PlannerSettings(
        alpha_left=0.2,
        alpha_track=0.2,
        alpha_heading=0.7,
        sigma_proc_left=None,
        sigma_proc_track=None,
        sigma_proc_heading=None,
        age_decay_lambda=0.0,
    )


def srp_experiments(tau_grid: list[float], nmax_grid: list[int]) -> list[Experiment]:
    """Construct the SRP search grid."""
    return [
        Experiment(
            f"srp_tau{str(tau).replace('.', 'p')}_n{nmax}",
            "self_normalizing_momentum",
            {"tau": tau, "nmax": nmax, "seed": 7},
        )
        for tau in tau_grid
        for nmax in nmax_grid
    ]


def run_map_experiments(
    map_name: str,
    experiments: list[Experiment],
    cloud_latency: int,
    max_laps: int,
    settings: PlannerSettings,
    trials: int,
    max_steps: int,
    run_offset: int,
) -> list[dict[str, Any]]:
    """Run all requested experiments for one map."""
    track = track_config(map_name)
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        for trial_idx in range(trials):
            rows.append(
                run_episode(
                    exp,
                    track,
                    cloud_latency,
                    max_laps=max_laps,
                    settings=settings,
                    run_idx=run_offset + trial_idx,
                    max_steps=max_steps,
                )
            )
    return rows


def run_batch(
    maps: list[str],
    experiments: list[Experiment],
    cloud_latency: int,
    max_laps: int,
    settings: PlannerSettings,
    trials: int,
    max_steps: int,
    run_offset: int,
    phase_name: str,
    workers: int,
) -> list[dict[str, Any]]:
    """Run a phase across maps, optionally in parallel."""
    print(
        f"[{phase_name}] {len(experiments)} configs x "
        f"{len(maps)} maps x {trials} trials"
    )
    if workers <= 1:
        rows: list[dict[str, Any]] = []
        for map_name in maps:
            print(f"[{phase_name}] map {map_name}")
            rows.extend(
                run_map_experiments(
                    map_name,
                    experiments,
                    cloud_latency,
                    max_laps,
                    settings,
                    trials,
                    max_steps,
                    run_offset,
                )
            )
        return rows

    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_map_experiments,
                map_name,
                experiments,
                cloud_latency,
                max_laps,
                settings,
                trials,
                max_steps,
                run_offset,
            ): map_name
            for map_name in maps
        }
        for future in concurrent.futures.as_completed(futures):
            map_name = futures[future]
            map_rows = future.result()
            print(f"[{phase_name}] completed {map_name}: {len(map_rows)} episodes")
            rows.extend(map_rows)
    return rows


def aggregate(summary_df: pd.DataFrame, experiments_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per experiment across maps."""
    params_lookup = experiments_df.groupby("experiment", as_index=False).agg(
        strategy=("strategy", "first"),
        params_json=("params_json", "first"),
    )
    return (
        summary_df.groupby(["experiment", "strategy"], as_index=False)
        .agg(
            mean_collision_rate=("collision_rate", "mean"),
            mean_step_cap_rate=("step_cap_rate", "mean"),
            mean_crosstrack_rmse=("crosstrack_rmse_m_mean", "mean"),
            mean_crosstrack_max=("crosstrack_max_m_mean", "mean"),
            mean_cloud_call_rate=("cloud_call_rate_mean", "mean"),
            mean_lap_time_s=("lap_time_s_mean", "mean"),
            mean_wall_min_distance=("wall_min_distance_m_mean", "mean"),
            maps_collision_free=("collision_free_rate", "mean"),
        )
        .merge(params_lookup, on=["experiment", "strategy"], how="left")
        .sort_values(
            [
                "mean_collision_rate",
                "mean_step_cap_rate",
                "mean_crosstrack_max",
                "mean_crosstrack_rmse",
                "mean_cloud_call_rate",
                "experiment",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def save_phase(
    stem: str,
    rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write raw, summary, and cross-map CSVs for one phase."""
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments_df, summary_df, _ = summarize(rows, selection_metric="max_cte")
    cross_map_df = aggregate(summary_df, experiments_df)
    experiments_df.to_csv(out_dir / f"{stem}.csv", index=False)
    summary_df.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    cross_map_df.to_csv(out_dir / f"{stem}_cross_map.csv", index=False)
    return experiments_df, summary_df, cross_map_df


def main() -> None:
    """Run the focused SRP train/eval pipeline and write a report."""
    args = parse_args()
    settings = static_settings()
    tau_grid = parse_tau_grid(args.tau_grid)
    nmax_grid = parse_nmax_grid(args.nmax_grid)
    baseline = Experiment("fixed_interval_k2", "fixed_interval", {"interval": 2})
    srp_grid = srp_experiments(tau_grid, nmax_grid)

    coarse_rows = run_batch(
        maps=list(SCOUT_TRAIN_MAPS),
        experiments=[baseline] + srp_grid,
        cloud_latency=args.cloud_latency,
        max_laps=args.max_laps,
        settings=settings,
        trials=args.coarse_trials,
        max_steps=args.max_steps,
        run_offset=0,
        phase_name="train-coarse",
        workers=args.workers,
    )
    _, _, coarse_cross_map = save_phase(f"{args.train_output_stem}_coarse", coarse_rows)

    srp_ranked = coarse_cross_map[
        coarse_cross_map["strategy"] == "self_normalizing_momentum"
    ].copy()
    top_names = list(srp_ranked["experiment"].head(max(1, int(args.finalists))))
    finalists = [exp for exp in srp_grid if exp.name in top_names]
    print("[train-refine] top SRP configs:", ", ".join(top_names))

    refine_rows = run_batch(
        maps=list(SCOUT_TRAIN_MAPS),
        experiments=[baseline] + finalists,
        cloud_latency=args.cloud_latency,
        max_laps=args.max_laps,
        settings=settings,
        trials=args.refine_trials,
        max_steps=args.max_steps,
        run_offset=100,
        phase_name="train-refine",
        workers=args.workers,
    )
    train_rows = coarse_rows + refine_rows
    _, _, train_cross_map = save_phase(args.train_output_stem, train_rows)

    srp_train = train_cross_map[
        (train_cross_map["strategy"] == "self_normalizing_momentum")
        & (train_cross_map["experiment"].isin(top_names))
    ].copy()
    winner = srp_train.iloc[0]
    winner_name = str(winner["experiment"])
    winner_params = json.loads(winner["params_json"]) if winner["params_json"] else {}
    winner_exp = next(exp for exp in srp_grid if exp.name == winner_name)
    print("[train] selected winner:", winner_name, winner_params)

    eval_rows = run_batch(
        maps=list(NON_TRAIN_EVAL_MAPS),
        experiments=[baseline, winner_exp],
        cloud_latency=args.cloud_latency,
        max_laps=args.max_laps,
        settings=settings,
        trials=args.eval_trials,
        max_steps=args.max_steps,
        run_offset=0,
        phase_name="eval",
        workers=args.workers,
    )
    _, _, eval_cross_map = save_phase(args.eval_output_stem, eval_rows)

    eval_winner = eval_cross_map[eval_cross_map["experiment"] == winner_name].iloc[0]
    eval_baseline = eval_cross_map[
        eval_cross_map["experiment"] == baseline.name
    ].iloc[0]
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_maps": list(SCOUT_TRAIN_MAPS),
        "eval_maps": list(NON_TRAIN_EVAL_MAPS),
        "cloud_latency": args.cloud_latency,
        "settings": settings.__dict__,
        "candidate_taus": tau_grid,
        "candidate_nmax": nmax_grid,
        "selected_srp": {
            "experiment": winner_name,
            "params": winner_params,
            "train_metrics": winner.to_dict(),
            "eval_metrics": eval_winner.to_dict(),
        },
        "baseline_fixed_interval_k2": {
            "experiment": baseline.name,
            "params": baseline.params,
            "eval_metrics": eval_baseline.to_dict(),
        },
    }
    report_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print("[done] report written to", report_path)
    print(json.dumps(payload["selected_srp"], indent=2, default=_json_default))
    print(json.dumps(payload["baseline_fixed_interval_k2"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
