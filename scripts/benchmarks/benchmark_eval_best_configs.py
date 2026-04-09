#!/usr/bin/env python3
"""Eval benchmark: run optimized configs on the held-out eval maps.

Reads: ``best_configs.json`` from the optimization step and F1 map assets.
Writes: eval benchmark artifacts under ``data/benchmarks`` in the same format
as the canonical single-tier benchmark, compatible with the curated plot script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import pandas as pd

from benchmark_single_tier_paper_strategies import (
    DEFAULT_CLOUD_LATENCY,
    DEFAULT_MAX_STEPS,
    DEFAULT_TRIALS,
    Experiment,
    PlannerSettings,
    _json_default,
    maybe_write_xlsx,
    run_episode,
    summarize,
    track_config,
)
from map_split import EVAL_MAPS


def load_best_configs(path: Path) -> list[Experiment]:
    """Load best configs JSON and reconstruct Experiment objects."""
    data = json.loads(path.read_text())
    configs = data["best_configs"]
    experiments: list[Experiment] = []
    for strategy, info in configs.items():
        experiments.append(
            Experiment(
                name=info["experiment_name"],
                strategy=info["strategy"],
                params=info["params"],
            )
        )
    return experiments


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate optimized configurations on the held-out eval maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs-json",
        type=str,
        default="data/benchmarks/best_configs.json",
        help="Path to the best_configs.json from the optimization step.",
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=",".join(EVAL_MAPS),
        help="Comma-separated eval map names under data/maps/F1.",
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
        default="eval_best_configs_10maps",
        help="Base filename stem for outputs under data/benchmarks.",
    )
    parser.add_argument(
        "--alpha-left",
        type=float,
        default=0.2,
        help="Cloud blend weight for the left-distance feature.",
    )
    parser.add_argument(
        "--alpha-track",
        type=float,
        default=0.2,
        help="Cloud blend weight for the track-width feature.",
    )
    parser.add_argument(
        "--alpha-heading",
        type=float,
        default=0.7,
        help="Cloud blend weight for the heading-error feature.",
    )
    parser.add_argument(
        "--sigma-proc-left",
        type=float,
        default=None,
        help="Optional left-feature process-noise sigma.",
    )
    parser.add_argument(
        "--sigma-proc-track",
        type=float,
        default=None,
        help="Optional track-feature process-noise sigma.",
    )
    parser.add_argument(
        "--sigma-proc-heading",
        type=float,
        default=None,
        help="Optional heading-feature process-noise sigma.",
    )
    parser.add_argument(
        "--age-decay-lambda",
        type=float,
        default=0.0,
        help="Global anchored age-decay scale for stale cloud features.",
    )
    parser.add_argument(
        "--deviation-steer-weight",
        type=float,
        default=1.0,
        help="Weight on steering disagreement.",
    )
    parser.add_argument(
        "--deviation-speed-weight",
        type=float,
        default=1.0,
        help="Weight on speed disagreement.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (parallelizes across maps only).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("rmse", "max_cte"),
        default="rmse",
        help="Primary ranking metric used when ordering held-out configurations within each map.",
    )
    return parser.parse_args()


def _run_map_experiments(
    map_name: str,
    experiments: list[Experiment],
    cloud_latency: int,
    max_laps: int,
    settings: PlannerSettings,
    trials: int,
    max_steps: int,
) -> list[dict[str, object]]:
    """Run all eval experiments for one map."""
    track = track_config(map_name)
    results: list[dict[str, object]] = []
    for exp in experiments:
        for run_idx in range(trials):
            results.append(
                run_episode(
                    exp,
                    track,
                    cloud_latency,
                    max_laps=max_laps,
                    settings=settings,
                    run_idx=run_idx,
                    max_steps=max_steps,
                )
            )
    return results


def main() -> None:
    """Run the eval benchmark and save reports."""
    args = parse_args()
    tracks = [
        track_config(name.strip())
        for name in args.maps.split(",")
        if name.strip()
    ]
    settings = PlannerSettings(
        alpha_left=args.alpha_left,
        alpha_track=args.alpha_track,
        alpha_heading=args.alpha_heading,
        sigma_proc_left=args.sigma_proc_left,
        sigma_proc_track=args.sigma_proc_track,
        sigma_proc_heading=args.sigma_proc_heading,
        age_decay_lambda=args.age_decay_lambda,
        deviation_steer_weight=args.deviation_steer_weight,
        deviation_speed_weight=args.deviation_speed_weight,
    )

    configs_path = Path(args.configs_json)
    experiments = load_best_configs(configs_path)
    print(f"Loaded {len(experiments)} configs from {configs_path}")
    for exp in experiments:
        print(f"  {exp.name} ({exp.strategy}): {exp.params}")

    total = len(experiments) * len(tracks) * args.trials
    print(
        f"\nRunning {len(experiments)} configs x {len(tracks)} maps x "
        f"{args.trials} trials = {total} episodes"
    )

    start_time = time.perf_counter()
    if args.workers > 1:
        print(f"Using {args.workers} parallel workers")
        results: list[dict[str, object]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    _run_map_experiments,
                    track.name,
                    experiments,
                    args.cloud_latency,
                    args.max_laps,
                    settings,
                    args.trials,
                    args.max_steps,
                ): track.name
                for track in tracks
            }
            for future in concurrent.futures.as_completed(futures):
                map_name = futures[future]
                map_results = future.result()
                print(f"  Completed {map_name}: {len(map_results)} results")
                results.extend(map_results)
    else:
        results = []
        for index, track in enumerate(tracks, start=1):
            print(f"  Map {index}/{len(tracks)}: {track.name}")
            results.extend(
                _run_map_experiments(
                    track.name,
                    experiments,
                    args.cloud_latency,
                    args.max_laps,
                    settings,
                    args.trials,
                    args.max_steps,
                )
            )
    elapsed = time.perf_counter() - start_time
    print(f"\nCompleted {len(results)} episodes in {elapsed:.1f}s")

    experiments_df, summary_df, near_target_df = summarize(
        results,
        selection_metric=args.selection_metric,
    )
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
        "configs_source": str(configs_path),
        "tracks": [t.__dict__ for t in tracks],
        "cloud_latency": args.cloud_latency,
        "cloud_latencies": [args.cloud_latency],
        "settings": settings.__dict__,
        "selection_metric": args.selection_metric,
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
        print("Skipped XLSX (no openpyxl or xlsxwriter).")
    else:
        print(f"Wrote XLSX to {xlsx_path} using {engine}.")


if __name__ == "__main__":
    main()
