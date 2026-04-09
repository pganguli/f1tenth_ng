#!/usr/bin/env python3
"""Hyperparameter optimization: expanded grid search on training maps.

Reads: F1 map assets for the training-set maps and edge/cloud model checkpoints.
Writes: best-config JSON and training-sweep artifacts under ``data/benchmarks``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from benchmark_single_tier_paper_strategies import (
    DEFAULT_CLOUD_LATENCY,
    DEFAULT_MAX_STEPS,
    DEFAULT_TRIALS,
    TARGET_CCR_HIGH,
    TARGET_CCR_LOW,
    Experiment,
    PlannerSettings,
    _json_default,
    maybe_write_xlsx,
    run_episode,
    summarize,
    track_config,
)
from map_split import TRAIN_MAPS


STRATEGY_FAMILIES = [
    "always",
    "fixed_interval",
    "fixed_bernoulli",
    "bernoulli_max_miss",
    "logistic",
    "exponential",
    "piecewise_ramp",
    "deterministic",
]


def optimization_grid() -> list[Experiment]:
    """Return the expanded hyperparameter grid for training-set optimization."""
    experiments: list[Experiment] = [Experiment("always_hit", "always", {})]

    for interval in (2, 3, 4, 5, 7, 10):
        experiments.append(
            Experiment(
                f"fixed_interval_k{interval}",
                "fixed_interval",
                {"interval": interval},
            )
        )

    for p in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        suffix = int(round(p * 100))
        experiments.append(
            Experiment(
                f"fixed_bernoulli_p{suffix}",
                "fixed_bernoulli",
                {"p": p, "seed": 7},
            )
        )

    for p in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for max_miss in (2, 3, 4, 5, 7):
            suffix = int(round(p * 100))
            experiments.append(
                Experiment(
                    f"bernoulli_max_miss_p{suffix}_m{max_miss}",
                    "bernoulli_max_miss",
                    {"p": p, "max_miss": max_miss, "seed": 7},
                )
            )

    for center in (0.01, 0.02, 0.03, 0.04, 0.05):
        for slope in (20.0, 30.0, 40.0, 50.0, 70.0):
            experiments.append(
                Experiment(
                    f"logistic_c{str(center).replace('.', 'p')}_s{int(slope)}",
                    "logistic",
                    {"center": center, "slope": slope, "seed": 7},
                )
            )

    for center in (0.01, 0.02, 0.03, 0.04, 0.05):
        for rate in (10.0, 15.0, 20.0, 30.0, 50.0):
            experiments.append(
                Experiment(
                    f"exponential_c{str(center).replace('.', 'p')}_r{int(rate)}",
                    "exponential",
                    {"center": center, "rate": rate, "seed": 7},
                )
            )

    for d_low in (0.005, 0.01, 0.02, 0.03):
        for d_high in (0.03, 0.04, 0.05, 0.06, 0.08):
            if d_low >= d_high:
                continue
            experiments.append(
                Experiment(
                    f"piecewise_ramp_{str(d_low).replace('.', 'p')}_{str(d_high).replace('.', 'p')}",
                    "piecewise_ramp",
                    {"d_low": d_low, "d_high": d_high, "seed": 7},
                )
            )

    for threshold in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07):
        experiments.append(
            Experiment(
                f"deterministic_t{str(threshold).replace('.', 'p')}",
                "deterministic",
                {"threshold": threshold},
            )
        )

    return experiments


def sanity_optimization_grid() -> list[Experiment]:
    """Return the coarse, edge-shifted grid used by the sanity study."""
    return [
        Experiment("always_hit", "always", {}),
        Experiment("fixed_interval_k2", "fixed_interval", {"interval": 2}),
        Experiment("fixed_interval_k4", "fixed_interval", {"interval": 4}),
        Experiment("fixed_interval_k7", "fixed_interval", {"interval": 7}),
        Experiment("fixed_bernoulli_p55", "fixed_bernoulli", {"p": 0.55, "seed": 7}),
        Experiment("fixed_bernoulli_p65", "fixed_bernoulli", {"p": 0.65, "seed": 7}),
        Experiment("fixed_bernoulli_p75", "fixed_bernoulli", {"p": 0.75, "seed": 7}),
        Experiment("bernoulli_max_miss_p55_m3", "bernoulli_max_miss", {"p": 0.55, "max_miss": 3, "seed": 7}),
        Experiment("bernoulli_max_miss_p55_m6", "bernoulli_max_miss", {"p": 0.55, "max_miss": 6, "seed": 7}),
        Experiment("bernoulli_max_miss_p65_m3", "bernoulli_max_miss", {"p": 0.65, "max_miss": 3, "seed": 7}),
        Experiment("bernoulli_max_miss_p65_m6", "bernoulli_max_miss", {"p": 0.65, "max_miss": 6, "seed": 7}),
        Experiment("bernoulli_max_miss_p75_m3", "bernoulli_max_miss", {"p": 0.75, "max_miss": 3, "seed": 7}),
        Experiment("bernoulli_max_miss_p75_m6", "bernoulli_max_miss", {"p": 0.75, "max_miss": 6, "seed": 7}),
        Experiment("logistic_c0p04_s10", "logistic", {"center": 0.04, "slope": 10.0, "seed": 7}),
        Experiment("logistic_c0p04_s30", "logistic", {"center": 0.04, "slope": 30.0, "seed": 7}),
        Experiment("logistic_c0p06_s10", "logistic", {"center": 0.06, "slope": 10.0, "seed": 7}),
        Experiment("logistic_c0p06_s30", "logistic", {"center": 0.06, "slope": 30.0, "seed": 7}),
        Experiment("logistic_c0p08_s10", "logistic", {"center": 0.08, "slope": 10.0, "seed": 7}),
        Experiment("logistic_c0p08_s30", "logistic", {"center": 0.08, "slope": 30.0, "seed": 7}),
        Experiment("exponential_c0p04_r5", "exponential", {"center": 0.04, "rate": 5.0, "seed": 7}),
        Experiment("exponential_c0p04_r15", "exponential", {"center": 0.04, "rate": 15.0, "seed": 7}),
        Experiment("exponential_c0p06_r5", "exponential", {"center": 0.06, "rate": 5.0, "seed": 7}),
        Experiment("exponential_c0p06_r15", "exponential", {"center": 0.06, "rate": 15.0, "seed": 7}),
        Experiment("exponential_c0p08_r5", "exponential", {"center": 0.08, "rate": 5.0, "seed": 7}),
        Experiment("exponential_c0p08_r15", "exponential", {"center": 0.08, "rate": 15.0, "seed": 7}),
        Experiment("piecewise_ramp_0p03_0p08", "piecewise_ramp", {"d_low": 0.03, "d_high": 0.08, "seed": 7}),
        Experiment("piecewise_ramp_0p04_0p10", "piecewise_ramp", {"d_low": 0.04, "d_high": 0.10, "seed": 7}),
        Experiment("piecewise_ramp_0p05_0p12", "piecewise_ramp", {"d_low": 0.05, "d_high": 0.12, "seed": 7}),
        Experiment("deterministic_t0p05", "deterministic", {"threshold": 0.05}),
        Experiment("deterministic_t0p07", "deterministic", {"threshold": 0.07}),
        Experiment("deterministic_t0p09", "deterministic", {"threshold": 0.09}),
        Experiment("deterministic_t0p11", "deterministic", {"threshold": 0.11}),
    ]


def aggregate_across_maps(
    summary_df: pd.DataFrame,
    experiments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-experiment metrics across all training maps."""
    # Get params_json from the raw experiments (one per experiment name)
    params_lookup = (
        experiments_df.groupby("experiment", as_index=False)["params_json"]
        .first()
    )

    cross_map = (
        summary_df.groupby(["experiment", "strategy"], as_index=False)
        .agg(
            mean_collision_rate=("collision_rate", "mean"),
            mean_step_cap_rate=("step_cap_rate", "mean"),
            mean_crosstrack_rmse=("crosstrack_rmse_m_mean", "mean"),
            mean_crosstrack_max=("crosstrack_max_m_mean", "mean"),
            mean_cloud_call_rate=("cloud_call_rate_mean", "mean"),
            std_cloud_call_rate=("cloud_call_rate_mean", "std"),
            maps_in_band=("in_target_ccr_band", "mean"),
            maps_collision_free=(
                "collision_rate",
                lambda x: float((x == 0).mean()),
            ),
        )
        .fillna(0.0)
    )
    cross_map = cross_map.merge(params_lookup, on="experiment", how="left")
    cross_map["in_target_ccr_band"] = cross_map["mean_cloud_call_rate"].between(
        TARGET_CCR_LOW, TARGET_CCR_HIGH
    )
    return cross_map


def select_best_per_strategy(
    cross_map_df: pd.DataFrame,
    selection_metric: str,
) -> dict[str, dict[str, Any]]:
    """Select the best configuration per strategy family from cross-map aggregates."""
    best: dict[str, dict[str, Any]] = {}
    metric_column = {
        "rmse": "mean_crosstrack_rmse",
        "max_cte": "mean_crosstrack_max",
    }[selection_metric]

    for strategy in STRATEGY_FAMILIES:
        family = cross_map_df[cross_map_df["strategy"] == strategy].copy()
        if family.empty:
            continue

        # Prefer configs in the target CCR band
        in_band = family[family["in_target_ccr_band"]].copy()
        if not in_band.empty:
            in_band["ccr_dist"] = (in_band["mean_cloud_call_rate"] - 0.60).abs()
            ranked = in_band.sort_values(
                [
                    "mean_collision_rate",
                    "mean_step_cap_rate",
                    metric_column,
                    "ccr_dist",
                ],
                kind="stable",
            )
        else:
            # Fallback: pick nearest to band midpoint among collision-free
            band_mid = (TARGET_CCR_LOW + TARGET_CCR_HIGH) / 2.0
            family["ccr_dist"] = (family["mean_cloud_call_rate"] - band_mid).abs()
            collision_free = family[family["mean_collision_rate"] == 0]
            candidates = collision_free if not collision_free.empty else family
            ranked = candidates.sort_values(
                ["ccr_dist", "mean_collision_rate", "mean_step_cap_rate", metric_column],
                kind="stable",
            )
            print(
                f"  WARNING: no in-band config for '{strategy}', "
                f"using nearest-to-band fallback."
            )

        winner = ranked.iloc[0]
        params = json.loads(winner["params_json"]) if winner["params_json"] else {}

        best[strategy] = {
            "experiment_name": winner["experiment"],
            "strategy": strategy,
            "params": params,
            "train_metrics": {
                "mean_collision_rate": float(winner["mean_collision_rate"]),
                "mean_step_cap_rate": float(winner["mean_step_cap_rate"]),
                "mean_crosstrack_rmse": float(winner["mean_crosstrack_rmse"]),
                "mean_crosstrack_max": float(winner["mean_crosstrack_max"]),
                "mean_cloud_call_rate": float(winner["mean_cloud_call_rate"]),
                "maps_in_band": float(winner.get("maps_in_band", 0.0)),
                "maps_collision_free": float(winner.get("maps_collision_free", 0.0)),
            },
        }

    return best


def _run_map_experiments(
    map_name: str,
    experiments: list[Experiment],
    cloud_latency: int,
    max_laps: int,
    settings: PlannerSettings,
    trials: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Run all experiments on a single map (used for parallelism)."""
    track = track_config(map_name)
    results: list[dict[str, Any]] = []
    for exp in experiments:
        for run_idx in range(trials):
            try:
                result = run_episode(
                    exp,
                    track,
                    cloud_latency,
                    max_laps=max_laps,
                    settings=settings,
                    run_idx=run_idx,
                    max_steps=max_steps,
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  ERROR: {exp.name} on {map_name} trial {run_idx}: {exc}"
                )
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization via expanded grid search on training maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=",".join(TRAIN_MAPS),
        help="Comma-separated training map names under data/maps/F1.",
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
        default="hyperparam_optimization_train_13maps",
        help="Base filename stem for outputs under data/benchmarks.",
    )
    parser.add_argument(
        "--grid-profile",
        choices=("full", "sanity"),
        default="full",
        help="Which train-time hyperparameter grid to evaluate.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("rmse", "max_cte"),
        default="rmse",
        help="Primary metric used to rank configurations within each strategy family.",
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
        help="Number of parallel workers (1 = sequential). Parallelizes across maps.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategy families to include (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the optimization grid search and export best configs."""
    args = parse_args()
    maps = [name.strip() for name in args.maps.split(",") if name.strip()]
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

    experiments = (
        sanity_optimization_grid()
        if args.grid_profile == "sanity"
        else optimization_grid()
    )
    if args.strategies:
        allowed = {s.strip() for s in args.strategies.split(",")}
        experiments = [e for e in experiments if e.strategy in allowed]

    total = len(experiments) * len(maps) * args.trials
    print(
        f"Running {len(experiments)} configs x {len(maps)} maps x {args.trials} trials "
        f"= {total} episodes"
    )
    if args.workers > 1:
        print(f"Using {args.workers} parallel workers")

    start_time = time.perf_counter()

    if args.workers > 1:
        all_results: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    _run_map_experiments,
                    map_name,
                    experiments,
                    args.cloud_latency,
                    args.max_laps,
                    settings,
                    args.trials,
                    args.max_steps,
                ): map_name
                for map_name in maps
            }
            for future in concurrent.futures.as_completed(futures):
                map_name = futures[future]
                map_results = future.result()
                print(f"  Completed {map_name}: {len(map_results)} results")
                all_results.extend(map_results)
    else:
        all_results = []
        for i, map_name in enumerate(maps, 1):
            print(f"  Map {i}/{len(maps)}: {map_name}")
            map_results = _run_map_experiments(
                map_name,
                experiments,
                args.cloud_latency,
                args.max_laps,
                settings,
                args.trials,
                args.max_steps,
            )
            all_results.extend(map_results)

    elapsed = time.perf_counter() - start_time
    print(f"\nCompleted {len(all_results)} episodes in {elapsed:.1f}s")

    # Summarize per-map (reuse existing summarize function)
    experiments_df, summary_df, _ = summarize(
        all_results,
        selection_metric=args.selection_metric,
    )

    # Aggregate across maps
    cross_map_df = aggregate_across_maps(summary_df, experiments_df)

    # Select best config per strategy
    print("\nSelecting best configuration per strategy family:")
    best_configs = select_best_per_strategy(cross_map_df, args.selection_metric)
    for strategy, info in best_configs.items():
        metrics = info["train_metrics"]
        print(
            f"  {strategy}: {info['experiment_name']} "
            f"(CCR={metrics['mean_cloud_call_rate']:.3f}, "
            f"maxCTE={metrics['mean_crosstrack_max']:.4f}, "
            f"collision={metrics['mean_collision_rate']:.3f})"
        )

    # Write outputs
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem

    # Best configs JSON (consumed by eval script)
    best_configs_path = out_dir / "best_configs.json"
    stem_best_configs_path = out_dir / f"{stem}_best_configs.json"
    best_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_maps": maps,
        "cloud_latency": args.cloud_latency,
        "settings": settings.__dict__,
        "selection_metric": args.selection_metric,
        "grid_profile": args.grid_profile,
        "target_ccr_band": [TARGET_CCR_LOW, TARGET_CCR_HIGH],
        "best_configs": best_configs,
    }
    best_configs_path.write_text(
        json.dumps(best_payload, indent=2, default=_json_default)
    )
    stem_best_configs_path.write_text(
        json.dumps(best_payload, indent=2, default=_json_default)
    )
    print(f"\nWrote best configs to {best_configs_path}")
    print(f"Wrote stem-specific best configs to {stem_best_configs_path}")

    # Full report JSON
    json_path = out_dir / f"{stem}.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_maps": maps,
        "cloud_latency": args.cloud_latency,
        "settings": settings.__dict__,
        "selection_metric": args.selection_metric,
        "grid_profile": args.grid_profile,
        "target_ccr_band": [TARGET_CCR_LOW, TARGET_CCR_HIGH],
        "experiments": all_results,
        "summary": summary_df.to_dict(orient="records"),
        "cross_map_summary": cross_map_df.to_dict(orient="records"),
        "best_configs": best_configs,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    # CSVs
    csv_path = out_dir / f"{stem}.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    cross_map_csv_path = out_dir / f"{stem}_cross_map.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"

    experiments_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    cross_map_df.to_csv(cross_map_csv_path, index=False)
    engine = maybe_write_xlsx(
        {
            "experiments": experiments_df,
            "summary": summary_df,
            "cross_map": cross_map_df,
        },
        xlsx_path,
    )

    print(f"Wrote JSON report to {json_path}")
    print(f"Wrote experiment CSV to {csv_path}")
    print(f"Wrote summary CSV to {summary_csv_path}")
    print(f"Wrote cross-map CSV to {cross_map_csv_path}")
    if engine is None:
        print("Skipped XLSX (no openpyxl or xlsxwriter).")
    else:
        print(f"Wrote XLSX to {xlsx_path} using {engine}.")


if __name__ == "__main__":
    main()
