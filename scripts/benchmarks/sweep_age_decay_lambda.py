#!/usr/bin/env python3
"""Tune anchored age-decay lambda on training maps with representative strategies."""

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
    Experiment,
    PlannerSettings,
    TARGET_CCR_HIGH,
    TARGET_CCR_LOW,
    _json_default,
    maybe_write_xlsx,
    run_episode,
    summarize,
    track_config,
)
from map_split import TRAIN_MAPS


DEFAULT_LAMBDA_GRID = (0.0, 1.0, 4.0, 16.0, 64.0)


def build_refinement_grid(
    coarse_grid: list[float],
    winner: float,
    refine_points: int = 5,
) -> tuple[list[float], str]:
    """Build a fine grid around *winner* by subdividing between its coarse neighbors.

    Returns ``(fine_grid, boundary_note)`` where *boundary_note* is one of
    ``"none"``, ``"lower"``, or ``"upper"``.
    """
    grid = sorted(coarse_grid)
    if len(grid) < 2:
        return [], "none"

    # Find winner in the sorted grid (tolerance for float comparison).
    idx: int | None = None
    for i, val in enumerate(grid):
        if abs(val - winner) < 1e-9:
            idx = i
            break
    if idx is None:
        return [], "none"

    left = grid[idx - 1] if idx > 0 else None
    right = grid[idx + 1] if idx < len(grid) - 1 else None

    if left is not None and right is not None:
        # Interior: subdivide (left, right).
        low, high = left, right
        boundary_note = "none"
    elif left is None:
        # Lower boundary: subdivide (0, right).
        low, high = 0.0, right  # type: ignore[assignment]
        boundary_note = "lower"
    else:
        # Upper boundary: extend by last interval width.
        extension = grid[-1] - grid[-2]
        low, high = left, winner + extension
        boundary_note = "upper"

    points = np.linspace(low, high, refine_points + 2)[1:-1]
    fine: list[float] = []
    for p in points:
        p_rounded = round(float(p), 6)
        if all(abs(p_rounded - c) > 1e-9 for c in grid):
            fine.append(p_rounded)
    return fine, boundary_note


def representative_experiments() -> list[Experiment]:
    """Return the representative strategies used for lambda tuning."""
    return [
        Experiment("always_hit", "always", {}),
        Experiment("fixed_interval_k5", "fixed_interval", {"interval": 5}),
        Experiment("fixed_bernoulli_p60", "fixed_bernoulli", {"p": 0.60, "seed": 7}),
        Experiment(
            "bernoulli_max_miss_p60_m5",
            "bernoulli_max_miss",
            {"p": 0.60, "max_miss": 5, "seed": 7},
        ),
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Sweep the anchored age-decay lambda on the training-map split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=",".join(TRAIN_MAPS),
        help="Comma-separated training map names under data/maps/F1.",
    )
    parser.add_argument(
        "--lambda-grid",
        type=str,
        default=",".join(str(value) for value in DEFAULT_LAMBDA_GRID),
        help="Comma-separated lambda values to evaluate.",
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
        default=3,
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
        default="lambda_sweep_train_13maps",
        help="Base filename stem for outputs under data/benchmarks.",
    )
    parser.add_argument(
        "--selected-lambda-json",
        type=str,
        default="data/benchmarks/lambda_sweep_optimal.json",
        help="Path for the JSON artifact containing the selected lambda.",
    )
    parser.add_argument("--alpha-left", type=float, default=0.2)
    parser.add_argument("--alpha-track", type=float, default=0.2)
    parser.add_argument("--alpha-heading", type=float, default=0.7)
    parser.add_argument("--sigma-proc-left", type=float, default=0.044961)
    parser.add_argument("--sigma-proc-track", type=float, default=0.067937)
    parser.add_argument("--sigma-proc-heading", type=float, default=0.033182)
    parser.add_argument("--deviation-steer-weight", type=float, default=1.0)
    parser.add_argument("--deviation-speed-weight", type=float, default=1.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (parallelizes across maps only).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("rmse", "max_cte"),
        default="max_cte",
        help="Primary metric used to rank lambda candidates.",
    )
    parser.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a refinement pass around the coarse-grid winner.",
    )
    parser.add_argument(
        "--refine-points",
        type=int,
        default=5,
        help="Number of intermediate lambda values in the refinement grid.",
    )
    return parser.parse_args()


def _run_map_experiments(
    map_name: str,
    lambda_value: float,
    experiments: list[Experiment],
    cloud_latency: int,
    max_laps: int,
    settings: PlannerSettings,
    trials: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Run the representative lambda-sweep experiments for one map."""
    track = track_config(map_name)
    results: list[dict[str, Any]] = []
    for exp in experiments:
        for run_idx in range(trials):
            row = run_episode(
                exp,
                track,
                cloud_latency,
                max_laps=max_laps,
                settings=settings,
                run_idx=run_idx,
                max_steps=max_steps,
            )
            row["age_decay_lambda"] = float(lambda_value)
            results.append(row)
    return results


def aggregate_lambda_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark summaries to one row per lambda."""
    grouped = (
        summary_df.groupby("age_decay_lambda", as_index=False)
        .agg(
            representative_rows=("experiment", "count"),
            mean_collision_rate=("collision_rate", "mean"),
            mean_collision_free_rate=("collision_free_rate", "mean"),
            mean_step_cap_rate=("step_cap_rate", "mean"),
            mean_crosstrack_rmse=("crosstrack_rmse_m_mean", "mean"),
            std_crosstrack_rmse=("crosstrack_rmse_m_mean", "std"),
            mean_crosstrack_max=("crosstrack_max_m_mean", "mean"),
            std_crosstrack_max=("crosstrack_max_m_mean", "std"),
            mean_cloud_call_rate=("cloud_call_rate_mean", "mean"),
            std_cloud_call_rate=("cloud_call_rate_mean", "std"),
            frac_in_target_band=("in_target_ccr_band", "mean"),
        )
        .fillna(0.0)
        .sort_values("age_decay_lambda", kind="stable")
        .reset_index(drop=True)
    )
    grouped["mean_abs_ccr_distance"] = (
        grouped["mean_cloud_call_rate"] - 0.60
    ).abs()
    if (grouped["mean_collision_rate"] == 0.0).any():
        grouped["collision_feasible"] = grouped["mean_collision_rate"] == 0.0
    else:
        grouped["collision_feasible"] = True
    return grouped


def select_best_lambda(summary_df: pd.DataFrame, selection_metric: str) -> pd.Series:
    """Select the best lambda from the aggregated sweep summary."""
    candidates = summary_df[summary_df["collision_feasible"]].copy()
    metric_column = {
        "rmse": "mean_crosstrack_rmse",
        "max_cte": "mean_crosstrack_max",
    }[selection_metric]
    ranked = candidates.sort_values(
        [
            metric_column,
            "mean_abs_ccr_distance",
            "mean_step_cap_rate",
            "age_decay_lambda",
        ],
        kind="stable",
    )
    return ranked.iloc[0]


def run_lambda_grid(
    lambda_grid: list[float],
    experiments: list[Experiment],
    maps: list[str],
    args: argparse.Namespace,
    stage_label: str = "",
) -> list[dict[str, Any]]:
    """Run representative experiments for every lambda value in *lambda_grid*."""
    prefix = f"[{stage_label}] " if stage_label else ""
    results: list[dict[str, Any]] = []
    for lambda_value in lambda_grid:
        settings = PlannerSettings(
            alpha_left=args.alpha_left,
            alpha_track=args.alpha_track,
            alpha_heading=args.alpha_heading,
            sigma_proc_left=args.sigma_proc_left,
            sigma_proc_track=args.sigma_proc_track,
            sigma_proc_heading=args.sigma_proc_heading,
            age_decay_lambda=lambda_value,
            deviation_steer_weight=args.deviation_steer_weight,
            deviation_speed_weight=args.deviation_speed_weight,
        )
        print(f"\n{prefix}Lambda {lambda_value:g}")
        if args.workers > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            ) as executor:
                futures = {
                    executor.submit(
                        _run_map_experiments,
                        map_name,
                        lambda_value,
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
                    results.extend(map_results)
        else:
            for index, map_name in enumerate(maps, start=1):
                print(f"  Map {index}/{len(maps)}: {map_name}")
                results.extend(
                    _run_map_experiments(
                        map_name,
                        lambda_value,
                        experiments,
                        args.cloud_latency,
                        args.max_laps,
                        settings,
                        args.trials,
                        args.max_steps,
                    )
                )
    return results


def main() -> None:
    """Run the representative lambda sweep and export the chosen lambda."""
    args = parse_args()
    maps = [name.strip() for name in args.maps.split(",") if name.strip()]
    lambda_grid = [float(value) for value in args.lambda_grid.split(",") if value.strip()]
    experiments = representative_experiments()
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(lambda_grid) * len(experiments) * len(maps) * args.trials
    print(
        f"Running {len(lambda_grid)} lambda values x {len(experiments)} configs x "
        f"{len(maps)} maps x {args.trials} trials = {total} episodes"
    )
    if args.workers > 1:
        print(f"Using {args.workers} parallel workers")

    # --- Stage 1: coarse grid ---
    start_time = time.perf_counter()
    all_results = run_lambda_grid(lambda_grid, experiments, maps, args, "Coarse")

    # --- Stage 2: optional refinement ---
    fine_grid: list[float] = []
    boundary_note = "none"
    coarse_winner: float | None = None

    if args.refine:
        coarse_exp_df, coarse_sum_df, _ = summarize(
            all_results, selection_metric=args.selection_metric,
        )
        coarse_lambda_summary = aggregate_lambda_summary(coarse_sum_df)
        coarse_best = select_best_lambda(coarse_lambda_summary, args.selection_metric)
        coarse_winner = float(coarse_best["age_decay_lambda"])
        print(f"\nCoarse winner: lambda = {coarse_winner:g}")

        fine_grid, boundary_note = build_refinement_grid(
            lambda_grid, coarse_winner, args.refine_points,
        )
        if fine_grid:
            fine_total = len(fine_grid) * len(experiments) * len(maps) * args.trials
            print(
                f"Refining with {len(fine_grid)} points: {fine_grid} "
                f"({fine_total} episodes)"
            )
            fine_results = run_lambda_grid(
                fine_grid, experiments, maps, args, "Refine",
            )
            all_results.extend(fine_results)
        else:
            print("No refinement points generated (single-element grid).")

    elapsed = time.perf_counter() - start_time
    print(f"\nCompleted {len(all_results)} episodes in {elapsed:.1f}s")

    # --- Summarize combined results ---
    full_lambda_grid = sorted(set(lambda_grid + fine_grid))
    experiments_df, summary_df, _ = summarize(
        all_results,
        selection_metric=args.selection_metric,
    )
    lambda_summary = aggregate_lambda_summary(summary_df)
    best = select_best_lambda(lambda_summary, args.selection_metric)

    raw_csv_path = out_dir / f"{args.output_stem}.csv"
    summary_csv_path = out_dir / f"{args.output_stem}_summary.csv"
    json_path = Path(args.selected_lambda_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"{args.output_stem}.xlsx"

    experiments_df.to_csv(raw_csv_path, index=False)
    lambda_summary.to_csv(summary_csv_path, index=False)
    engine = maybe_write_xlsx(
        {
            "experiments": experiments_df,
            "per_config_summary": summary_df,
            "lambda_summary": lambda_summary,
        },
        xlsx_path,
    )

    best_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_maps": maps,
        "cloud_latency": args.cloud_latency,
        "target_ccr_band": [TARGET_CCR_LOW, TARGET_CCR_HIGH],
        "selection_metric": args.selection_metric,
        "lambda_grid": full_lambda_grid,
        "coarse_lambda_grid": lambda_grid,
        "fine_lambda_grid": fine_grid,
        "selected_lambda": float(best["age_decay_lambda"]),
        "boundary_hit": bool(
            float(best["age_decay_lambda"]) == min(full_lambda_grid)
            or float(best["age_decay_lambda"]) == max(full_lambda_grid)
        ),
        "refinement_used": bool(fine_grid),
        "coarse_winner": coarse_winner,
        "refinement_boundary": boundary_note,
        "selection_metrics": {
            "mean_collision_rate": float(best["mean_collision_rate"]),
            "mean_collision_free_rate": float(best["mean_collision_free_rate"]),
            "mean_step_cap_rate": float(best["mean_step_cap_rate"]),
            "mean_crosstrack_rmse": float(best["mean_crosstrack_rmse"]),
            "mean_crosstrack_max": float(best["mean_crosstrack_max"]),
            "mean_cloud_call_rate": float(best["mean_cloud_call_rate"]),
            "mean_abs_ccr_distance": float(best["mean_abs_ccr_distance"]),
            "frac_in_target_band": float(best["frac_in_target_band"]),
        },
        "settings": {
            "alpha_left": args.alpha_left,
            "alpha_track": args.alpha_track,
            "alpha_heading": args.alpha_heading,
            "sigma_proc_left": args.sigma_proc_left,
            "sigma_proc_track": args.sigma_proc_track,
            "sigma_proc_heading": args.sigma_proc_heading,
            "deviation_steer_weight": args.deviation_steer_weight,
            "deviation_speed_weight": args.deviation_speed_weight,
        },
        "summary": lambda_summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(best_payload, indent=2, default=_json_default))

    print(lambda_summary.to_string(index=False))
    print(f"\nSelected lambda*: {float(best['age_decay_lambda']):g}")
    print(f"Wrote raw lambda sweep CSV to {raw_csv_path}")
    print(f"Wrote lambda summary CSV to {summary_csv_path}")
    print(f"Wrote selected lambda JSON to {json_path}")
    if engine is None:
        print("Skipped XLSX (no openpyxl or xlsxwriter).")
    else:
        print(f"Wrote XLSX to {xlsx_path} using {engine}.")


if __name__ == "__main__":
    main()
