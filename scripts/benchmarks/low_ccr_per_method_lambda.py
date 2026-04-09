#!/usr/bin/env python3
"""Exploratory low-CCR benchmark with per-method lambda refinement."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from benchmark_single_tier_paper_strategies import (
    DEFAULT_CLOUD_LATENCY,
    DEFAULT_MAX_STEPS,
    Experiment,
    PlannerSettings,
    _json_default,
    maybe_write_xlsx,
    run_episode,
    summarize,
    track_config,
)
from map_split import NON_TRAIN_EVAL_MAPS
from sweep_age_decay_lambda import build_refinement_grid


FAMILY_ORDER = [
    "always",
    "fixed_interval",
    "fixed_bernoulli",
    "bernoulli_max_miss",
    "deterministic",
    "self_normalizing_momentum",
    "srpv2",
]

COARSE_LAMBDA_GRID = [0.0, 1.0, 4.0, 16.0, 24.0, 64.0]
K2_EQUIVALENCE_FACTOR = 1.01
SCOUT_TRAIN_MAPS = [
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "IMS",
    "Melbourne",
    "Oschersleben",
    "Sakhir",
    "Sepang",
    "Zandvoort",
]
FINALIST_LIMITS = {
    "always": 1,
    "fixed_interval": 5,
    "fixed_bernoulli": 3,
    "bernoulli_max_miss": 3,
    "deterministic": 2,
    "self_normalizing_momentum": 4,
    "srpv2": 4,
}

SUPERVISOR_SETTINGS = PlannerSettings(
    alpha_left=0.996,
    alpha_track=0.988,
    alpha_heading=0.974,
    sigma_proc_left=0.044961,
    sigma_proc_track=0.067937,
    sigma_proc_heading=0.033182,
    age_decay_lambda=24.0,
)


def lambda_suffix(value: float) -> str:
    """Return a filesystem-friendly label for a lambda value."""
    return str(value).replace(".", "p")


def base_experiment_name(experiment_name: str) -> str:
    """Strip any lambda suffix from an experiment name."""
    return experiment_name.split("__lambda_", maxsplit=1)[0]


def family_grid() -> list[Experiment]:
    """Return the low-CCR study grid before lambda expansion."""
    experiments = [
        Experiment("always_hit", "always", {}),
    ]
    for interval in (2, 3, 5, 7, 10):
        experiments.append(
            Experiment(
                f"fixed_interval_k{interval}",
                "fixed_interval",
                {"interval": interval},
            )
        )
    for probability in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45):
        experiments.append(
            Experiment(
                f"fixed_bernoulli_p{int(round(probability * 100))}",
                "fixed_bernoulli",
                {"p": probability, "seed": 7},
            )
        )
    for probability in (0.15, 0.25, 0.40):
        for max_miss in (2, 3, 5):
            experiments.append(
                Experiment(
                    f"bernoulli_max_miss_p{int(round(probability * 100))}_m{max_miss}",
                    "bernoulli_max_miss",
                    {"p": probability, "max_miss": max_miss, "seed": 7},
                )
            )
    for threshold in (0.10, 0.20, 0.30):
        experiments.append(
            Experiment(
                f"deterministic_t{str(threshold).replace('.', 'p')}",
                "deterministic",
                {"threshold": threshold},
            )
        )
    for tau in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for nmax in (2, 3, 4, 5):
            experiments.append(
                Experiment(
                    f"self_norm_tau{str(tau).replace('.', 'p')}_n{nmax}",
                    "self_normalizing_momentum",
                    {
                        "tau": tau,
                        "nmax": nmax,
                        "seed": 7,
                    },
                )
            )
            experiments.append(
                Experiment(
                    f"srpv2_tau{str(tau).replace('.', 'p')}_n{nmax}",
                    "srpv2",
                    {
                        "tau": tau,
                        "nmax": nmax,
                        "seed": 7,
                    },
                )
            )
    return experiments


def experiment_with_lambda(exp: Experiment, age_decay_lambda: float) -> Experiment:
    """Return a lambda-specialized copy of an experiment."""
    params = dict(exp.params)
    params["age_decay_lambda"] = float(age_decay_lambda)
    return Experiment(
        name=f"{exp.name}__lambda_{lambda_suffix(age_decay_lambda)}",
        strategy=exp.strategy,
        params=params,
    )


def expand_lambda_grid(
    experiments: list[Experiment],
    lambda_grid: list[float],
) -> list[Experiment]:
    """Expand each base experiment across a lambda grid."""
    expanded: list[Experiment] = []
    for exp in experiments:
        for age_decay_lambda in lambda_grid:
            expanded.append(experiment_with_lambda(exp, age_decay_lambda))
    return expanded


def _run_map_experiments(
    map_name: str,
    experiments: list[Experiment],
    cloud_latency: int,
    max_laps: int,
    settings: PlannerSettings,
    trials: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Run all experiments for one map."""
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
            row["age_decay_lambda"] = float(exp.params["age_decay_lambda"])
            row["base_experiment"] = base_experiment_name(exp.name)
            results.append(row)
    return results


def run_experiment_set(
    maps: list[str],
    experiments: list[Experiment],
    cloud_latency: int,
    max_laps: int,
    settings: PlannerSettings,
    trials: int,
    max_steps: int,
    workers: int,
) -> list[dict[str, Any]]:
    """Run a batch of experiments, optionally parallelized across maps."""
    if workers > 1:
        results: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_map_experiments,
                    map_name,
                    experiments,
                    cloud_latency,
                    max_laps,
                    settings,
                    trials,
                    max_steps,
                ): map_name
                for map_name in maps
            }
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
    results = []
    for map_name in maps:
        results.extend(
            _run_map_experiments(
                map_name,
                experiments,
                cloud_latency,
                max_laps,
                settings,
                trials,
                max_steps,
            )
        )
    return results


def aggregate_across_maps(
    experiments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate summary metrics across maps to one row per experiment."""
    params_lookup = (
        experiments_df.groupby("experiment", as_index=False)
        .agg(
            strategy=("strategy", "first"),
            params_json=("params_json", "first"),
            base_experiment=("base_experiment", "first"),
            age_decay_lambda=("age_decay_lambda", "first"),
        )
    )
    cross_map = (
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
        .fillna(0.0)
    )
    cross_map = cross_map.merge(
        params_lookup[["experiment", "params_json", "base_experiment", "age_decay_lambda"]],
        on="experiment",
        how="left",
    )
    return rank_cross_map(cross_map).reset_index(drop=True)


def rank_cross_map(cross_map_df: pd.DataFrame) -> pd.DataFrame:
    """Rank train-phase cross-map results using safety, max CTE, RMSE, then CCR."""
    return cross_map_df.sort_values(
        [
            "mean_collision_rate",
            "mean_step_cap_rate",
            "mean_crosstrack_max",
            "mean_crosstrack_rmse",
            "mean_cloud_call_rate",
            "strategy",
            "experiment",
        ],
        kind="stable",
    )


def select_top_k_per_family(
    cross_map_df: pd.DataFrame,
    finalist_limits: dict[str, int],
) -> list[dict[str, Any]]:
    """Select the top-k unique base experiments per family."""
    finalists: list[dict[str, Any]] = []
    for strategy in FAMILY_ORDER:
        family = cross_map_df[cross_map_df["strategy"] == strategy].copy()
        limit = finalist_limits.get(strategy, 0)
        if family.empty or limit <= 0:
            continue
        seen: set[str] = set()
        for row in family.itertuples(index=False):
            if row.base_experiment in seen:
                continue
            seen.add(row.base_experiment)
            finalists.append(
                {
                    "strategy": row.strategy,
                    "experiment": row.experiment,
                    "base_experiment": row.base_experiment,
                    "age_decay_lambda": float(row.age_decay_lambda),
                    "params": json.loads(row.params_json or "{}"),
                }
            )
            if len(seen) >= limit:
                break
    return finalists


def finalists_to_refined_experiments(
    finalists: list[dict[str, Any]],
    coarse_grid: list[float],
) -> list[Experiment]:
    """Build fine-grid experiments around each finalist's best coarse lambda."""
    refined: list[Experiment] = []
    for finalist in finalists:
        fine_grid, _ = build_refinement_grid(
            coarse_grid=coarse_grid,
            winner=float(finalist["age_decay_lambda"]),
        )
        if not fine_grid:
            continue
        params = dict(finalist["params"])
        params.pop("age_decay_lambda", None)
        base_exp = Experiment(
            name=str(finalist["base_experiment"]),
            strategy=str(finalist["strategy"]),
            params=params,
        )
        refined.extend(expand_lambda_grid([base_exp], fine_grid))
    return refined


def select_best_per_family(cross_map_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Select the best experiment for each strategy family."""
    best: dict[str, dict[str, Any]] = {}
    for strategy in FAMILY_ORDER:
        family = cross_map_df[cross_map_df["strategy"] == strategy].copy()
        if family.empty:
            continue
        winner = family.iloc[0]
        params = json.loads(winner["params_json"] or "{}")
        best[strategy] = {
            "experiment_name": winner["experiment"],
            "base_experiment": winner["base_experiment"],
            "strategy": strategy,
            "params": params,
            "train_metrics": {
                "mean_collision_rate": float(winner["mean_collision_rate"]),
                "mean_step_cap_rate": float(winner["mean_step_cap_rate"]),
                "mean_crosstrack_rmse": float(winner["mean_crosstrack_rmse"]),
                "mean_crosstrack_max": float(winner["mean_crosstrack_max"]),
                "mean_cloud_call_rate": float(winner["mean_cloud_call_rate"]),
            },
        }
    return best


def select_fixed_interval_k2(cross_map_df: pd.DataFrame) -> dict[str, Any]:
    """Select the best lambda-specialized fixed_interval_k2 baseline."""
    candidates = cross_map_df[cross_map_df["base_experiment"] == "fixed_interval_k2"].copy()
    if candidates.empty:
        raise ValueError("No fixed_interval_k2 candidate found in the train-time results.")
    winner = candidates.iloc[0]
    return {
        "experiment_name": winner["experiment"],
        "base_experiment": winner["base_experiment"],
        "strategy": winner["strategy"],
        "params": json.loads(winner["params_json"] or "{}"),
        "train_metrics": {
            "mean_collision_rate": float(winner["mean_collision_rate"]),
            "mean_step_cap_rate": float(winner["mean_step_cap_rate"]),
            "mean_crosstrack_rmse": float(winner["mean_crosstrack_rmse"]),
            "mean_crosstrack_max": float(winner["mean_crosstrack_max"]),
            "mean_cloud_call_rate": float(winner["mean_cloud_call_rate"]),
        },
    }


def select_fixed_interval_controls(cross_map_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Select the best lambda-specialized rerun control for each interval ladder point."""
    controls: list[dict[str, Any]] = []
    for interval in (2, 3, 5, 7, 10):
        base_name = f"fixed_interval_k{interval}"
        candidates = cross_map_df[cross_map_df["base_experiment"] == base_name].copy()
        if candidates.empty:
            continue
        winner = candidates.iloc[0]
        controls.append(
            {
                "experiment_name": winner["experiment"],
                "base_experiment": winner["base_experiment"],
                "strategy": winner["strategy"],
                "params": json.loads(winner["params_json"] or "{}"),
                "train_metrics": {
                    "mean_collision_rate": float(winner["mean_collision_rate"]),
                    "mean_step_cap_rate": float(winner["mean_step_cap_rate"]),
                    "mean_crosstrack_rmse": float(winner["mean_crosstrack_rmse"]),
                    "mean_crosstrack_max": float(winner["mean_crosstrack_max"]),
                    "mean_cloud_call_rate": float(winner["mean_cloud_call_rate"]),
                },
            }
        )
    return controls


def load_eval_experiments(path: Path) -> tuple[list[Experiment], dict[str, Any]]:
    """Load the selected best configs and mandatory k2 baseline."""
    payload = json.loads(path.read_text())
    experiments: list[Experiment] = []
    seen: set[str] = set()
    infos = (
        list(payload["best_configs"].values())
        + list(payload.get("fixed_interval_controls", []))
        + [payload["baseline_config"]]
    )
    for info in infos:
        name = info["experiment_name"]
        if name in seen:
            continue
        seen.add(name)
        experiments.append(
            Experiment(
                name=name,
                strategy=info["strategy"],
                params=info["params"],
            )
        )
    return experiments, payload


def acceptable_by_k2(
    cross_map_df: pd.DataFrame,
    baseline_experiment_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Annotate configs that satisfy the held-out k2-relative acceptance rule."""
    baseline_rows = cross_map_df[cross_map_df["experiment"] == baseline_experiment_name]
    if baseline_rows.empty:
        raise ValueError(f"Missing fixed_interval_k2 baseline row: {baseline_experiment_name}")
    baseline_row = baseline_rows.iloc[0]
    threshold = float(baseline_row["mean_crosstrack_max"]) * K2_EQUIVALENCE_FACTOR

    ranked = cross_map_df.copy()
    ranked["baseline_experiment"] = baseline_experiment_name
    ranked["baseline_mean_crosstrack_max"] = float(baseline_row["mean_crosstrack_max"])
    ranked["baseline_mean_crosstrack_rmse"] = float(baseline_row["mean_crosstrack_rmse"])
    ranked["baseline_mean_cloud_call_rate"] = float(baseline_row["mean_cloud_call_rate"])
    ranked["k2_threshold"] = threshold
    ranked["within_k2_band"] = ranked["mean_crosstrack_max"] <= threshold
    ranked["collision_free_acceptance"] = ranked["mean_collision_rate"] <= 0.0
    ranked["step_cap_free_acceptance"] = ranked["mean_step_cap_rate"] <= 0.0
    ranked["acceptable"] = (
        ranked["collision_free_acceptance"]
        & ranked["step_cap_free_acceptance"]
        & ranked["within_k2_band"]
    )
    ranked["max_cte_delta_vs_k2"] = (
        ranked["mean_crosstrack_max"] - float(baseline_row["mean_crosstrack_max"])
    )
    ranked["max_cte_delta_vs_k2_pct"] = (
        ranked["max_cte_delta_vs_k2"] / max(float(baseline_row["mean_crosstrack_max"]), 1e-8)
    )
    ranked["rmse_delta_vs_k2"] = (
        ranked["mean_crosstrack_rmse"] - float(baseline_row["mean_crosstrack_rmse"])
    )
    ranked["rmse_delta_vs_k2_pct"] = (
        ranked["rmse_delta_vs_k2"] / max(float(baseline_row["mean_crosstrack_rmse"]), 1e-8)
    )
    ranked["ccr_reduction_vs_k2"] = (
        float(baseline_row["mean_cloud_call_rate"]) - ranked["mean_cloud_call_rate"]
    )
    acceptable = ranked[ranked["acceptable"]].copy().sort_values(
        [
            "mean_cloud_call_rate",
            "mean_crosstrack_rmse",
            "mean_crosstrack_max",
            "experiment",
        ],
        kind="stable",
    )
    if not acceptable.empty:
        acceptable["acceptable_rank"] = range(1, len(acceptable) + 1)
    ranked = ranked.merge(
        acceptable[["experiment", "acceptable_rank"]],
        on="experiment",
        how="left",
    )
    ranked = ranked.sort_values(
        [
            "acceptable",
            "mean_cloud_call_rate",
            "mean_crosstrack_rmse",
            "mean_crosstrack_max",
            "experiment",
        ],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return ranked, baseline_row


def per_map_winners(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return one winner row per map using max-CTE, RMSE, then CCR."""
    return (
        summary_df.sort_values(
            [
                "map_name",
                "collision_rate",
                "step_cap_rate",
                "crosstrack_max_m_mean",
                "crosstrack_rmse_m_mean",
                "cloud_call_rate_mean",
                "experiment",
            ],
            kind="stable",
        )
        .groupby("map_name", as_index=False)
        .first()
    )


def win_counts(per_map_df: pd.DataFrame) -> pd.DataFrame:
    """Return map-win counts by experiment."""
    if per_map_df.empty:
        return pd.DataFrame(columns=["experiment", "strategy", "maps_won"])
    return (
        per_map_df.groupby(["experiment", "strategy"], as_index=False)
        .size()
        .rename(columns={"size": "maps_won"})
        .sort_values(["maps_won", "experiment"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Low-CCR exploratory benchmark with per-method lambda refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=("train", "eval", "full"),
        default="train",
        help="Whether to run train-time search, held-out eval, or both.",
    )
    parser.add_argument("--cloud-latency", type=int, default=DEFAULT_CLOUD_LATENCY)
    parser.add_argument("--train-trials", type=int, default=3)
    parser.add_argument(
        "--coarse-trials",
        type=int,
        default=1,
        help="Scout-stage train trials.",
    )
    parser.add_argument(
        "--refined-trials",
        type=int,
        default=None,
        help="Optional override for refined-stage train trials (defaults to --train-trials).",
    )
    parser.add_argument("--eval-trials", type=int, default=5)
    parser.add_argument("--max-laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--train-output-stem",
        type=str,
        default="low_ccr_per_method_lambda_train",
    )
    parser.add_argument(
        "--eval-output-stem",
        type=str,
        default="low_ccr_per_method_lambda_eval",
    )
    parser.add_argument(
        "--configs-json",
        type=str,
        default="data/benchmarks/low_ccr_per_method_lambda_train_best_configs.json",
        help="Train-time config JSON used by eval mode.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategy families to include in the train-time search.",
    )
    return parser.parse_args()


def write_train_outputs(
    stem: str,
    maps: list[str],
    cloud_latency: int,
    settings: PlannerSettings,
    coarse_trials: int,
    refined_trials: int,
    experiments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cross_map_df: pd.DataFrame,
    best_configs: dict[str, dict[str, Any]],
    baseline_config: dict[str, Any],
    fixed_interval_controls: list[dict[str, Any]],
    finalists: list[dict[str, Any]],
) -> Path:
    """Write train-time artifacts and return the best-configs JSON path."""
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    cross_map_csv_path = out_dir / f"{stem}_cross_map.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"
    json_path = out_dir / f"{stem}.json"
    best_configs_path = out_dir / f"{stem}_best_configs.json"

    experiments_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    cross_map_df.to_csv(cross_map_csv_path, index=False)
    maybe_write_xlsx(
        {
            "experiments": experiments_df,
            "summary": summary_df,
            "cross_map": cross_map_df,
        },
        xlsx_path,
    )
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_maps": maps,
        "cloud_latency": cloud_latency,
        "settings": settings.__dict__,
        "coarse_trials": coarse_trials,
        "refined_trials": refined_trials,
        "coarse_lambda_grid": COARSE_LAMBDA_GRID,
        "finalist_limits": FINALIST_LIMITS,
        "best_configs": best_configs,
        "baseline_config": baseline_config,
        "fixed_interval_controls": fixed_interval_controls,
        "finalists": finalists,
        "summary": summary_df.to_dict(orient="records"),
        "cross_map_summary": cross_map_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    best_configs_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return best_configs_path


def train_phase(args: argparse.Namespace) -> Path:
    """Run the train-time search and write artifacts."""
    base_grid = family_grid()
    if args.strategies:
        allowed = {value.strip() for value in args.strategies.split(",") if value.strip()}
        base_grid = [exp for exp in base_grid if exp.strategy in allowed]
    coarse_trials = args.coarse_trials if args.coarse_trials is not None else args.train_trials
    refined_trials = args.refined_trials if args.refined_trials is not None else args.train_trials
    coarse_experiments = expand_lambda_grid(base_grid, COARSE_LAMBDA_GRID)
    coarse_results = run_experiment_set(
        maps=list(SCOUT_TRAIN_MAPS),
        experiments=coarse_experiments,
        cloud_latency=args.cloud_latency,
        max_laps=args.max_laps,
        settings=SUPERVISOR_SETTINGS,
        trials=coarse_trials,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    coarse_experiments_df, coarse_summary_df, _ = summarize(
        coarse_results,
        selection_metric="max_cte",
    )
    coarse_cross_map = aggregate_across_maps(coarse_experiments_df, coarse_summary_df)
    finalists = select_top_k_per_family(coarse_cross_map, finalist_limits=FINALIST_LIMITS)
    refined_experiments = finalists_to_refined_experiments(finalists, COARSE_LAMBDA_GRID)

    combined_results = list(coarse_results)
    if refined_experiments:
        combined_results.extend(
            run_experiment_set(
                maps=list(SCOUT_TRAIN_MAPS),
                experiments=refined_experiments,
                cloud_latency=args.cloud_latency,
                max_laps=args.max_laps,
                settings=SUPERVISOR_SETTINGS,
                trials=refined_trials,
                max_steps=args.max_steps,
                workers=args.workers,
            )
        )
    experiments_df, summary_df, _ = summarize(combined_results, selection_metric="max_cte")
    cross_map_df = aggregate_across_maps(experiments_df, summary_df)
    best_configs = select_best_per_family(cross_map_df)
    baseline_config = select_fixed_interval_k2(cross_map_df)
    fixed_interval_controls = select_fixed_interval_controls(cross_map_df)
    return write_train_outputs(
        stem=args.train_output_stem,
        maps=list(SCOUT_TRAIN_MAPS),
        cloud_latency=args.cloud_latency,
        settings=SUPERVISOR_SETTINGS,
        coarse_trials=coarse_trials,
        refined_trials=refined_trials,
        experiments_df=experiments_df,
        summary_df=summary_df,
        cross_map_df=cross_map_df,
        best_configs=best_configs,
        baseline_config=baseline_config,
        fixed_interval_controls=fixed_interval_controls,
        finalists=finalists,
    )


def write_eval_outputs(
    stem: str,
    maps: list[str],
    cloud_latency: int,
    settings: PlannerSettings,
    experiments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cross_map_df: pd.DataFrame,
    acceptable_df: pd.DataFrame,
    per_map_df: pd.DataFrame,
    wins_df: pd.DataFrame,
    baseline_row: pd.Series,
    train_payload: dict[str, Any],
) -> Path:
    """Write held-out eval artifacts and return the JSON path."""
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    cross_map_csv_path = out_dir / f"{stem}_cross_map.csv"
    acceptable_csv_path = out_dir / f"{stem}_acceptable.csv"
    per_map_csv_path = out_dir / f"{stem}_per_map_winners.csv"
    wins_csv_path = out_dir / f"{stem}_win_counts.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"
    json_path = out_dir / f"{stem}.json"

    experiments_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    cross_map_df.to_csv(cross_map_csv_path, index=False)
    acceptable_df.to_csv(acceptable_csv_path, index=False)
    per_map_df.to_csv(per_map_csv_path, index=False)
    wins_df.to_csv(wins_csv_path, index=False)
    maybe_write_xlsx(
        {
            "experiments": experiments_df,
            "summary": summary_df,
            "cross_map": cross_map_df,
            "acceptable": acceptable_df,
            "per_map_winners": per_map_df,
            "win_counts": wins_df,
        },
        xlsx_path,
    )
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_maps": maps,
        "cloud_latency": cloud_latency,
        "settings": settings.__dict__,
        "baseline_experiment": baseline_row["experiment"],
        "baseline_mean_crosstrack_max": float(baseline_row["mean_crosstrack_max"]),
        "baseline_mean_crosstrack_rmse": float(baseline_row["mean_crosstrack_rmse"]),
        "baseline_mean_cloud_call_rate": float(baseline_row["mean_cloud_call_rate"]),
        "k2_threshold": float(baseline_row["mean_crosstrack_max"]) * K2_EQUIVALENCE_FACTOR,
        "train_config_source": train_payload,
        "summary": summary_df.to_dict(orient="records"),
        "cross_map_summary": cross_map_df.to_dict(orient="records"),
        "acceptable_configs": acceptable_df.to_dict(orient="records"),
        "per_map_winners": per_map_df.to_dict(orient="records"),
        "win_counts": wins_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return json_path


def eval_phase(args: argparse.Namespace, configs_path: Path | None = None) -> Path:
    """Run the held-out evaluation and write artifacts."""
    configs_file = configs_path or Path(args.configs_json)
    experiments, train_payload = load_eval_experiments(configs_file)
    eval_results = run_experiment_set(
        maps=list(NON_TRAIN_EVAL_MAPS),
        experiments=experiments,
        cloud_latency=args.cloud_latency,
        max_laps=args.max_laps,
        settings=SUPERVISOR_SETTINGS,
        trials=args.eval_trials,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    experiments_df, summary_df, _ = summarize(eval_results, selection_metric="max_cte")
    cross_map_df = aggregate_across_maps(experiments_df, summary_df)
    cross_map_df, baseline_row = acceptable_by_k2(
        cross_map_df,
        baseline_experiment_name=train_payload["baseline_config"]["experiment_name"],
    )
    acceptable_df = cross_map_df[cross_map_df["acceptable"]].sort_values(
        ["mean_cloud_call_rate", "mean_crosstrack_rmse", "mean_crosstrack_max", "experiment"],
        kind="stable",
    )
    per_map_df = per_map_winners(summary_df)
    wins_df = win_counts(per_map_df)
    return write_eval_outputs(
        stem=args.eval_output_stem,
        maps=list(NON_TRAIN_EVAL_MAPS),
        cloud_latency=args.cloud_latency,
        settings=SUPERVISOR_SETTINGS,
        experiments_df=experiments_df,
        summary_df=summary_df,
        cross_map_df=cross_map_df,
        acceptable_df=acceptable_df,
        per_map_df=per_map_df,
        wins_df=wins_df,
        baseline_row=baseline_row,
        train_payload=train_payload,
    )


def main() -> None:
    """Run the requested low-CCR study phase."""
    args = parse_args()
    configs_path: Path | None = None
    if args.phase in ("train", "full"):
        configs_path = train_phase(args)
        print(f"Wrote train-time configs to {configs_path}")
    if args.phase in ("eval", "full"):
        eval_json = eval_phase(args, configs_path=configs_path)
        print(f"Wrote eval report to {eval_json}")


if __name__ == "__main__":
    main()
