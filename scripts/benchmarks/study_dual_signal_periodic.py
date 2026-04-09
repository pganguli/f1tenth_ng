#!/usr/bin/env python3
"""Focused dual-signal periodic scheduler study with prior-result comparison."""

from __future__ import annotations

import argparse
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
    summarize,
)
from low_ccr_per_method_lambda import (
    COARSE_LAMBDA_GRID,
    SCOUT_TRAIN_MAPS,
    SUPERVISOR_SETTINGS,
    acceptable_by_k2,
    aggregate_across_maps,
    expand_lambda_grid,
    finalists_to_refined_experiments,
    per_map_winners,
    run_experiment_set,
    win_counts,
)
from map_split import NON_TRAIN_EVAL_MAPS


WEIGHT_PROFILES = {
    "devheavy": (0.20, 0.60, 0.20),
    "balanced": (0.25, 0.50, 0.25),
    "transition": (0.15, 0.55, 0.30),
}
PRIOR_BEST_CONFIGS_JSON = Path(
    "data/benchmarks/low_ccr_per_method_lambda_train_best_configs.json"
)
PRIOR_EVAL_CROSS_MAP_CSV = Path(
    "data/benchmarks/low_ccr_per_method_lambda_eval_cross_map.csv"
)
EXPLICIT_CONTROL_NAMES = (
    "fixed_interval",
    "fixed_bernoulli",
    "bernoulli_max_miss",
    "deterministic",
    "always",
    "self_normalizing_momentum",
    "srpv2",
)
DUAL_STRATEGY = "dual_signal_periodic"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Focused dual-signal periodic scheduler study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=("train", "eval", "full"),
        default="train",
    )
    parser.add_argument("--cloud-latency", type=int, default=DEFAULT_CLOUD_LATENCY)
    parser.add_argument("--coarse-trials", type=int, default=1)
    parser.add_argument("--refined-trials", type=int, default=3)
    parser.add_argument("--eval-trials", type=int, default=5)
    parser.add_argument("--max-laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--train-output-stem",
        type=str,
        default="dual_signal_periodic_train",
    )
    parser.add_argument(
        "--eval-output-stem",
        type=str,
        default="dual_signal_periodic_eval",
    )
    parser.add_argument(
        "--comparison-stem",
        type=str,
        default="dual_signal_periodic_vs_previous",
    )
    parser.add_argument(
        "--configs-json",
        type=str,
        default="data/benchmarks/dual_signal_periodic_train_best_configs.json",
    )
    parser.add_argument(
        "--prior-best-configs-json",
        type=str,
        default=str(PRIOR_BEST_CONFIGS_JSON),
    )
    parser.add_argument(
        "--prior-eval-cross-map-csv",
        type=str,
        default=str(PRIOR_EVAL_CROSS_MAP_CSV),
    )
    return parser.parse_args()


def _profile_weights(profile_name: str) -> tuple[float, float, float]:
    return WEIGHT_PROFILES[profile_name]


def family_grid() -> list[Experiment]:
    """Return the focused dual-signal search grid before lambda expansion."""
    experiments: list[Experiment] = []
    for base_interval in (3, 4, 5, 6):
        for burst_threshold in (0.60, 0.70, 0.80):
            for tau in (0.75, 1.0, 1.25):
                for profile_name in ("devheavy", "balanced", "transition"):
                    age_weight, deviation_weight, momentum_weight = _profile_weights(profile_name)
                    name = (
                        f"dual_signal_k{base_interval}_bt{int(round(burst_threshold * 100))}"
                        f"_tau{str(tau).replace('.', 'p')}_{profile_name}"
                    )
                    experiments.append(
                        Experiment(
                            name=name,
                            strategy=DUAL_STRATEGY,
                            params={
                                "base_interval": base_interval,
                                "burst_threshold": burst_threshold,
                                "tau": tau,
                                "age_weight": age_weight,
                                "deviation_weight": deviation_weight,
                                "momentum_weight": momentum_weight,
                                "deviation_cap": 0.10,
                                "age_horizon_multiplier": 2,
                                "force_age_multiplier": 3,
                                "min_extra_gap": 1,
                                "burst_queue_cap": 1,
                                "eps": 1e-8,
                                "seed": 7,
                            },
                        )
                    )
    return experiments


def rank_cross_map(cross_map_df: pd.DataFrame) -> pd.DataFrame:
    """Rank results by safety, max CTE, RMSE, then CCR."""
    return cross_map_df.sort_values(
        [
            "mean_collision_rate",
            "mean_step_cap_rate",
            "mean_crosstrack_max",
            "mean_crosstrack_rmse",
            "mean_cloud_call_rate",
            "experiment",
        ],
        kind="stable",
    ).reset_index(drop=True)


def select_top_unique_configs(cross_map_df: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    """Select top-k base experiments from a ranked cross-map dataframe."""
    finalists: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rank_cross_map(cross_map_df).itertuples(index=False):
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
                "train_metrics": {
                    "mean_collision_rate": float(row.mean_collision_rate),
                    "mean_step_cap_rate": float(row.mean_step_cap_rate),
                    "mean_crosstrack_rmse": float(row.mean_crosstrack_rmse),
                    "mean_crosstrack_max": float(row.mean_crosstrack_max),
                    "mean_cloud_call_rate": float(row.mean_cloud_call_rate),
                },
            }
        )
        if len(finalists) >= top_k:
            break
    return finalists


def aggregate_scheduler_diagnostics(experiments_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate optional scheduler diagnostic columns when present."""
    diagnostic_columns = [
        "scheduler_calls_bootstrap",
        "scheduler_calls_backbone",
        "scheduler_calls_burst",
        "scheduler_calls_force_age",
        "scheduler_calls_total",
        "scheduler_burst_fraction",
    ]
    available = [column for column in diagnostic_columns if column in experiments_df.columns]
    if not available:
        return pd.DataFrame(columns=["experiment"])
    grouped = experiments_df.groupby("experiment", as_index=False).agg(
        {column: "mean" for column in available}
    )
    rename_map = {column: f"mean_{column}" for column in available}
    return grouped.rename(columns=rename_map)


def require_prior_artifacts(
    best_configs_path: Path,
    eval_cross_map_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load prior-study artifacts or raise a clear error."""
    if not best_configs_path.exists():
        raise FileNotFoundError(
            f"Missing prior best-configs artifact: {best_configs_path}"
        )
    if not eval_cross_map_path.exists():
        raise FileNotFoundError(
            f"Missing prior eval cross-map artifact: {eval_cross_map_path}"
        )
    payload = json.loads(best_configs_path.read_text())
    prior_cross_map_df = pd.read_csv(eval_cross_map_path)
    return payload, prior_cross_map_df


def load_prior_control_experiments(payload: dict[str, Any]) -> list[Experiment]:
    """Load and validate the prior-study rerun controls."""
    experiments: list[Experiment] = []
    seen: set[str] = set()
    for info in list(payload["best_configs"].values()) + [payload["baseline_config"]]:
        name = str(info["experiment_name"])
        if name in seen:
            continue
        seen.add(name)
        experiments.append(
            Experiment(
                name=name,
                strategy=str(info["strategy"]),
                params=dict(info["params"]),
            )
        )
    available_families = set(payload["best_configs"].keys())
    required_families = {
        "fixed_interval",
        "fixed_bernoulli",
        "bernoulli_max_miss",
        "deterministic",
        "always",
        "self_normalizing_momentum",
    }
    missing_controls = sorted(required_families - available_families)
    if missing_controls:
        raise ValueError(
            "Prior best-configs artifact is missing required control families: "
            + ", ".join(missing_controls)
        )
    return experiments


def control_specs_from_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Return the named control rows that downstream comparison logic expects."""
    best_configs = payload["best_configs"]
    specs = [
        {
            "label": "baseline_k2",
            "experiment": str(payload["baseline_config"]["experiment_name"]),
            "base_experiment": str(payload["baseline_config"]["base_experiment"]),
        },
        {
            "label": "interval_best",
            "experiment": str(best_configs["fixed_interval"]["experiment_name"]),
            "base_experiment": str(best_configs["fixed_interval"]["base_experiment"]),
        },
        {
            "label": "bernoulli_best",
            "experiment": str(best_configs["bernoulli_max_miss"]["experiment_name"]),
            "base_experiment": str(best_configs["bernoulli_max_miss"]["base_experiment"]),
        },
        {
            "label": "srp_best",
            "experiment": str(best_configs["self_normalizing_momentum"]["experiment_name"]),
            "base_experiment": str(best_configs["self_normalizing_momentum"]["base_experiment"]),
        },
    ]
    srpv2_info = best_configs.get("srpv2")
    if srpv2_info is not None:
        specs.append(
            {
                "label": "srpv2_best",
                "experiment": str(srpv2_info["experiment_name"]),
                "base_experiment": str(srpv2_info["base_experiment"]),
            }
        )
    return specs


def build_comparison_frame(
    cross_map_df: pd.DataFrame,
    prior_cross_map_df: pd.DataFrame,
    prior_payload: dict[str, Any],
) -> pd.DataFrame:
    """Join current eval metrics with prior findings and control deltas."""
    prior_lookup = prior_cross_map_df[
        [
            "experiment",
            "mean_crosstrack_max",
            "mean_crosstrack_rmse",
            "mean_cloud_call_rate",
            "mean_collision_rate",
            "mean_step_cap_rate",
            "mean_lap_time_s",
            "mean_wall_min_distance",
        ]
    ].rename(
        columns={
            "mean_crosstrack_max": "stored_prev_mean_crosstrack_max",
            "mean_crosstrack_rmse": "stored_prev_mean_crosstrack_rmse",
            "mean_cloud_call_rate": "stored_prev_mean_cloud_call_rate",
            "mean_collision_rate": "stored_prev_mean_collision_rate",
            "mean_step_cap_rate": "stored_prev_mean_step_cap_rate",
            "mean_lap_time_s": "stored_prev_mean_lap_time_s",
            "mean_wall_min_distance": "stored_prev_mean_wall_min_distance",
        }
    )
    comparison = cross_map_df.merge(prior_lookup, on="experiment", how="left")

    def _row(experiment_name: str) -> pd.Series:
        rows = comparison[comparison["experiment"] == experiment_name]
        if rows.empty:
            raise ValueError(f"Missing rerun control row for {experiment_name}")
        return rows.iloc[0]

    controls: dict[str, dict[str, Any]] = {}
    for spec in control_specs_from_payload(prior_payload):
        rows = comparison[comparison["experiment"] == spec["experiment"]]
        if rows.empty:
            continue
        controls[spec["label"]] = {
            "row": rows.iloc[0],
            "experiment": spec["experiment"],
            "base_experiment": spec["base_experiment"],
        }
    for label, info in controls.items():
        row = info["row"]
        comparison[f"{label}_experiment"] = info["experiment"]
        comparison[f"{label}_base_experiment"] = info["base_experiment"]
        comparison[f"delta_max_cte_vs_{label}"] = (
            comparison["mean_crosstrack_max"] - float(row["mean_crosstrack_max"])
        )
        comparison[f"delta_rmse_vs_{label}"] = (
            comparison["mean_crosstrack_rmse"] - float(row["mean_crosstrack_rmse"])
        )
        comparison[f"delta_ccr_vs_{label}"] = (
            comparison["mean_cloud_call_rate"] - float(row["mean_cloud_call_rate"])
        )

    stored_available = comparison["stored_prev_mean_crosstrack_max"].notna()
    comparison["improved_over_stored_previous"] = pd.Series(pd.NA, index=comparison.index)
    comparison.loc[stored_available, "improved_over_stored_previous"] = (
        (
            comparison.loc[stored_available, "mean_crosstrack_max"]
            <= comparison.loc[stored_available, "stored_prev_mean_crosstrack_max"]
        )
        & (
            comparison.loc[stored_available, "mean_cloud_call_rate"]
            <= comparison.loc[stored_available, "stored_prev_mean_cloud_call_rate"]
        )
    )
    comparison["improved_over_rerun_k3_control"] = (
        (
            comparison["mean_crosstrack_max"]
            <= float(controls["interval_best"]["row"]["mean_crosstrack_max"])
        )
        & (
            comparison["mean_cloud_call_rate"]
            < float(controls["interval_best"]["row"]["mean_cloud_call_rate"])
        )
    )
    comparison["meets_primary_success"] = (
        comparison["acceptable"]
        & (
            comparison["mean_cloud_call_rate"]
            < float(controls["interval_best"]["row"]["mean_cloud_call_rate"])
        )
    )
    return comparison.sort_values(
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


def write_train_outputs(
    stem: str,
    experiments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cross_map_df: pd.DataFrame,
    scout_finalists: list[dict[str, Any]],
    selected_dual_configs: list[dict[str, Any]],
    prior_best_configs_json: Path,
    prior_eval_cross_map_csv: Path,
    explicit_controls: list[str],
    coarse_trials: int,
    refined_trials: int,
    settings: PlannerSettings,
    cloud_latency: int,
) -> Path:
    """Write train-stage artifacts and return the config JSON path."""
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
        "train_maps": list(SCOUT_TRAIN_MAPS),
        "cloud_latency": cloud_latency,
        "settings": settings.__dict__,
        "coarse_trials": coarse_trials,
        "refined_trials": refined_trials,
        "coarse_lambda_grid": COARSE_LAMBDA_GRID,
        "scout_finalists": scout_finalists,
        "selected_dual_configs": selected_dual_configs,
        "prior_artifacts": {
            "best_configs_json": str(prior_best_configs_json),
            "eval_cross_map_csv": str(prior_eval_cross_map_csv),
        },
        "explicit_controls": explicit_controls,
        "summary": summary_df.to_dict(orient="records"),
        "cross_map_summary": cross_map_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    best_configs_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return best_configs_path


def write_eval_outputs(
    stem: str,
    experiments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cross_map_df: pd.DataFrame,
    acceptable_df: pd.DataFrame,
    per_map_df: pd.DataFrame,
    wins_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    train_payload: dict[str, Any],
    baseline_row: pd.Series,
) -> Path:
    """Write eval-stage artifacts and return the JSON path."""
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
            "comparison": comparison_df,
        },
        xlsx_path,
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_maps": list(NON_TRAIN_EVAL_MAPS),
        "baseline_experiment": baseline_row["experiment"],
        "baseline_mean_crosstrack_max": float(baseline_row["mean_crosstrack_max"]),
        "baseline_mean_crosstrack_rmse": float(baseline_row["mean_crosstrack_rmse"]),
        "baseline_mean_cloud_call_rate": float(baseline_row["mean_cloud_call_rate"]),
        "k2_threshold": float(baseline_row["mean_crosstrack_max"]) * 1.01,
        "train_config_source": train_payload,
        "summary": summary_df.to_dict(orient="records"),
        "cross_map_summary": cross_map_df.to_dict(orient="records"),
        "acceptable_configs": acceptable_df.to_dict(orient="records"),
        "per_map_winners": per_map_df.to_dict(orient="records"),
        "win_counts": wins_df.to_dict(orient="records"),
        "comparison": comparison_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return json_path


def write_comparison_outputs(
    stem: str,
    comparison_df: pd.DataFrame,
    train_payload: dict[str, Any],
    prior_cross_map_df: pd.DataFrame,
) -> Path:
    """Write explicit comparison artifacts."""
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    comparison_df.to_csv(csv_path, index=False)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prior_findings_source": train_payload["prior_artifacts"],
        "prior_cross_map_rows": prior_cross_map_df.to_dict(orient="records"),
        "comparison": comparison_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return json_path


def train_phase(args: argparse.Namespace) -> Path:
    """Run scout/refine on the dual-signal family and write train artifacts."""
    prior_payload, prior_cross_map_df = require_prior_artifacts(
        Path(args.prior_best_configs_json),
        Path(args.prior_eval_cross_map_csv),
    )
    del prior_cross_map_df
    base_grid = family_grid()
    coarse_experiments = expand_lambda_grid(base_grid, COARSE_LAMBDA_GRID)
    coarse_results = run_experiment_set(
        maps=list(SCOUT_TRAIN_MAPS),
        experiments=coarse_experiments,
        cloud_latency=args.cloud_latency,
        max_laps=args.max_laps,
        settings=SUPERVISOR_SETTINGS,
        trials=args.coarse_trials,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    coarse_experiments_df, coarse_summary_df, _ = summarize(
        coarse_results,
        selection_metric="max_cte",
    )
    coarse_cross_map_df = aggregate_across_maps(coarse_experiments_df, coarse_summary_df)
    scout_finalists = select_top_unique_configs(coarse_cross_map_df, top_k=4)
    refined_experiments = finalists_to_refined_experiments(scout_finalists, COARSE_LAMBDA_GRID)

    combined_results = list(coarse_results)
    if refined_experiments:
        combined_results.extend(
            run_experiment_set(
                maps=list(SCOUT_TRAIN_MAPS),
                experiments=refined_experiments,
                cloud_latency=args.cloud_latency,
                max_laps=args.max_laps,
                settings=SUPERVISOR_SETTINGS,
                trials=args.refined_trials,
                max_steps=args.max_steps,
                workers=args.workers,
            )
        )

    experiments_df, summary_df, _ = summarize(combined_results, selection_metric="max_cte")
    cross_map_df = aggregate_across_maps(experiments_df, summary_df)
    cross_map_df = cross_map_df.merge(
        aggregate_scheduler_diagnostics(experiments_df),
        on="experiment",
        how="left",
    )
    selected_dual_configs = select_top_unique_configs(cross_map_df, top_k=2)
    return write_train_outputs(
        stem=args.train_output_stem,
        experiments_df=experiments_df,
        summary_df=summary_df,
        cross_map_df=cross_map_df,
        scout_finalists=scout_finalists,
        selected_dual_configs=selected_dual_configs,
        prior_best_configs_json=Path(args.prior_best_configs_json),
        prior_eval_cross_map_csv=Path(args.prior_eval_cross_map_csv),
        explicit_controls=[spec["experiment"] for spec in control_specs_from_payload(prior_payload)],
        coarse_trials=args.coarse_trials,
        refined_trials=args.refined_trials,
        settings=SUPERVISOR_SETTINGS,
        cloud_latency=args.cloud_latency,
    )


def eval_phase(args: argparse.Namespace, configs_path: Path | None = None) -> tuple[Path, Path]:
    """Rerun prior controls with the selected dual-signal finalists."""
    train_payload = json.loads((configs_path or Path(args.configs_json)).read_text())
    prior_payload, prior_cross_map_df = require_prior_artifacts(
        Path(args.prior_best_configs_json),
        Path(args.prior_eval_cross_map_csv),
    )
    selected_dual_experiments = [
        Experiment(
            name=str(info.get("experiment_name", info["experiment"])),
            strategy=str(info["strategy"]),
            params=dict(info["params"]),
        )
        for info in train_payload["selected_dual_configs"]
    ]
    prior_experiments = load_prior_control_experiments(prior_payload)
    experiments: list[Experiment] = []
    seen: set[str] = set()
    for exp in selected_dual_experiments + prior_experiments:
        if exp.name in seen:
            continue
        seen.add(exp.name)
        experiments.append(exp)

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
    cross_map_df = cross_map_df.merge(
        aggregate_scheduler_diagnostics(experiments_df),
        on="experiment",
        how="left",
    )
    cross_map_df, baseline_row = acceptable_by_k2(
        cross_map_df,
        baseline_experiment_name=str(prior_payload["baseline_config"]["experiment_name"]),
    )
    acceptable_df = cross_map_df[cross_map_df["acceptable"]].sort_values(
        ["mean_cloud_call_rate", "mean_crosstrack_rmse", "mean_crosstrack_max", "experiment"],
        kind="stable",
    )
    per_map_df = per_map_winners(summary_df)
    wins_df = win_counts(per_map_df)
    comparison_df = build_comparison_frame(cross_map_df, prior_cross_map_df, prior_payload)

    eval_json = write_eval_outputs(
        stem=args.eval_output_stem,
        experiments_df=experiments_df,
        summary_df=summary_df,
        cross_map_df=cross_map_df,
        acceptable_df=acceptable_df,
        per_map_df=per_map_df,
        wins_df=wins_df,
        comparison_df=comparison_df,
        train_payload=train_payload,
        baseline_row=baseline_row,
    )
    comparison_json = write_comparison_outputs(
        stem=args.comparison_stem,
        comparison_df=comparison_df,
        train_payload=train_payload,
        prior_cross_map_df=prior_cross_map_df,
    )
    return eval_json, comparison_json


def main() -> None:
    """Run the requested study phase."""
    args = parse_args()
    configs_path: Path | None = None
    if args.phase in ("train", "full"):
        configs_path = train_phase(args)
        print(f"Wrote train-time configs to {configs_path}")
    if args.phase in ("eval", "full"):
        eval_json, comparison_json = eval_phase(args, configs_path=configs_path)
        print(f"Wrote eval report to {eval_json}")
        print(f"Wrote comparison report to {comparison_json}")


if __name__ == "__main__":
    main()
