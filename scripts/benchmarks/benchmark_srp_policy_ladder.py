#!/usr/bin/env python3
"""Focused latency-10 policy ladder study for SRP, SRPv2, and interval baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from benchmark_single_tier_paper_strategies import (
    DEFAULT_MAX_STEPS,
    Experiment,
    _json_default,
    maybe_write_xlsx,
    summarize,
)
from low_ccr_per_method_lambda import (
    SCOUT_TRAIN_MAPS,
    SUPERVISOR_SETTINGS,
    acceptable_by_k2,
    aggregate_across_maps,
    expand_lambda_grid,
    lambda_suffix,
    rank_cross_map,
    run_experiment_set,
)
from map_split import NON_TRAIN_EVAL_MAPS, SANITY_EVAL_MAPS, SANITY_TRAIN_MAPS


DEFAULT_CLOUD_LATENCY = 10
DEFAULT_BOOTSTRAP_SAMPLES = 5000
DEFAULT_INTERVAL_GRID = [2, 3, 4, 7, 10, 15]
DEFAULT_TAU_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
DEFAULT_NMAX_GRID = [2, 3, 4, 5]
DEFAULT_LAMBDA_GRID = [0.0, 1.0, 4.0, 16.0, 24.0, 64.0]
DEFAULT_MIN_SHORTLIST = 4
DEFAULT_MAX_SHORTLIST = 6


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the focused SRP/SRPv2 policy ladder study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", choices=("train", "eval", "full"), default="full")
    parser.add_argument("--cloud-latency", type=int, default=DEFAULT_CLOUD_LATENCY)
    parser.add_argument("--train-maps", type=str, default=",".join(SCOUT_TRAIN_MAPS))
    parser.add_argument("--eval-maps", type=str, default=",".join(NON_TRAIN_EVAL_MAPS))
    parser.add_argument("--interval-grid", type=str, default=",".join(map(str, DEFAULT_INTERVAL_GRID)))
    parser.add_argument("--tau-grid", type=str, default=",".join(map(str, DEFAULT_TAU_GRID)))
    parser.add_argument("--nmax-grid", type=str, default=",".join(map(str, DEFAULT_NMAX_GRID)))
    parser.add_argument("--lambda-grid", type=str, default=",".join(map(str, DEFAULT_LAMBDA_GRID)))
    parser.add_argument("--train-trials", type=int, default=1)
    parser.add_argument("--eval-trials", type=int, default=5)
    parser.add_argument("--max-laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-shortlist", type=int, default=DEFAULT_MIN_SHORTLIST)
    parser.add_argument("--max-shortlist", type=int, default=DEFAULT_MAX_SHORTLIST)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument(
        "--train-output-stem",
        type=str,
        default="srp_policy_ladder_L10_train",
    )
    parser.add_argument(
        "--eval-output-stem",
        type=str,
        default="srp_policy_ladder_L10_eval",
    )
    parser.add_argument(
        "--selected-configs-json",
        type=str,
        default="data/benchmarks/srp_policy_ladder_L10_train_selected_configs.json",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="data/benchmarks/srp_policy_ladder_L10_report.json",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default="data/benchmarks/paper_figures_srp_policy_ladder_L10",
    )
    parser.add_argument("--sanity", action="store_true")
    return parser.parse_args()


def _sanity_adjusted_args(args: argparse.Namespace) -> argparse.Namespace:
    """Return a copy of the args with reduced maps for quick smoke validation."""
    if not args.sanity:
        return args
    clone = argparse.Namespace(**vars(args))
    clone.train_maps = ",".join(SANITY_TRAIN_MAPS)
    clone.eval_maps = ",".join(SANITY_EVAL_MAPS)
    clone.train_trials = 1
    clone.eval_trials = 1
    clone.workers = 1
    return clone


def build_base_experiments(
    interval_grid: list[int],
    tau_grid: list[float],
    nmax_grid: list[int],
) -> list[Experiment]:
    """Return the base policy ladder grid before lambda expansion."""
    experiments: list[Experiment] = []
    for interval in interval_grid:
        experiments.append(
            Experiment(
                f"fixed_interval_k{interval}",
                "fixed_interval",
                {"interval": interval},
            )
        )
    for tau in tau_grid:
        tau_label = str(tau).replace(".", "p")
        for nmax in nmax_grid:
            experiments.append(
                Experiment(
                    f"self_norm_tau{tau_label}_n{nmax}",
                    "self_normalizing_momentum",
                    {"tau": tau, "nmax": nmax, "seed": 7},
                )
            )
            experiments.append(
                Experiment(
                    f"srpv2_tau{tau_label}_n{nmax}",
                    "srpv2",
                    {"tau": tau, "nmax": nmax, "seed": 7},
                )
            )
    return experiments


def _row_to_info(row: pd.Series) -> dict[str, Any]:
    return {
        "experiment_name": str(row["experiment"]),
        "base_experiment": str(row["base_experiment"]),
        "strategy": str(row["strategy"]),
        "age_decay_lambda": float(row["age_decay_lambda"]),
        "params": json.loads(row["params_json"] or "{}"),
        "train_metrics": {
            "mean_collision_rate": float(row["mean_collision_rate"]),
            "mean_step_cap_rate": float(row["mean_step_cap_rate"]),
            "mean_crosstrack_rmse": float(row["mean_crosstrack_rmse"]),
            "mean_crosstrack_max": float(row["mean_crosstrack_max"]),
            "mean_cloud_call_rate": float(row["mean_cloud_call_rate"]),
        },
    }


def compute_pareto_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate a frame with a non-dominated frontier flag."""
    rows = frame.to_dict("records")
    pareto_names: set[str] = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                float(other["mean_cloud_call_rate"]) <= float(row["mean_cloud_call_rate"])
                and float(other["mean_crosstrack_max"]) <= float(row["mean_crosstrack_max"])
            )
            strictly_better = (
                float(other["mean_cloud_call_rate"]) < float(row["mean_cloud_call_rate"])
                or float(other["mean_crosstrack_max"]) < float(row["mean_crosstrack_max"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_names.add(str(row["experiment"]))
    annotated = frame.copy()
    annotated["pareto_frontier"] = annotated["experiment"].isin(pareto_names)
    return annotated


def select_interval_controls(cross_map_df: pd.DataFrame, interval_grid: list[int]) -> list[dict[str, Any]]:
    """Select the best tuned interval control for each requested interval."""
    controls: list[dict[str, Any]] = []
    for interval in interval_grid:
        base_name = f"fixed_interval_k{interval}"
        candidates = cross_map_df[cross_map_df["base_experiment"] == base_name].copy()
        if candidates.empty:
            continue
        controls.append(_row_to_info(candidates.iloc[0]))
    return controls


def select_family_shortlist(
    cross_map_df: pd.DataFrame,
    strategy: str,
    min_shortlist: int,
    max_shortlist: int,
) -> list[dict[str, Any]]:
    """Select a Pareto-led shortlist for one family."""
    family = cross_map_df[cross_map_df["strategy"] == strategy].copy()
    if family.empty:
        return []
    family = compute_pareto_flags(family).sort_values(
        [
            "pareto_frontier",
            "mean_collision_rate",
            "mean_step_cap_rate",
            "mean_crosstrack_max",
            "mean_crosstrack_rmse",
            "mean_cloud_call_rate",
            "experiment",
        ],
        ascending=[False, True, True, True, True, True, True],
        kind="stable",
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frontier_only in (True, False):
        for row in family.itertuples(index=False):
            if frontier_only and not bool(row.pareto_frontier):
                continue
            if row.base_experiment in seen:
                continue
            seen.add(str(row.base_experiment))
            selected.append(
                {
                    "experiment_name": str(row.experiment),
                    "base_experiment": str(row.base_experiment),
                    "strategy": str(row.strategy),
                    "age_decay_lambda": float(row.age_decay_lambda),
                    "params": json.loads(row.params_json or "{}"),
                    "pareto_frontier": bool(row.pareto_frontier),
                    "train_metrics": {
                        "mean_collision_rate": float(row.mean_collision_rate),
                        "mean_step_cap_rate": float(row.mean_step_cap_rate),
                        "mean_crosstrack_rmse": float(row.mean_crosstrack_rmse),
                        "mean_crosstrack_max": float(row.mean_crosstrack_max),
                        "mean_cloud_call_rate": float(row.mean_cloud_call_rate),
                    },
                }
            )
            if len(selected) >= max_shortlist:
                return selected
        if len(selected) >= min_shortlist:
            break
    return selected


def _experiment_from_info(info: dict[str, Any]) -> Experiment:
    return Experiment(
        name=str(info["experiment_name"]),
        strategy=str(info["strategy"]),
        params=dict(info["params"]),
    )


def build_eval_experiments(payload: dict[str, Any]) -> list[Experiment]:
    """Build the held-out eval set from the selected train-time controls."""
    experiments: list[Experiment] = [
        Experiment("always_hit__lambda_0p0", "always", {"age_decay_lambda": 0.0}),
        Experiment("never_query__lambda_0p0", "never_query", {"age_decay_lambda": 0.0}),
    ]
    experiments.extend(_experiment_from_info(info) for info in payload["interval_controls"])
    experiments.extend(_experiment_from_info(info) for info in payload["srp_shortlist"])
    experiments.extend(_experiment_from_info(info) for info in payload["srpv2_shortlist"])
    unique: list[Experiment] = []
    seen: set[str] = set()
    for exp in experiments:
        if exp.name in seen:
            continue
        seen.add(exp.name)
        unique.append(exp)
    return unique


def add_display_columns(cross_map_df: pd.DataFrame) -> pd.DataFrame:
    """Add compact labels and centimeter columns."""
    display_strategy = {
        "always": "Always query",
        "never_query": "Never query",
        "fixed_interval": "Fixed Interval",
        "self_normalizing_momentum": "SRP (Ours)",
        "srpv2": "SRPv2",
    }
    frame = cross_map_df.copy()
    frame["display_strategy"] = frame["strategy"].map(display_strategy).fillna(frame["strategy"])
    frame["mean_crosstrack_max_cm"] = frame["mean_crosstrack_max"] * 100.0
    frame["mean_crosstrack_rmse_cm"] = frame["mean_crosstrack_rmse"] * 100.0
    return frame


def build_budget_matches(cross_map_df: pd.DataFrame) -> pd.DataFrame:
    """Pair each SRP/SRPv2 point to the nearest interval control by CCR."""
    interval = cross_map_df[cross_map_df["strategy"] == "fixed_interval"].copy()
    targets = cross_map_df[
        cross_map_df["strategy"].isin(["self_normalizing_momentum", "srpv2"])
    ].copy()
    rows: list[dict[str, Any]] = []
    if interval.empty or targets.empty:
        return pd.DataFrame(rows)
    for row in targets.itertuples(index=False):
        distances = (interval["mean_cloud_call_rate"] - float(row.mean_cloud_call_rate)).abs()
        match = interval.iloc[int(distances.argmin())]
        interval_k = json.loads(match["params_json"] or "{}").get("interval")
        rows.append(
            {
                "experiment": row.experiment,
                "strategy": row.strategy,
                "display_strategy": "SRP (Ours)"
                if row.strategy == "self_normalizing_momentum"
                else "SRPv2",
                "base_experiment": row.base_experiment,
                "mean_cloud_call_rate": float(row.mean_cloud_call_rate),
                "mean_crosstrack_max": float(row.mean_crosstrack_max),
                "mean_crosstrack_max_cm": float(row.mean_crosstrack_max) * 100.0,
                "mean_crosstrack_rmse": float(row.mean_crosstrack_rmse),
                "mean_crosstrack_rmse_cm": float(row.mean_crosstrack_rmse) * 100.0,
                "matched_interval_experiment": str(match["experiment"]),
                "matched_interval_base_experiment": str(match["base_experiment"]),
                "matched_interval_k": int(interval_k) if interval_k is not None else None,
                "matched_interval_cloud_call_rate": float(match["mean_cloud_call_rate"]),
                "matched_interval_crosstrack_max": float(match["mean_crosstrack_max"]),
                "matched_interval_crosstrack_max_cm": float(match["mean_crosstrack_max"]) * 100.0,
                "matched_interval_crosstrack_rmse": float(match["mean_crosstrack_rmse"]),
                "matched_interval_crosstrack_rmse_cm": float(match["mean_crosstrack_rmse"]) * 100.0,
                "delta_max_cte_vs_interval": (
                    float(row.mean_crosstrack_max) - float(match["mean_crosstrack_max"])
                ),
                "delta_max_cte_vs_interval_cm": (
                    float(row.mean_crosstrack_max) - float(match["mean_crosstrack_max"])
                )
                * 100.0,
                "delta_rmse_vs_interval": (
                    float(row.mean_crosstrack_rmse) - float(match["mean_crosstrack_rmse"])
                ),
                "delta_rmse_vs_interval_cm": (
                    float(row.mean_crosstrack_rmse) - float(match["mean_crosstrack_rmse"])
                )
                * 100.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "mean_cloud_call_rate", "mean_crosstrack_max", "experiment"],
        kind="stable",
    )


def _representative_rows(cross_map_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Pick one representative eval row per family for paired-delta and per-map plots."""
    frame = cross_map_df.copy()

    def pick(strategy: str) -> pd.Series:
        family = frame[frame["strategy"] == strategy].copy()
        if family.empty:
            raise ValueError(f"Missing strategy in eval frame: {strategy}")
        family = family.sort_values(
            ["acceptable", "mean_crosstrack_max", "mean_cloud_call_rate", "experiment"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        return family.iloc[0]

    return {
        "Always query": pick("always"),
        "Never query": pick("never_query"),
        "Fixed Interval": pick("fixed_interval"),
        "SRP (Ours)": pick("self_normalizing_momentum"),
        "SRPv2": pick("srpv2"),
    }


def _paired_delta_rows(
    experiments_df: pd.DataFrame,
    reference_experiment: str,
    target_experiment: str,
    map_names: list[str] | None,
) -> np.ndarray:
    subset = experiments_df.copy()
    if map_names is not None:
        subset = subset[subset["map_name"].isin(map_names)].copy()
    reference = subset[subset["experiment"] == reference_experiment][
        ["map_name", "run_idx", "crosstrack_max_m"]
    ].rename(columns={"crosstrack_max_m": "reference_max_cte"})
    target = subset[subset["experiment"] == target_experiment][
        ["map_name", "run_idx", "crosstrack_max_m"]
    ].rename(columns={"crosstrack_max_m": "target_max_cte"})
    merged = target.merge(reference, on=["map_name", "run_idx"], how="inner")
    if merged.empty:
        return np.array([], dtype=float)
    return merged["target_max_cte"].to_numpy(dtype=float) - merged["reference_max_cte"].to_numpy(
        dtype=float
    )


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_paired_delta_table(
    experiments_df: pd.DataFrame,
    cross_map_df: pd.DataFrame,
    bootstrap_samples: int,
) -> pd.DataFrame:
    """Compute paired deltas vs always-query for representative controls."""
    representatives = _representative_rows(cross_map_df)
    reference_experiment = str(representatives["Always query"]["experiment"])
    panels = [("Sochi", ["Sochi"]), ("Spa", ["Spa"]), ("Aggregate", None)]
    labels = ["Never query", "Fixed Interval", "SRP (Ours)", "SRPv2"]
    rows: list[dict[str, Any]] = []
    for panel_idx, (panel_name, map_names) in enumerate(panels):
        for label_idx, label in enumerate(labels):
            target_experiment = str(representatives[label]["experiment"])
            deltas = _paired_delta_rows(
                experiments_df,
                reference_experiment=reference_experiment,
                target_experiment=target_experiment,
                map_names=map_names,
            )
            ci_low, ci_high = _bootstrap_ci(
                deltas,
                samples=bootstrap_samples,
                seed=7 + panel_idx * 100 + label_idx,
            )
            rows.append(
                {
                    "panel": panel_name,
                    "label": label,
                    "experiment": target_experiment,
                    "mean_delta_max_cte_vs_always": float(deltas.mean()) if deltas.size else np.nan,
                    "mean_delta_max_cte_vs_always_cm": float(deltas.mean()) * 100.0
                    if deltas.size
                    else np.nan,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "ci95_low_cm": ci_low * 100.0 if np.isfinite(ci_low) else np.nan,
                    "ci95_high_cm": ci_high * 100.0 if np.isfinite(ci_high) else np.nan,
                    "paired_samples": int(deltas.size),
                }
            )
    return pd.DataFrame(rows)


def build_per_map_comparison(
    summary_df: pd.DataFrame,
    cross_map_df: pd.DataFrame,
    budget_matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one-point-per-map comparisons against the matched interval baseline."""
    representatives = _representative_rows(cross_map_df)
    rows: list[dict[str, Any]] = []
    for label in ("SRP (Ours)", "SRPv2"):
        target_row = representatives[label]
        target_experiment = str(target_row["experiment"])
        budget_row = budget_matches_df[budget_matches_df["experiment"] == target_experiment]
        if budget_row.empty:
            continue
        matched_interval_experiment = str(budget_row.iloc[0]["matched_interval_experiment"])
        target_map = summary_df[summary_df["experiment"] == target_experiment][
            ["map_name", "crosstrack_max_m_mean", "cloud_call_rate_mean"]
        ].rename(
            columns={
                "crosstrack_max_m_mean": "target_max_cte_m",
                "cloud_call_rate_mean": "target_cloud_call_rate",
            }
        )
        interval_map = summary_df[summary_df["experiment"] == matched_interval_experiment][
            ["map_name", "crosstrack_max_m_mean", "cloud_call_rate_mean"]
        ].rename(
            columns={
                "crosstrack_max_m_mean": "interval_max_cte_m",
                "cloud_call_rate_mean": "interval_cloud_call_rate",
            }
        )
        merged = target_map.merge(interval_map, on="map_name", how="inner")
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "family": label,
                    "target_experiment": target_experiment,
                    "matched_interval_experiment": matched_interval_experiment,
                    "map_name": row.map_name,
                    "target_max_cte_m": float(row.target_max_cte_m),
                    "target_max_cte_cm": float(row.target_max_cte_m) * 100.0,
                    "interval_max_cte_m": float(row.interval_max_cte_m),
                    "interval_max_cte_cm": float(row.interval_max_cte_m) * 100.0,
                    "delta_max_cte_cm": (float(row.target_max_cte_m) - float(row.interval_max_cte_m))
                    * 100.0,
                    "target_cloud_call_rate": float(row.target_cloud_call_rate),
                    "interval_cloud_call_rate": float(row.interval_cloud_call_rate),
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "map_name"], kind="stable")


def build_compact_table(cross_map_df: pd.DataFrame, interval_grid: list[int]) -> pd.DataFrame:
    """Build the headline table rows for the study report."""
    frame = cross_map_df.copy()
    representatives = _representative_rows(frame)
    rows: list[dict[str, Any]] = []
    for label in ("Always query", "Never query", "SRP (Ours)", "SRPv2"):
        row = representatives[label]
        rows.append(
            {
                "Method": label,
                "Experiment": str(row["experiment"]),
                "CCR (%)": float(row["mean_cloud_call_rate"]) * 100.0,
                "Max CTE (cm)": float(row["mean_crosstrack_max"]) * 100.0,
                "RMSE (cm)": float(row["mean_crosstrack_rmse"]) * 100.0,
                "Collision rate": float(row["mean_collision_rate"]),
                "Acceptable": bool(row["acceptable"]),
            }
        )
    for interval in (2, 4, 10, 15):
        if interval not in interval_grid:
            continue
        base_name = f"fixed_interval_k{interval}"
        row = frame[frame["base_experiment"] == base_name].iloc[0]
        rows.append(
            {
                "Method": f"Fixed interval (k={interval})",
                "Experiment": str(row["experiment"]),
                "CCR (%)": float(row["mean_cloud_call_rate"]) * 100.0,
                "Max CTE (cm)": float(row["mean_crosstrack_max"]) * 100.0,
                "RMSE (cm)": float(row["mean_crosstrack_rmse"]) * 100.0,
                "Collision rate": float(row["mean_collision_rate"]),
                "Acceptable": bool(row["acceptable"]),
            }
        )
    return pd.DataFrame(rows)


def write_bundle(
    stem: str,
    experiments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cross_map_df: pd.DataFrame,
    budget_matches_df: pd.DataFrame,
    paired_delta_df: pd.DataFrame,
    per_map_df: pd.DataFrame,
    headline_df: pd.DataFrame | None = None,
) -> None:
    """Write a standard artifact bundle under data/benchmarks."""
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments_df.to_csv(out_dir / f"{stem}.csv", index=False)
    summary_df.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    cross_map_df.to_csv(out_dir / f"{stem}_cross_map.csv", index=False)
    budget_matches_df.to_csv(out_dir / f"{stem}_budget_matches.csv", index=False)
    paired_delta_df.to_csv(out_dir / f"{stem}_paired_delta.csv", index=False)
    per_map_df.to_csv(out_dir / f"{stem}_per_map.csv", index=False)
    if headline_df is not None:
        headline_df.to_csv(out_dir / f"{stem}_headline_table.csv", index=False)
    sheets = {
        "experiments": experiments_df,
        "summary": summary_df,
        "cross_map": cross_map_df,
        "budget_matches": budget_matches_df,
        "paired_delta": paired_delta_df,
        "per_map": per_map_df,
    }
    if headline_df is not None:
        sheets["headline_table"] = headline_df
    maybe_write_xlsx(sheets, out_dir / f"{stem}.xlsx")


def run_train_phase(args: argparse.Namespace) -> dict[str, Any]:
    """Run the train-time sweep and write train artifacts plus selected config JSON."""
    train_maps = _parse_str_list(args.train_maps)
    interval_grid = _parse_int_list(args.interval_grid)
    tau_grid = _parse_float_list(args.tau_grid)
    nmax_grid = _parse_int_list(args.nmax_grid)
    lambda_grid = _parse_float_list(args.lambda_grid)

    base_experiments = build_base_experiments(interval_grid, tau_grid, nmax_grid)
    train_experiments = expand_lambda_grid(base_experiments, lambda_grid)
    print(
        f"[srp_policy_ladder] train phase: {len(train_maps)} maps, "
        f"{len(train_experiments)} experiments, {args.train_trials} trial(s), "
        f"latency={args.cloud_latency}, workers={args.workers}",
        flush=True,
    )
    results = run_experiment_set(
        maps=train_maps,
        experiments=train_experiments,
        cloud_latency=args.cloud_latency,
        max_laps=args.max_laps,
        settings=SUPERVISOR_SETTINGS,
        trials=args.train_trials,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    experiments_df, summary_df, _ = summarize(results, selection_metric="max_cte")
    cross_map_df = rank_cross_map(aggregate_across_maps(experiments_df, summary_df)).reset_index(
        drop=True
    )

    interval_controls = select_interval_controls(cross_map_df, interval_grid)
    srp_shortlist = select_family_shortlist(
        cross_map_df,
        strategy="self_normalizing_momentum",
        min_shortlist=args.min_shortlist,
        max_shortlist=args.max_shortlist,
    )
    srpv2_shortlist = select_family_shortlist(
        cross_map_df,
        strategy="srpv2",
        min_shortlist=args.min_shortlist,
        max_shortlist=args.max_shortlist,
    )
    baseline_config = next(
        (
            control
            for control in interval_controls
            if control["base_experiment"] == "fixed_interval_k2"
        ),
        interval_controls[0] if interval_controls else None,
    )
    if baseline_config is None:
        raise ValueError("No interval controls were selected in the train phase.")
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cloud_latency": args.cloud_latency,
        "train_maps": train_maps,
        "settings": SUPERVISOR_SETTINGS.__dict__,
        "grids": {
            "interval_grid": interval_grid,
            "tau_grid": tau_grid,
            "nmax_grid": nmax_grid,
            "lambda_grid": lambda_grid,
        },
        "baseline_config": baseline_config,
        "interval_controls": interval_controls,
        "srp_shortlist": srp_shortlist,
        "srpv2_shortlist": srpv2_shortlist,
        "plot_output_dir": args.plot_output_dir,
    }
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments_df.to_csv(out_dir / f"{args.train_output_stem}.csv", index=False)
    summary_df.to_csv(out_dir / f"{args.train_output_stem}_summary.csv", index=False)
    cross_map_df.to_csv(out_dir / f"{args.train_output_stem}_cross_map.csv", index=False)
    maybe_write_xlsx(
        {"experiments": experiments_df, "summary": summary_df, "cross_map": cross_map_df},
        out_dir / f"{args.train_output_stem}.xlsx",
    )
    Path(args.selected_configs_json).write_text(json.dumps(payload, indent=2, default=_json_default))
    print(
        f"[srp_policy_ladder] train artifacts written: "
        f"data/benchmarks/{args.train_output_stem}_cross_map.csv and {args.selected_configs_json}",
        flush=True,
    )
    return payload


def run_eval_phase(args: argparse.Namespace, selected_payload: dict[str, Any]) -> dict[str, Any]:
    """Run the held-out eval rerun and write the eval artifact bundle plus report."""
    eval_maps = _parse_str_list(args.eval_maps)
    interval_grid = _parse_int_list(args.interval_grid)
    eval_experiments = build_eval_experiments(selected_payload)
    print(
        f"[srp_policy_ladder] eval phase: {len(eval_maps)} maps, "
        f"{len(eval_experiments)} experiments, {args.eval_trials} trial(s), "
        f"latency={args.cloud_latency}, workers={args.workers}",
        flush=True,
    )
    results = run_experiment_set(
        maps=eval_maps,
        experiments=eval_experiments,
        cloud_latency=args.cloud_latency,
        max_laps=args.max_laps,
        settings=SUPERVISOR_SETTINGS,
        trials=args.eval_trials,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    experiments_df, summary_df, _ = summarize(results, selection_metric="max_cte")
    cross_map_df = aggregate_across_maps(experiments_df, summary_df)
    baseline_experiment = str(selected_payload["baseline_config"]["experiment_name"])
    cross_map_df, baseline_row = acceptable_by_k2(
        cross_map_df,
        baseline_experiment_name=baseline_experiment,
    )
    cross_map_df = add_display_columns(compute_pareto_flags(cross_map_df))
    budget_matches_df = build_budget_matches(cross_map_df)
    paired_delta_df = build_paired_delta_table(
        experiments_df=experiments_df,
        cross_map_df=cross_map_df,
        bootstrap_samples=args.bootstrap_samples,
    )
    per_map_df = build_per_map_comparison(summary_df, cross_map_df, budget_matches_df)
    headline_df = build_compact_table(cross_map_df, interval_grid=interval_grid)
    write_bundle(
        stem=args.eval_output_stem,
        experiments_df=experiments_df,
        summary_df=summary_df,
        cross_map_df=cross_map_df,
        budget_matches_df=budget_matches_df,
        paired_delta_df=paired_delta_df,
        per_map_df=per_map_df,
        headline_df=headline_df,
    )
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cloud_latency": args.cloud_latency,
        "train_maps": selected_payload["train_maps"],
        "eval_maps": eval_maps,
        "settings": SUPERVISOR_SETTINGS.__dict__,
        "grids": selected_payload["grids"],
        "baseline_config": selected_payload["baseline_config"],
        "baseline_mean_crosstrack_max": float(baseline_row["mean_crosstrack_max"]),
        "baseline_mean_crosstrack_rmse": float(baseline_row["mean_crosstrack_rmse"]),
        "baseline_mean_cloud_call_rate": float(baseline_row["mean_cloud_call_rate"]),
        "selected_representatives": {
            label: {
                "experiment": str(row["experiment"]),
                "strategy": str(row["strategy"]),
                "mean_crosstrack_max_cm": float(row["mean_crosstrack_max"]) * 100.0,
                "mean_crosstrack_rmse_cm": float(row["mean_crosstrack_rmse"]) * 100.0,
                "mean_cloud_call_rate": float(row["mean_cloud_call_rate"]),
                "acceptable": bool(row["acceptable"]),
            }
            for label, row in _representative_rows(cross_map_df).items()
        },
        "interval_controls": selected_payload["interval_controls"],
        "srp_shortlist": selected_payload["srp_shortlist"],
        "srpv2_shortlist": selected_payload["srpv2_shortlist"],
        "pareto_experiments": cross_map_df[cross_map_df["pareto_frontier"]][
            ["experiment", "strategy", "mean_cloud_call_rate", "mean_crosstrack_max"]
        ].to_dict(orient="records"),
        "headline_table": headline_df.to_dict(orient="records"),
        "budget_matches": budget_matches_df.to_dict(orient="records"),
        "paired_delta": paired_delta_df.to_dict(orient="records"),
        "per_map_comparison": per_map_df.to_dict(orient="records"),
        "plot_paths": {
            "output_dir": args.plot_output_dir,
            "max_cte_vs_cloud_call_rate": f"{args.plot_output_dir}/max_cte_vs_cloud_call_rate.pdf",
            "pareto_frontier_max_cte": f"{args.plot_output_dir}/pareto_frontier_max_cte.pdf",
            "budget_matched_vs_interval": f"{args.plot_output_dir}/budget_matched_vs_interval.pdf",
            "interval_ladder_only": f"{args.plot_output_dir}/interval_ladder_only.pdf",
            "per_map_srpx_vs_interval": f"{args.plot_output_dir}/per_map_srpx_vs_interval.pdf",
            "paired_delta_vs_always": f"{args.plot_output_dir}/paired_delta_vs_always.pdf",
        },
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2, default=_json_default))
    print(
        f"[srp_policy_ladder] eval artifacts written: "
        f"data/benchmarks/{args.eval_output_stem}_cross_map.csv and {args.report_json}",
        flush=True,
    )
    return report


def main() -> None:
    """Run the requested phase of the focused policy ladder study."""
    args = _sanity_adjusted_args(parse_args())
    selected_payload: dict[str, Any] | None = None
    if args.phase in {"train", "full"}:
        selected_payload = run_train_phase(args)
    if args.phase in {"eval", "full"}:
        if selected_payload is None:
            selected_payload = json.loads(Path(args.selected_configs_json).read_text())
        run_eval_phase(args, selected_payload)


if __name__ == "__main__":
    main()
