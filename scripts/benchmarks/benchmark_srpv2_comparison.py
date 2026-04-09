#!/usr/bin/env python3
"""Latency-10 held-out comparison for SRP, SRPv2, interval controls, and prior winners."""

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
    NON_TRAIN_EVAL_MAPS,
    SUPERVISOR_SETTINGS,
    acceptable_by_k2,
    aggregate_across_maps,
    run_experiment_set,
)


DEFAULT_CLOUD_LATENCY = 10
DEFAULT_BOOTSTRAP_SAMPLES = 5000


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the latency-10 SRPv2 comprehensive held-out comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cloud-latency", type=int, default=DEFAULT_CLOUD_LATENCY)
    parser.add_argument("--eval-trials", type=int, default=5)
    parser.add_argument("--max-laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--low-ccr-configs-json",
        type=str,
        default="data/benchmarks/low_ccr_per_method_lambda_L10_train_best_configs.json",
    )
    parser.add_argument(
        "--low-ccr-train-cross-map-csv",
        type=str,
        default="data/benchmarks/low_ccr_per_method_lambda_L10_train_cross_map.csv",
    )
    parser.add_argument(
        "--dual-configs-json",
        type=str,
        default="data/benchmarks/dual_signal_periodic_L10_train_best_configs.json",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="srpv2_comparison_L10_eval",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="data/benchmarks/srpv2_comparison_L10_report.json",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _experiment_from_info(info: dict[str, Any]) -> Experiment:
    name = str(info.get("experiment_name", info.get("experiment")))
    return Experiment(
        name=name,
        strategy=str(info["strategy"]),
        params=dict(info["params"]),
    )


def _top_k_unique_by_strategy(
    cross_map_df: pd.DataFrame,
    strategy: str,
    top_k: int,
) -> list[dict[str, Any]]:
    family = cross_map_df[cross_map_df["strategy"] == strategy].copy()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in family.itertuples(index=False):
        if row.base_experiment in seen:
            continue
        seen.add(str(row.base_experiment))
        selected.append(
            {
                "experiment_name": str(row.experiment),
                "base_experiment": str(row.base_experiment),
                "strategy": str(row.strategy),
                "params": json.loads(row.params_json or "{}"),
            }
        )
        if len(selected) >= top_k:
            break
    return selected


def _dedupe_experiments(experiments: list[Experiment]) -> list[Experiment]:
    unique: list[Experiment] = []
    seen: set[str] = set()
    for exp in experiments:
        if exp.name in seen:
            continue
        seen.add(exp.name)
        unique.append(exp)
    return unique


def build_eval_experiments(
    low_ccr_payload: dict[str, Any],
    low_ccr_train_cross_map_df: pd.DataFrame,
    dual_payload: dict[str, Any],
) -> list[Experiment]:
    """Assemble the held-out rerun set for the latency-10 comparison."""
    best_configs = low_ccr_payload["best_configs"]
    experiments: list[Experiment] = [
        _experiment_from_info(best_configs["always"]),
        Experiment(
            name="never_query__lambda_0p0",
            strategy="never_query",
            params={"age_decay_lambda": 0.0},
        ),
    ]
    experiments.extend(
        _experiment_from_info(info)
        for info in low_ccr_payload.get("fixed_interval_controls", [])
    )
    for family in ("fixed_bernoulli", "bernoulli_max_miss", "deterministic"):
        experiments.append(_experiment_from_info(best_configs[family]))
    experiments.extend(
        _experiment_from_info(info)
        for info in _top_k_unique_by_strategy(
            low_ccr_train_cross_map_df,
            strategy="self_normalizing_momentum",
            top_k=4,
        )
    )
    experiments.extend(
        _experiment_from_info(info)
        for info in _top_k_unique_by_strategy(
            low_ccr_train_cross_map_df,
            strategy="srpv2",
            top_k=4,
        )
    )
    if dual_payload.get("selected_dual_configs"):
        experiments.append(_experiment_from_info(dual_payload["selected_dual_configs"][0]))
    return _dedupe_experiments(experiments)


def add_display_columns(cross_map_df: pd.DataFrame) -> pd.DataFrame:
    """Add compact labels used by the report and focused plots."""
    display_strategy = {
        "always": "Always query",
        "never_query": "Never query",
        "fixed_interval": "Fixed Interval",
        "fixed_bernoulli": "Fixed Bernoulli",
        "bernoulli_max_miss": "Bernoulli Max-Miss",
        "deterministic": "Deterministic",
        "self_normalizing_momentum": "SRP (Ours)",
        "srpv2": "SRPv2",
        "dual_signal_periodic": "Dual-Signal",
    }
    frame = cross_map_df.copy()
    frame["display_strategy"] = frame["strategy"].map(display_strategy).fillna(frame["strategy"])
    frame["ccr_band"] = frame["mean_cloud_call_rate"].map(ccr_band_label)
    return frame


def ccr_band_label(rate: float) -> str:
    """Return the display band used by the budget-matched plot."""
    percent = float(rate) * 100.0
    for low, high in ((10, 20), (20, 30), (30, 40), (40, 50), (50, 60)):
        if low <= percent < high:
            return f"{low}-{high}% CCR"
    return "Out of Band"


def build_budget_matches(cross_map_df: pd.DataFrame) -> pd.DataFrame:
    """Pair each SRP/SRPv2 finalist to the nearest fixed-interval CCR point."""
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
                "ccr_band": ccr_band_label(float(row.mean_cloud_call_rate)),
                "matched_interval_experiment": str(match["experiment"]),
                "matched_interval_base_experiment": str(match["base_experiment"]),
                "matched_interval_cloud_call_rate": float(match["mean_cloud_call_rate"]),
                "matched_interval_crosstrack_max": float(match["mean_crosstrack_max"]),
                "delta_max_cte_vs_interval": (
                    float(row.mean_crosstrack_max) - float(match["mean_crosstrack_max"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "mean_cloud_call_rate", "mean_crosstrack_max", "experiment"],
        kind="stable",
    )


def _representative_experiment_names(
    low_ccr_payload: dict[str, Any],
) -> dict[str, str]:
    """Return the representative experiment used by the paired-delta plot."""
    best_configs = low_ccr_payload["best_configs"]
    representatives = {
        "Always query": str(best_configs["always"]["experiment_name"]),
        "Never query": "never_query__lambda_0p0",
        "Fixed Interval": str(best_configs["fixed_interval"]["experiment_name"]),
        "SRP (Ours)": str(best_configs["self_normalizing_momentum"]["experiment_name"]),
    }
    srpv2_info = best_configs.get("srpv2")
    if srpv2_info is not None:
        representatives["SRPv2"] = str(srpv2_info["experiment_name"])
    return representatives


def _paired_delta_rows(
    experiments_df: pd.DataFrame,
    reference_experiment: str,
    target_experiment: str,
    map_names: list[str] | None,
) -> np.ndarray:
    """Return aligned per-trial deltas for a target experiment vs the reference."""
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
    return (
        merged["target_max_cte"].to_numpy(dtype=float)
        - merged["reference_max_cte"].to_numpy(dtype=float)
    )


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    """Return a percentile bootstrap CI for the mean."""
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = draws.mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def build_paired_delta_table(
    experiments_df: pd.DataFrame,
    low_ccr_payload: dict[str, Any],
    bootstrap_samples: int,
) -> pd.DataFrame:
    """Compute paired max-CTE deltas vs always-query for the headline plot."""
    representatives = _representative_experiment_names(low_ccr_payload)
    reference_experiment = representatives["Always query"]
    panels = [
        ("Sochi", ["Sochi"]),
        ("Spa", ["Spa"]),
        ("Aggregate", None),
    ]
    labels = ["Never query", "Fixed Interval", "SRP (Ours)", "SRPv2"]
    rows: list[dict[str, Any]] = []
    for panel_idx, (panel_name, map_names) in enumerate(panels):
        for label_idx, label in enumerate(labels):
            target_experiment = representatives.get(label)
            if target_experiment is None:
                continue
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
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "paired_samples": int(deltas.size),
                }
            )
    return pd.DataFrame(rows)


def build_compact_table(
    cross_map_df: pd.DataFrame,
    low_ccr_payload: dict[str, Any],
    dual_payload: dict[str, Any],
) -> pd.DataFrame:
    """Render a compact, report-friendly table across all rerun control types."""
    representative_names = set(_representative_experiment_names(low_ccr_payload).values())
    if dual_payload.get("selected_dual_configs"):
        representative_names.add(
            str(
                dual_payload["selected_dual_configs"][0].get(
                    "experiment_name",
                    dual_payload["selected_dual_configs"][0]["experiment"],
                )
            )
        )
    table = add_display_columns(cross_map_df)
    table["role"] = "comparison"
    table.loc[table["experiment"].isin(representative_names), "role"] = "representative"
    table.loc[table["strategy"] == "fixed_interval", "role"] = "fixed_interval_ladder"
    table.loc[table["strategy"] == "self_normalizing_momentum", "role"] = "srp_finalist"
    table.loc[table["strategy"] == "srpv2", "role"] = "srpv2_finalist"
    table.loc[table["strategy"] == "dual_signal_periodic", "role"] = "dual_signal_winner"
    return table[
        [
            "experiment",
            "base_experiment",
            "strategy",
            "display_strategy",
            "role",
            "mean_collision_rate",
            "mean_step_cap_rate",
            "mean_crosstrack_rmse",
            "mean_crosstrack_max",
            "mean_cloud_call_rate",
            "mean_lap_time_s",
            "mean_wall_min_distance",
            "ccr_band",
            "acceptable",
        ]
    ].sort_values(
        ["display_strategy", "mean_cloud_call_rate", "mean_crosstrack_max", "experiment"],
        kind="stable",
    )


def write_outputs(
    stem: str,
    report_json: str,
    experiments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cross_map_df: pd.DataFrame,
    acceptable_df: pd.DataFrame,
    compact_table_df: pd.DataFrame,
    budget_matches_df: pd.DataFrame,
    paired_delta_df: pd.DataFrame,
    payload: dict[str, Any],
) -> None:
    """Write the full comparison artifact bundle."""
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    cross_map_path = out_dir / f"{stem}_cross_map.csv"
    acceptable_path = out_dir / f"{stem}_acceptable.csv"
    table_path = out_dir / f"{stem}_table.csv"
    budget_path = out_dir / f"{stem}_budget_matches.csv"
    paired_path = out_dir / f"{stem}_paired_delta.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"

    experiments_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    cross_map_df.to_csv(cross_map_path, index=False)
    acceptable_df.to_csv(acceptable_path, index=False)
    compact_table_df.to_csv(table_path, index=False)
    budget_matches_df.to_csv(budget_path, index=False)
    paired_delta_df.to_csv(paired_path, index=False)
    maybe_write_xlsx(
        {
            "experiments": experiments_df,
            "summary": summary_df,
            "cross_map": cross_map_df,
            "acceptable": acceptable_df,
            "table": compact_table_df,
            "budget_matches": budget_matches_df,
            "paired_delta": paired_delta_df,
        },
        xlsx_path,
    )
    Path(report_json).write_text(json.dumps(payload, indent=2, default=_json_default))


def main() -> None:
    """Run the latency-10 held-out comparison and write artifacts."""
    args = parse_args()
    low_ccr_payload = _load_json(args.low_ccr_configs_json)
    low_ccr_train_cross_map_df = pd.read_csv(args.low_ccr_train_cross_map_csv)
    dual_payload = _load_json(args.dual_configs_json)
    experiments = build_eval_experiments(
        low_ccr_payload=low_ccr_payload,
        low_ccr_train_cross_map_df=low_ccr_train_cross_map_df,
        dual_payload=dual_payload,
    )
    results = run_experiment_set(
        maps=list(NON_TRAIN_EVAL_MAPS),
        experiments=experiments,
        cloud_latency=args.cloud_latency,
        max_laps=args.max_laps,
        settings=SUPERVISOR_SETTINGS,
        trials=args.eval_trials,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    experiments_df, summary_df, _ = summarize(results, selection_metric="max_cte")
    cross_map_df = aggregate_across_maps(experiments_df, summary_df)
    cross_map_df, baseline_row = acceptable_by_k2(
        cross_map_df,
        baseline_experiment_name=str(low_ccr_payload["baseline_config"]["experiment_name"]),
    )
    cross_map_df = add_display_columns(cross_map_df)
    acceptable_df = cross_map_df[cross_map_df["acceptable"]].copy().sort_values(
        ["mean_cloud_call_rate", "mean_crosstrack_max", "experiment"],
        kind="stable",
    )
    compact_table_df = build_compact_table(cross_map_df, low_ccr_payload, dual_payload)
    budget_matches_df = build_budget_matches(cross_map_df)
    paired_delta_df = build_paired_delta_table(
        experiments_df=experiments_df,
        low_ccr_payload=low_ccr_payload,
        bootstrap_samples=args.bootstrap_samples,
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cloud_latency": args.cloud_latency,
        "eval_maps": list(NON_TRAIN_EVAL_MAPS),
        "settings": SUPERVISOR_SETTINGS.__dict__,
        "sources": {
            "low_ccr_configs_json": args.low_ccr_configs_json,
            "low_ccr_train_cross_map_csv": args.low_ccr_train_cross_map_csv,
            "dual_configs_json": args.dual_configs_json,
        },
        "baseline_experiment": str(low_ccr_payload["baseline_config"]["experiment_name"]),
        "baseline_mean_crosstrack_max": float(baseline_row["mean_crosstrack_max"]),
        "baseline_mean_crosstrack_rmse": float(baseline_row["mean_crosstrack_rmse"]),
        "baseline_mean_cloud_call_rate": float(baseline_row["mean_cloud_call_rate"]),
        "experiments": [exp.__dict__ for exp in experiments],
        "cross_map_summary": cross_map_df.to_dict(orient="records"),
        "acceptable_configs": acceptable_df.to_dict(orient="records"),
        "compact_table": compact_table_df.to_dict(orient="records"),
        "budget_matches": budget_matches_df.to_dict(orient="records"),
        "paired_delta": paired_delta_df.to_dict(orient="records"),
    }
    write_outputs(
        stem=args.output_stem,
        report_json=args.report_json,
        experiments_df=experiments_df,
        summary_df=summary_df,
        cross_map_df=cross_map_df,
        acceptable_df=acceptable_df,
        compact_table_df=compact_table_df,
        budget_matches_df=budget_matches_df,
        paired_delta_df=paired_delta_df,
        payload=payload,
    )
    print(f"Wrote comparison report to {args.report_json}")


if __name__ == "__main__":
    main()
