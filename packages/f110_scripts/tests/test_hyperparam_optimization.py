"""Tests for the hyperparameter optimization and eval benchmark scripts."""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


_BENCHMARKS_DIR: str | None = None


def _load_module(script_name: str):
    """Load a benchmark script as a module, with sibling imports on sys.path."""
    global _BENCHMARKS_DIR  # noqa: PLW0603
    root = Path(__file__).resolve().parents[3]
    benchmarks_dir = str(root / "scripts" / "benchmarks")
    if _BENCHMARKS_DIR is None:
        _BENCHMARKS_DIR = benchmarks_dir
        if benchmarks_dir not in sys.path:
            sys.path.insert(0, benchmarks_dir)
    script_path = root / "scripts" / "benchmarks" / script_name
    module_name = script_name.replace(".py", "")
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_map_split():
    return _load_module("map_split.py")


def _load_optimize():
    return _load_module("optimize_hyperparameters.py")


def _load_eval():
    return _load_module("benchmark_eval_best_configs.py")


def _load_lambda_sweep():
    return _load_module("sweep_age_decay_lambda.py")


# ---- Map split tests ----


def test_train_eval_maps_disjoint() -> None:
    """Train and eval map sets must not overlap."""
    mod = _load_map_split()
    assert set(mod.TRAIN_MAPS) & set(mod.NON_TRAIN_EVAL_MAPS) == set()


def test_all_maps_accounted_for() -> None:
    """All 23 F1 maps should be covered by the train/eval split."""
    mod = _load_map_split()
    root = Path(__file__).resolve().parents[3]
    on_disk = sorted(
        p.name for p in (root / "data" / "maps" / "F1").iterdir() if p.is_dir()
    )
    assert sorted(mod.TRAIN_MAPS + mod.NON_TRAIN_EVAL_MAPS) == on_disk
    assert mod.ALL_MAPS == on_disk


def test_non_train_eval_maps_match_revised_labels() -> None:
    """Non-train maps should match the revised label set."""
    mod = _load_map_split()
    assert len(mod.NON_TRAIN_EVAL_MAPS) == 7
    assert "Spa" in mod.NON_TRAIN_EVAL_MAPS
    assert "Sochi" in mod.NON_TRAIN_EVAL_MAPS


def test_train_maps_match_revised_labels() -> None:
    """Train maps should match the revised label set."""
    mod = _load_map_split()
    assert len(mod.TRAIN_MAPS) == 16
    assert "Austin" in mod.TRAIN_MAPS
    assert "Budapest" in mod.TRAIN_MAPS
    assert "Zandvoort" in mod.TRAIN_MAPS


def test_sanity_subsets_are_disjoint_and_drawn_from_full_split() -> None:
    """Sanity subsets should be valid subsets of the full revised split."""
    mod = _load_map_split()
    assert set(mod.SANITY_TRAIN_MAPS).issubset(set(mod.TRAIN_MAPS))
    assert set(mod.SANITY_EVAL_MAPS).issubset(set(mod.NON_TRAIN_EVAL_MAPS))
    assert set(mod.SANITY_TRAIN_MAPS).isdisjoint(set(mod.SANITY_EVAL_MAPS))


# ---- Optimization grid tests ----


def test_optimization_grid_covers_all_strategies() -> None:
    """The expanded grid should include all 8 strategy families."""
    mod = _load_optimize()
    strategies = {exp.strategy for exp in mod.optimization_grid()}
    assert strategies == {
        "always",
        "fixed_interval",
        "fixed_bernoulli",
        "bernoulli_max_miss",
        "logistic",
        "exponential",
        "piecewise_ramp",
        "deterministic",
    }


def test_optimization_grid_has_more_configs_than_default() -> None:
    """The expanded grid should be larger than the default paper-strategy grid."""
    opt_mod = _load_optimize()
    bench_mod = _load_module("benchmark_single_tier_paper_strategies.py")
    assert len(opt_mod.optimization_grid()) > len(bench_mod.default_experiments())


def test_sanity_optimization_grid_has_expected_size() -> None:
    """The sanity grid should contain the compact 32-config search space."""
    mod = _load_optimize()
    assert len(mod.sanity_optimization_grid()) == 32


def test_optimization_grid_unique_names() -> None:
    """Every experiment in the grid should have a unique name."""
    mod = _load_optimize()
    names = [exp.name for exp in mod.optimization_grid()]
    assert len(names) == len(set(names))


def test_optimize_parser_accepts_age_decay_lambda(monkeypatch: pytest.MonkeyPatch) -> None:
    """The training optimizer should expose the global age-decay lambda."""
    mod = _load_optimize()
    monkeypatch.setattr(
        sys,
        "argv",
        ["optimize_hyperparameters.py", "--age-decay-lambda", "2.5"],
    )

    args = mod.parse_args()

    assert args.age_decay_lambda == 2.5


def test_lambda_sweep_selects_lowest_max_cte_collision_free_candidate() -> None:
    """Lambda sweep ranking should prefer collision-feasible lambdas then max CTE."""
    mod = _load_lambda_sweep()
    summary_df = pd.DataFrame(
        [
            {
                "age_decay_lambda": 0.0,
                "mean_collision_rate": 0.1,
                "mean_collision_free_rate": 0.9,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.04,
                "mean_crosstrack_max": 0.20,
                "mean_cloud_call_rate": 0.60,
                "mean_abs_ccr_distance": 0.0,
                "frac_in_target_band": 1.0,
                "collision_feasible": False,
            },
            {
                "age_decay_lambda": 1.0,
                "mean_collision_rate": 0.0,
                "mean_collision_free_rate": 1.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.05,
                "mean_crosstrack_max": 0.15,
                "mean_cloud_call_rate": 0.60,
                "mean_abs_ccr_distance": 0.0,
                "frac_in_target_band": 1.0,
                "collision_feasible": True,
            },
            {
                "age_decay_lambda": 2.0,
                "mean_collision_rate": 0.0,
                "mean_collision_free_rate": 1.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.03,
                "mean_crosstrack_max": 0.12,
                "mean_cloud_call_rate": 0.63,
                "mean_abs_ccr_distance": 0.03,
                "frac_in_target_band": 1.0,
                "collision_feasible": True,
            },
        ]
    )

    winner = mod.select_best_lambda(summary_df, "max_cte")

    assert winner["age_decay_lambda"] == 2.0


# ---- Cross-map aggregation tests ----


def test_aggregate_across_maps_one_row_per_experiment() -> None:
    """Cross-map aggregation should produce one row per experiment."""
    mod = _load_optimize()

    # Build synthetic summary and experiments data
    experiments_data = []
    summary_rows = []
    for map_name in ["MapA", "MapB"]:
        for exp_name, strategy in [("exp1", "fixed_bernoulli"), ("exp2", "logistic")]:
            experiments_data.append({
                "experiment": exp_name,
                "strategy": strategy,
                "params_json": json.dumps({"p": 0.6}),
            })
            summary_rows.append({
                "experiment": exp_name,
                "strategy": strategy,
                "map_name": map_name,
                "collision_rate": 0.0,
                "step_cap_rate": 0.0,
                "crosstrack_rmse_m_mean": 0.05,
                "crosstrack_max_m_mean": 0.10,
                "cloud_call_rate_mean": 0.60,
                "in_target_ccr_band": True,
            })

    experiments_df = pd.DataFrame(experiments_data)
    summary_df = pd.DataFrame(summary_rows)
    result = mod.aggregate_across_maps(summary_df, experiments_df)

    assert len(result) == 2
    assert set(result["experiment"]) == {"exp1", "exp2"}


# ---- Best-config selection tests ----


def test_select_best_per_strategy_picks_in_band_winner() -> None:
    """Selection should prefer in-band configs and rank by RMSE."""
    mod = _load_optimize()

    cross_map_df = pd.DataFrame([
        {
            "experiment": "bernoulli_p50",
            "strategy": "fixed_bernoulli",
            "mean_collision_rate": 0.0,
            "mean_step_cap_rate": 0.0,
            "mean_crosstrack_rmse": 0.08,
            "mean_cloud_call_rate": 0.60,
            "in_target_ccr_band": True,
            "params_json": json.dumps({"p": 0.50, "seed": 7}),
        },
        {
            "experiment": "bernoulli_p65",
            "strategy": "fixed_bernoulli",
            "mean_collision_rate": 0.0,
            "mean_step_cap_rate": 0.0,
            "mean_crosstrack_rmse": 0.05,
            "mean_cloud_call_rate": 0.62,
            "in_target_ccr_band": True,
            "params_json": json.dumps({"p": 0.65, "seed": 7}),
        },
        {
            "experiment": "bernoulli_p70",
            "strategy": "fixed_bernoulli",
            "mean_collision_rate": 0.0,
            "mean_step_cap_rate": 0.0,
            "mean_crosstrack_rmse": 0.03,
            "mean_cloud_call_rate": 0.70,
            "in_target_ccr_band": False,
            "params_json": json.dumps({"p": 0.70, "seed": 7}),
        },
    ])

    cross_map_df["mean_crosstrack_max"] = [0.20, 0.12, 0.10]
    best = mod.select_best_per_strategy(cross_map_df, "max_cte")
    assert "fixed_bernoulli" in best
    # Should pick bernoulli_p65 (in-band, lowest RMSE among in-band)
    assert best["fixed_bernoulli"]["experiment_name"] == "bernoulli_p65"
    assert best["fixed_bernoulli"]["params"]["p"] == 0.65


def test_select_best_fallback_when_no_in_band() -> None:
    """Selection should fall back to nearest-to-band when none are in band."""
    mod = _load_optimize()

    cross_map_df = pd.DataFrame([
        {
            "experiment": "det_t01",
            "strategy": "deterministic",
            "mean_collision_rate": 0.0,
            "mean_step_cap_rate": 0.0,
            "mean_crosstrack_rmse": 0.04,
            "mean_cloud_call_rate": 0.80,
            "in_target_ccr_band": False,
            "params_json": json.dumps({"threshold": 0.01}),
        },
        {
            "experiment": "det_t03",
            "strategy": "deterministic",
            "mean_collision_rate": 0.0,
            "mean_step_cap_rate": 0.0,
            "mean_crosstrack_rmse": 0.06,
            "mean_cloud_call_rate": 0.50,
            "in_target_ccr_band": False,
            "params_json": json.dumps({"threshold": 0.03}),
        },
    ])

    cross_map_df["mean_crosstrack_max"] = [0.14, 0.15]
    best = mod.select_best_per_strategy(cross_map_df, "max_cte")
    assert "deterministic" in best
    # det_t03 (CCR=0.50) is closer to band midpoint (0.60) than det_t01 (CCR=0.80)
    assert best["deterministic"]["experiment_name"] == "det_t03"


# ---- Config round-trip tests ----


def test_load_best_configs_roundtrip(tmp_path: Path) -> None:
    """Writing and loading best_configs.json should reconstruct Experiments."""
    eval_mod = _load_eval()

    configs = {
        "always": {
            "experiment_name": "always_hit",
            "strategy": "always",
            "params": {},
            "train_metrics": {"mean_collision_rate": 0.0},
        },
        "fixed_bernoulli": {
            "experiment_name": "fixed_bernoulli_p60",
            "strategy": "fixed_bernoulli",
            "params": {"p": 0.60, "seed": 7},
            "train_metrics": {"mean_collision_rate": 0.0},
        },
    }
    payload = {
        "generated_at": "2026-03-22 12:00:00",
        "train_maps": ["Budapest"],
        "cloud_latency": 5,
        "settings": {},
        "target_ccr_band": [0.55, 0.65],
        "best_configs": configs,
    }
    json_path = tmp_path / "best_configs.json"
    json_path.write_text(json.dumps(payload))

    experiments = eval_mod.load_best_configs(json_path)
    assert len(experiments) == 2
    names = {exp.name for exp in experiments}
    assert names == {"always_hit", "fixed_bernoulli_p60"}

    bernoulli = [e for e in experiments if e.strategy == "fixed_bernoulli"][0]
    assert bernoulli.params["p"] == 0.60


# ---- CLI defaults tests ----


def test_optimize_defaults_use_train_maps(monkeypatch) -> None:
    """Optimize script CLI defaults should reference TRAIN_MAPS."""
    mod = _load_optimize()
    map_split = _load_map_split()
    monkeypatch.setattr(sys, "argv", ["optimize_hyperparameters.py"])

    args = mod.parse_args()
    parsed_maps = [m.strip() for m in args.maps.split(",")]
    assert parsed_maps == map_split.TRAIN_MAPS


def test_eval_defaults_use_eval_maps(monkeypatch) -> None:
    """Eval script CLI defaults should reference EVAL_MAPS."""
    mod = _load_eval()
    map_split = _load_map_split()
    monkeypatch.setattr(sys, "argv", ["benchmark_eval_best_configs.py"])

    args = mod.parse_args()
    parsed_maps = [m.strip() for m in args.maps.split(",")]
    assert parsed_maps == map_split.EVAL_MAPS
    assert args.output_stem == "eval_best_configs_10maps"


# ---- Refinement grid tests ----


def test_build_refinement_grid_interior() -> None:
    """Interior winner should produce points between its neighbors."""
    mod = _load_lambda_sweep()
    grid, note = mod.build_refinement_grid([0, 1, 4, 16, 64], 4, refine_points=5)
    assert note == "none"
    assert len(grid) == 5
    assert all(1 < v < 16 for v in grid)
    # No duplicates with coarse grid.
    assert all(abs(v - 4) > 1e-9 for v in grid)


def test_build_refinement_grid_lower_boundary() -> None:
    """Winner at lower boundary should refine between 0 and right neighbor."""
    mod = _load_lambda_sweep()
    grid, note = mod.build_refinement_grid([0, 1, 4, 16, 64], 0, refine_points=5)
    assert note == "lower"
    assert len(grid) == 5
    assert all(0 < v < 1 for v in grid)


def test_build_refinement_grid_upper_boundary() -> None:
    """Winner at upper boundary should extend past the last grid point."""
    mod = _load_lambda_sweep()
    grid, note = mod.build_refinement_grid([0, 1, 4, 16, 64], 64, refine_points=5)
    assert note == "upper"
    assert len(grid) > 0
    # Should extend beyond 64.
    assert any(v > 64 for v in grid)
    # Should not duplicate 64.
    assert all(abs(v - 64) > 1e-9 for v in grid)


def test_build_refinement_grid_single_element() -> None:
    """Single-element grid cannot be refined."""
    mod = _load_lambda_sweep()
    grid, note = mod.build_refinement_grid([4], 4)
    assert grid == []
    assert note == "none"
