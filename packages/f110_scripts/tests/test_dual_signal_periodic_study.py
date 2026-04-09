"""Tests for the focused dual-signal periodic study and plotter."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd
import pytest


_BENCHMARKS_DIR: str | None = None


def _load_module(script_name: str):
    """Load a benchmark script as a module."""
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


def test_dual_signal_family_grid_matches_focused_search_surface() -> None:
    """The focused search should cover the planned dual-signal grid only."""
    module = _load_module("study_dual_signal_periodic.py")

    experiments = module.family_grid()

    assert len(experiments) == 108
    assert {exp.strategy for exp in experiments} == {"dual_signal_periodic"}
    assert {
        exp.params["base_interval"] for exp in experiments
    } == {3, 4, 5, 6}


def test_load_prior_control_experiments_deduplicates_and_validates_controls() -> None:
    """Prior winners plus baseline should round-trip into a unique rerun set."""
    module = _load_module("study_dual_signal_periodic.py")
    payload = {
        "best_configs": {
            "fixed_interval": {
                "experiment_name": "fixed_interval_k3__lambda_16p0",
                "strategy": "fixed_interval",
                "params": {"interval": 3, "age_decay_lambda": 16.0},
            },
            "fixed_bernoulli": {
                "experiment_name": "fixed_bernoulli_p40__lambda_16p0",
                "strategy": "fixed_bernoulli",
                "params": {"p": 0.4, "seed": 7, "age_decay_lambda": 16.0},
            },
            "bernoulli_max_miss": {
                "experiment_name": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "strategy": "bernoulli_max_miss",
                "params": {"p": 0.15, "max_miss": 2, "seed": 7, "age_decay_lambda": 20.666667},
            },
            "deterministic": {
                "experiment_name": "deterministic_t0p1__lambda_90p666667",
                "strategy": "deterministic",
                "params": {"threshold": 0.1, "age_decay_lambda": 90.666667},
            },
            "always": {
                "experiment_name": "always_hit__lambda_0p0",
                "strategy": "always",
                "params": {"age_decay_lambda": 0.0},
            },
            "self_normalizing_momentum": {
                "experiment_name": "self_norm_tau1p0_n3__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "params": {"tau": 1.0, "staleness_multiplier": 3, "seed": 7, "age_decay_lambda": 4.0},
            },
            "srpv2": {
                "experiment_name": "srpv2_tau0p75_n2__lambda_4p0",
                "strategy": "srpv2",
                "params": {"tau": 0.75, "nmax": 2, "seed": 7, "age_decay_lambda": 4.0},
            },
        },
        "baseline_config": {
            "experiment_name": "fixed_interval_k2__lambda_32p0",
            "strategy": "fixed_interval",
            "params": {"interval": 2, "age_decay_lambda": 32.0},
        },
    }

    experiments = module.load_prior_control_experiments(payload)

    assert {exp.name for exp in experiments} == {
        "fixed_interval_k3__lambda_16p0",
        "fixed_bernoulli_p40__lambda_16p0",
        "bernoulli_max_miss_p15_m2__lambda_20p666667",
        "deterministic_t0p1__lambda_90p666667",
        "always_hit__lambda_0p0",
        "self_norm_tau1p0_n3__lambda_4p0",
        "srpv2_tau0p75_n2__lambda_4p0",
        "fixed_interval_k2__lambda_32p0",
    }


def test_build_comparison_frame_computes_required_delta_columns() -> None:
    """Comparison output should include deltas versus the required controls."""
    module = _load_module("study_dual_signal_periodic.py")
    prior_payload = {
        "best_configs": {
            "fixed_interval": {
                "experiment_name": "fixed_interval_k3__lambda_16p0",
                "base_experiment": "fixed_interval_k3",
            },
            "bernoulli_max_miss": {
                "experiment_name": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "base_experiment": "bernoulli_max_miss_p15_m2",
            },
            "self_normalizing_momentum": {
                "experiment_name": "self_norm_tau1p0_n3__lambda_4p0",
                "base_experiment": "self_norm_tau1p0_n3",
            },
            "fixed_bernoulli": {
                "experiment_name": "fixed_bernoulli_p40__lambda_16p0",
                "base_experiment": "fixed_bernoulli_p40",
            },
            "deterministic": {
                "experiment_name": "deterministic_t0p1__lambda_90p666667",
                "base_experiment": "deterministic_t0p1",
            },
            "always": {
                "experiment_name": "always_hit__lambda_0p0",
                "base_experiment": "always_hit",
            },
            "srpv2": {
                "experiment_name": "srpv2_tau0p75_n2__lambda_4p0",
                "base_experiment": "srpv2_tau0p75_n2",
            },
        },
        "baseline_config": {
            "experiment_name": "fixed_interval_k2__lambda_32p0",
            "base_experiment": "fixed_interval_k2",
        },
    }
    cross_map_df = pd.DataFrame(
        [
            {
                "experiment": "fixed_interval_k2__lambda_32p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k2",
                "acceptable": True,
                "mean_crosstrack_max": 0.4900,
                "mean_crosstrack_rmse": 0.0960,
                "mean_cloud_call_rate": 0.5000,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_lap_time_s": 118.0,
                "mean_wall_min_distance": 0.66,
            },
            {
                "experiment": "fixed_interval_k3__lambda_16p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k3",
                "acceptable": True,
                "mean_crosstrack_max": 0.4920,
                "mean_crosstrack_rmse": 0.0910,
                "mean_cloud_call_rate": 0.3333,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_lap_time_s": 117.0,
                "mean_wall_min_distance": 0.67,
            },
            {
                "experiment": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "strategy": "bernoulli_max_miss",
                "base_experiment": "bernoulli_max_miss_p15_m2",
                "acceptable": True,
                "mean_crosstrack_max": 0.4918,
                "mean_crosstrack_rmse": 0.0902,
                "mean_cloud_call_rate": 0.3876,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_lap_time_s": 116.8,
                "mean_wall_min_distance": 0.66,
            },
            {
                "experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p0_n3",
                "acceptable": False,
                "mean_crosstrack_max": 0.5098,
                "mean_crosstrack_rmse": 0.0927,
                "mean_cloud_call_rate": 0.1177,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_lap_time_s": 117.1,
                "mean_wall_min_distance": 0.65,
            },
            {
                "experiment": "dual_signal_k3_bt70_tau1p0_devheavy__lambda_16p0",
                "strategy": "dual_signal_periodic",
                "base_experiment": "dual_signal_k3_bt70_tau1p0_devheavy",
                "acceptable": True,
                "mean_crosstrack_max": 0.4915,
                "mean_crosstrack_rmse": 0.0905,
                "mean_cloud_call_rate": 0.2800,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_lap_time_s": 116.9,
                "mean_wall_min_distance": 0.67,
            },
        ]
    )
    prior_cross_map_df = pd.DataFrame(
        [
            {
                "experiment": "fixed_interval_k2__lambda_32p0",
                "mean_crosstrack_max": 0.4898,
                "mean_crosstrack_rmse": 0.0963,
                "mean_cloud_call_rate": 0.5000,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_lap_time_s": 118.2,
                "mean_wall_min_distance": 0.66,
            },
            {
                "experiment": "fixed_interval_k3__lambda_16p0",
                "mean_crosstrack_max": 0.4927,
                "mean_crosstrack_rmse": 0.0909,
                "mean_cloud_call_rate": 0.3333,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_lap_time_s": 116.6,
                "mean_wall_min_distance": 0.67,
            },
            {
                "experiment": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "mean_crosstrack_max": 0.4920,
                "mean_crosstrack_rmse": 0.0903,
                "mean_cloud_call_rate": 0.3876,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_lap_time_s": 116.8,
                "mean_wall_min_distance": 0.67,
            },
            {
                "experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "mean_crosstrack_max": 0.5098,
                "mean_crosstrack_rmse": 0.0927,
                "mean_cloud_call_rate": 0.1177,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_lap_time_s": 117.1,
                "mean_wall_min_distance": 0.65,
            },
        ]
    )

    comparison = module.build_comparison_frame(cross_map_df, prior_cross_map_df, prior_payload)
    dual_row = comparison[comparison["strategy"] == "dual_signal_periodic"].iloc[0]

    assert dual_row["delta_max_cte_vs_baseline_k2"] == pytest.approx(0.0015)
    assert dual_row["delta_ccr_vs_interval_best"] < 0.0
    assert dual_row["delta_rmse_vs_bernoulli_best"] > 0.0
    assert bool(dual_row["meets_primary_success"])


def test_plot_dual_signal_outputs_render_figures(tmp_path: Path) -> None:
    """The focused dual-signal plotter should render all requested figures."""
    module = _load_module("plot_dual_signal_periodic_study.py")
    cross_map_csv = tmp_path / "cross_map.csv"
    comparison_csv = tmp_path / "comparison.csv"
    summary_csv = tmp_path / "summary.csv"
    out_dir = tmp_path / "figs"

    pd.DataFrame(
        [
            {
                "experiment": "fixed_interval_k3__lambda_16p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k3",
                "acceptable": True,
                "mean_cloud_call_rate": 0.3333,
                "mean_crosstrack_max": 0.4920,
                "mean_crosstrack_rmse": 0.0910,
                "mean_scheduler_calls_bootstrap": 0.0,
                "mean_scheduler_calls_backbone": 0.0,
                "mean_scheduler_calls_burst": 0.0,
                "mean_scheduler_calls_force_age": 0.0,
                "k2_threshold": 0.4950,
            },
            {
                "experiment": "dual_signal_k3_bt70_tau1p0_devheavy__lambda_16p0",
                "strategy": "dual_signal_periodic",
                "base_experiment": "dual_signal_k3_bt70_tau1p0_devheavy",
                "acceptable": True,
                "mean_cloud_call_rate": 0.2800,
                "mean_crosstrack_max": 0.4915,
                "mean_crosstrack_rmse": 0.0905,
                "mean_scheduler_calls_bootstrap": 1.0,
                "mean_scheduler_calls_backbone": 20.0,
                "mean_scheduler_calls_burst": 4.0,
                "mean_scheduler_calls_force_age": 1.0,
                "k2_threshold": 0.4950,
            },
        ]
    ).to_csv(cross_map_csv, index=False)
    pd.DataFrame(
        [
            {
                "experiment": "fixed_interval_k2__lambda_32p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k2",
                "acceptable": True,
                "mean_cloud_call_rate": 0.5000,
                "mean_crosstrack_max": 0.4900,
                "mean_crosstrack_rmse": 0.0960,
                "baseline_k2_experiment": "fixed_interval_k2__lambda_32p0",
                "baseline_k2_base_experiment": "fixed_interval_k2",
                "interval_best_experiment": "fixed_interval_k3__lambda_16p0",
                "interval_best_base_experiment": "fixed_interval_k3",
                "bernoulli_best_experiment": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "bernoulli_best_base_experiment": "bernoulli_max_miss_p15_m2",
                "srp_best_experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "srp_best_base_experiment": "self_norm_tau1p0_n3",
                "delta_ccr_vs_interval_best": 0.1667,
                "delta_max_cte_vs_interval_best": -0.0020,
                "stored_prev_mean_cloud_call_rate": 0.5000,
                "stored_prev_mean_crosstrack_max": 0.4898,
            },
            {
                "experiment": "fixed_interval_k3__lambda_16p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k3",
                "acceptable": True,
                "mean_cloud_call_rate": 0.3333,
                "mean_crosstrack_max": 0.4920,
                "mean_crosstrack_rmse": 0.0910,
                "baseline_k2_experiment": "fixed_interval_k2__lambda_32p0",
                "baseline_k2_base_experiment": "fixed_interval_k2",
                "interval_best_experiment": "fixed_interval_k3__lambda_16p0",
                "interval_best_base_experiment": "fixed_interval_k3",
                "bernoulli_best_experiment": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "bernoulli_best_base_experiment": "bernoulli_max_miss_p15_m2",
                "srp_best_experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "srp_best_base_experiment": "self_norm_tau1p0_n3",
                "delta_ccr_vs_interval_best": 0.0,
                "delta_max_cte_vs_interval_best": 0.0,
                "stored_prev_mean_cloud_call_rate": 0.3333,
                "stored_prev_mean_crosstrack_max": 0.4927,
            },
            {
                "experiment": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "strategy": "bernoulli_max_miss",
                "base_experiment": "bernoulli_max_miss_p15_m2",
                "acceptable": True,
                "mean_cloud_call_rate": 0.3876,
                "mean_crosstrack_max": 0.4918,
                "mean_crosstrack_rmse": 0.0902,
                "baseline_k2_experiment": "fixed_interval_k2__lambda_32p0",
                "baseline_k2_base_experiment": "fixed_interval_k2",
                "interval_best_experiment": "fixed_interval_k3__lambda_16p0",
                "interval_best_base_experiment": "fixed_interval_k3",
                "bernoulli_best_experiment": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "bernoulli_best_base_experiment": "bernoulli_max_miss_p15_m2",
                "srp_best_experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "srp_best_base_experiment": "self_norm_tau1p0_n3",
                "delta_ccr_vs_interval_best": 0.0543,
                "delta_max_cte_vs_interval_best": -0.0002,
                "stored_prev_mean_cloud_call_rate": 0.3876,
                "stored_prev_mean_crosstrack_max": 0.4920,
            },
            {
                "experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p0_n3",
                "acceptable": False,
                "mean_cloud_call_rate": 0.1177,
                "mean_crosstrack_max": 0.5098,
                "mean_crosstrack_rmse": 0.0927,
                "baseline_k2_experiment": "fixed_interval_k2__lambda_32p0",
                "baseline_k2_base_experiment": "fixed_interval_k2",
                "interval_best_experiment": "fixed_interval_k3__lambda_16p0",
                "interval_best_base_experiment": "fixed_interval_k3",
                "bernoulli_best_experiment": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "bernoulli_best_base_experiment": "bernoulli_max_miss_p15_m2",
                "srp_best_experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "srp_best_base_experiment": "self_norm_tau1p0_n3",
                "delta_ccr_vs_interval_best": -0.2156,
                "delta_max_cte_vs_interval_best": 0.0178,
                "stored_prev_mean_cloud_call_rate": 0.1177,
                "stored_prev_mean_crosstrack_max": 0.5098,
            },
            {
                "experiment": "dual_signal_k3_bt70_tau1p0_devheavy__lambda_16p0",
                "strategy": "dual_signal_periodic",
                "base_experiment": "dual_signal_k3_bt70_tau1p0_devheavy",
                "acceptable": True,
                "mean_cloud_call_rate": 0.2800,
                "mean_crosstrack_max": 0.4915,
                "mean_crosstrack_rmse": 0.0905,
                "baseline_k2_experiment": "fixed_interval_k2__lambda_32p0",
                "baseline_k2_base_experiment": "fixed_interval_k2",
                "interval_best_experiment": "fixed_interval_k3__lambda_16p0",
                "interval_best_base_experiment": "fixed_interval_k3",
                "bernoulli_best_experiment": "bernoulli_max_miss_p15_m2__lambda_20p666667",
                "bernoulli_best_base_experiment": "bernoulli_max_miss_p15_m2",
                "srp_best_experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "srp_best_base_experiment": "self_norm_tau1p0_n3",
                "delta_ccr_vs_interval_best": -0.0533,
                "delta_max_cte_vs_interval_best": -0.0005,
                "stored_prev_mean_cloud_call_rate": None,
                "stored_prev_mean_crosstrack_max": None,
            },
        ]
    ).to_csv(comparison_csv, index=False)
    pd.DataFrame(
        [
            {
                "experiment": "dual_signal_k3_bt70_tau1p0_devheavy__lambda_16p0",
                "map_name": "MexicoCity",
                "crosstrack_max_m_mean": 0.45,
            },
            {
                "experiment": "dual_signal_k3_bt70_tau1p0_devheavy__lambda_16p0",
                "map_name": "Monza",
                "crosstrack_max_m_mean": 0.53,
            },
            {
                "experiment": "fixed_interval_k3__lambda_16p0",
                "map_name": "MexicoCity",
                "crosstrack_max_m_mean": 0.46,
            },
            {
                "experiment": "fixed_interval_k3__lambda_16p0",
                "map_name": "Monza",
                "crosstrack_max_m_mean": 0.54,
            },
        ]
    ).to_csv(summary_csv, index=False)

    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_dual_signal_periodic_study.py",
            "--cross-map-csv",
            str(cross_map_csv),
            "--comparison-csv",
            str(comparison_csv),
            "--summary-csv",
            str(summary_csv),
            "--output-dir",
            str(out_dir),
            "--formats",
            "png",
        ]
        module.main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "tradeoff_overlay.png").exists()
    assert (out_dir / "direct_comparison.png").exists()
    assert (out_dir / "per_map_delta_heatmap.png").exists()
    assert (out_dir / "call_reason_stacked.png").exists()
