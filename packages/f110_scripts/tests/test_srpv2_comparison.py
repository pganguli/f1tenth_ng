"""Tests for the latency-10 SRPv2 comparison benchmark and plotter."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


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


def test_build_eval_experiments_assembles_expected_control_mix() -> None:
    """The focused rerun should include the intended winners, finalists, and controls."""
    module = _load_module("benchmark_srpv2_comparison.py")
    low_ccr_payload = {
        "best_configs": {
            "always": {
                "experiment_name": "always_hit__lambda_0p0",
                "strategy": "always",
                "params": {"age_decay_lambda": 0.0},
            },
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
            "self_normalizing_momentum": {
                "experiment_name": "self_norm_tau1p0_n3__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "params": {"tau": 1.0, "nmax": 3, "seed": 7, "age_decay_lambda": 4.0},
            },
            "srpv2": {
                "experiment_name": "srpv2_tau1p0_n3__lambda_4p0",
                "strategy": "srpv2",
                "params": {"tau": 1.0, "nmax": 3, "seed": 7, "age_decay_lambda": 4.0},
            },
        },
        "fixed_interval_controls": [
            {
                "experiment_name": "fixed_interval_k2__lambda_32p0",
                "strategy": "fixed_interval",
                "params": {"interval": 2, "age_decay_lambda": 32.0},
            },
            {
                "experiment_name": "fixed_interval_k5__lambda_16p0",
                "strategy": "fixed_interval",
                "params": {"interval": 5, "age_decay_lambda": 16.0},
            },
        ],
    }
    low_ccr_cross_map = pd.DataFrame(
        [
            {
                "experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p0_n3",
                "params_json": '{"tau": 1.0, "nmax": 3, "seed": 7, "age_decay_lambda": 4.0}',
            },
            {
                "experiment": "self_norm_tau0p75_n2__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau0p75_n2",
                "params_json": '{"tau": 0.75, "nmax": 2, "seed": 7, "age_decay_lambda": 4.0}',
            },
            {
                "experiment": "srpv2_tau1p0_n3__lambda_4p0",
                "strategy": "srpv2",
                "base_experiment": "srpv2_tau1p0_n3",
                "params_json": '{"tau": 1.0, "nmax": 3, "seed": 7, "age_decay_lambda": 4.0}',
            },
            {
                "experiment": "srpv2_tau0p75_n2__lambda_4p0",
                "strategy": "srpv2",
                "base_experiment": "srpv2_tau0p75_n2",
                "params_json": '{"tau": 0.75, "nmax": 2, "seed": 7, "age_decay_lambda": 4.0}',
            },
        ]
    )
    dual_payload = {
        "selected_dual_configs": [
            {
                "experiment_name": "dual_signal_k3_bt60_tau0p75_balanced__lambda_16p0",
                "strategy": "dual_signal_periodic",
                "params": {"base_interval": 3, "burst_threshold": 0.6, "tau": 0.75},
            }
        ]
    }

    experiments = module.build_eval_experiments(low_ccr_payload, low_ccr_cross_map, dual_payload)
    names = {exp.name for exp in experiments}

    assert "never_query__lambda_0p0" in names
    assert "self_norm_tau1p0_n3__lambda_4p0" in names
    assert "srpv2_tau1p0_n3__lambda_4p0" in names
    assert "fixed_interval_k5__lambda_16p0" in names
    assert "dual_signal_k3_bt60_tau0p75_balanced__lambda_16p0" in names


def test_plot_srpv2_outputs_render_figures(tmp_path: Path) -> None:
    """The focused SRPv2 plotter should render both headline figures."""
    module = _load_module("plot_srpv2_comparison.py")
    cross_map_csv = tmp_path / "cross_map.csv"
    budget_csv = tmp_path / "budget.csv"
    paired_csv = tmp_path / "paired.csv"
    out_dir = tmp_path / "figs"

    pd.DataFrame(
        [
            {
                "experiment": "fixed_interval_k3__lambda_16p0",
                "strategy": "fixed_interval",
                "mean_crosstrack_max": 0.3500,
                "mean_cloud_call_rate": 0.3333,
                "ccr_band": "30-40% CCR",
            },
            {
                "experiment": "fixed_interval_k5__lambda_16p0",
                "strategy": "fixed_interval",
                "mean_crosstrack_max": 0.3440,
                "mean_cloud_call_rate": 0.2000,
                "ccr_band": "20-30% CCR",
            },
        ]
    ).to_csv(cross_map_csv, index=False)
    pd.DataFrame(
        [
            {
                "experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "display_strategy": "SRP (Ours)",
                "strategy": "self_normalizing_momentum",
                "mean_crosstrack_max": 0.3450,
                "mean_cloud_call_rate": 0.2100,
                "ccr_band": "20-30% CCR",
                "delta_max_cte_vs_interval": 0.0010,
            },
            {
                "experiment": "srpv2_tau1p0_n3__lambda_4p0",
                "display_strategy": "SRPv2",
                "strategy": "srpv2",
                "mean_crosstrack_max": 0.3445,
                "mean_cloud_call_rate": 0.2100,
                "ccr_band": "20-30% CCR",
                "delta_max_cte_vs_interval": 0.0005,
            },
        ]
    ).to_csv(budget_csv, index=False)
    pd.DataFrame(
        [
            {
                "panel": "Sochi",
                "label": "Never query",
                "mean_delta_max_cte_vs_always": 0.02,
                "ci95_low": 0.01,
                "ci95_high": 0.03,
            },
            {
                "panel": "Sochi",
                "label": "Fixed Interval",
                "mean_delta_max_cte_vs_always": 0.0,
                "ci95_low": -0.01,
                "ci95_high": 0.01,
            },
            {
                "panel": "Sochi",
                "label": "SRP (Ours)",
                "mean_delta_max_cte_vs_always": -0.01,
                "ci95_low": -0.02,
                "ci95_high": 0.0,
            },
            {
                "panel": "Sochi",
                "label": "SRPv2",
                "mean_delta_max_cte_vs_always": -0.015,
                "ci95_low": -0.03,
                "ci95_high": -0.005,
            },
            {
                "panel": "Spa",
                "label": "Never query",
                "mean_delta_max_cte_vs_always": 0.03,
                "ci95_low": 0.02,
                "ci95_high": 0.04,
            },
            {
                "panel": "Spa",
                "label": "Fixed Interval",
                "mean_delta_max_cte_vs_always": -0.005,
                "ci95_low": -0.015,
                "ci95_high": 0.005,
            },
            {
                "panel": "Spa",
                "label": "SRP (Ours)",
                "mean_delta_max_cte_vs_always": -0.02,
                "ci95_low": -0.03,
                "ci95_high": -0.01,
            },
            {
                "panel": "Spa",
                "label": "SRPv2",
                "mean_delta_max_cte_vs_always": -0.03,
                "ci95_low": -0.04,
                "ci95_high": -0.02,
            },
            {
                "panel": "Aggregate",
                "label": "Never query",
                "mean_delta_max_cte_vs_always": 0.025,
                "ci95_low": 0.015,
                "ci95_high": 0.035,
            },
            {
                "panel": "Aggregate",
                "label": "Fixed Interval",
                "mean_delta_max_cte_vs_always": -0.002,
                "ci95_low": -0.01,
                "ci95_high": 0.006,
            },
            {
                "panel": "Aggregate",
                "label": "SRP (Ours)",
                "mean_delta_max_cte_vs_always": -0.018,
                "ci95_low": -0.026,
                "ci95_high": -0.01,
            },
            {
                "panel": "Aggregate",
                "label": "SRPv2",
                "mean_delta_max_cte_vs_always": -0.024,
                "ci95_low": -0.032,
                "ci95_high": -0.016,
            },
        ]
    ).to_csv(paired_csv, index=False)

    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_srpv2_comparison.py",
            "--cross-map-csv",
            str(cross_map_csv),
            "--budget-matches-csv",
            str(budget_csv),
            "--paired-delta-csv",
            str(paired_csv),
            "--output-dir",
            str(out_dir),
            "--formats",
            "png",
        ]
        module.main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "budget_matched_equal_ccr.png").exists()
    assert (out_dir / "paired_delta_vs_always.png").exists()
