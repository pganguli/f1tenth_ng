"""Tests for the low-CCR per-method lambda exploratory scripts."""

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


def test_family_grid_includes_new_scheduler_family() -> None:
    """The exploratory grid should include the self-normalizing family and k2."""
    module = _load_module("low_ccr_per_method_lambda.py")

    experiments = module.family_grid()
    strategies = {exp.strategy for exp in experiments}
    names = {exp.name for exp in experiments}

    assert "self_normalizing_momentum" in strategies
    assert "srpv2" in strategies
    assert "fixed_interval_k2" in names
    assert module.SCOUT_TRAIN_MAPS == [
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


def test_select_top_k_per_family_deduplicates_base_experiments() -> None:
    """Finalists should be unique by base experiment within each family."""
    module = _load_module("low_ccr_per_method_lambda.py")
    cross_map_df = pd.DataFrame(
        [
            {
                "experiment": "fixed_interval_k2__lambda_1p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k2",
                "age_decay_lambda": 1.0,
                "params_json": '{"interval": 2, "age_decay_lambda": 1.0}',
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.1,
                "mean_crosstrack_max": 0.4,
                "mean_cloud_call_rate": 0.50,
            },
            {
                "experiment": "fixed_interval_k2__lambda_24p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k2",
                "age_decay_lambda": 24.0,
                "params_json": '{"interval": 2, "age_decay_lambda": 24.0}',
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.09,
                "mean_crosstrack_max": 0.39,
                "mean_cloud_call_rate": 0.49,
            },
            {
                "experiment": "self_norm_tau1p0_n3__lambda_24p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p0_n3",
                "age_decay_lambda": 24.0,
                "params_json": '{"tau": 1.0, "staleness_multiplier": 3, "age_decay_lambda": 24.0}',
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.1,
                "mean_crosstrack_max": 0.395,
                "mean_cloud_call_rate": 0.18,
            },
        ]
    )

    finalists = module.select_top_k_per_family(
        cross_map_df,
        finalist_limits={"fixed_interval": 1, "self_normalizing_momentum": 1},
    )
    fixed_interval_finalists = [
        finalist for finalist in finalists if finalist["strategy"] == "fixed_interval"
    ]

    assert len(fixed_interval_finalists) == 1
    assert fixed_interval_finalists[0]["base_experiment"] == "fixed_interval_k2"


def test_load_eval_experiments_keeps_interval_ladder_and_srpv2(tmp_path: Path) -> None:
    """Eval loading should preserve the interval ladder and the SRPv2 winner."""
    module = _load_module("low_ccr_per_method_lambda.py")
    payload_path = tmp_path / "configs.json"
    payload_path.write_text(
        """
        {
          "best_configs": {
            "always": {
              "experiment_name": "always_hit__lambda_0p0",
              "strategy": "always",
              "params": {"age_decay_lambda": 0.0}
            },
            "fixed_interval": {
              "experiment_name": "fixed_interval_k3__lambda_16p0",
              "strategy": "fixed_interval",
              "params": {"interval": 3, "age_decay_lambda": 16.0}
            },
            "fixed_bernoulli": {
              "experiment_name": "fixed_bernoulli_p40__lambda_16p0",
              "strategy": "fixed_bernoulli",
              "params": {"p": 0.4, "seed": 7, "age_decay_lambda": 16.0}
            },
            "bernoulli_max_miss": {
              "experiment_name": "bernoulli_max_miss_p15_m2__lambda_20p666667",
              "strategy": "bernoulli_max_miss",
              "params": {"p": 0.15, "max_miss": 2, "seed": 7, "age_decay_lambda": 20.666667}
            },
            "deterministic": {
              "experiment_name": "deterministic_t0p1__lambda_90p666667",
              "strategy": "deterministic",
              "params": {"threshold": 0.1, "age_decay_lambda": 90.666667}
            },
            "self_normalizing_momentum": {
              "experiment_name": "self_norm_tau1p0_n3__lambda_4p0",
              "strategy": "self_normalizing_momentum",
              "params": {"tau": 1.0, "nmax": 3, "seed": 7, "age_decay_lambda": 4.0}
            },
            "srpv2": {
              "experiment_name": "srpv2_tau1p0_n3__lambda_4p0",
              "strategy": "srpv2",
              "params": {"tau": 1.0, "nmax": 3, "seed": 7, "age_decay_lambda": 4.0}
            }
          },
          "fixed_interval_controls": [
            {
              "experiment_name": "fixed_interval_k2__lambda_32p0",
              "strategy": "fixed_interval",
              "params": {"interval": 2, "age_decay_lambda": 32.0}
            },
            {
              "experiment_name": "fixed_interval_k5__lambda_16p0",
              "strategy": "fixed_interval",
              "params": {"interval": 5, "age_decay_lambda": 16.0}
            }
          ],
          "baseline_config": {
            "experiment_name": "fixed_interval_k2__lambda_32p0",
            "strategy": "fixed_interval",
            "params": {"interval": 2, "age_decay_lambda": 32.0}
          }
        }
        """
    )

    experiments, _payload = module.load_eval_experiments(payload_path)
    names = {exp.name for exp in experiments}

    assert "fixed_interval_k2__lambda_32p0" in names
    assert "fixed_interval_k5__lambda_16p0" in names
    assert "srpv2_tau1p0_n3__lambda_4p0" in names


def test_acceptable_by_k2_uses_one_percent_threshold() -> None:
    """Acceptance should require the k2 band and zero collision/step-cap rate."""
    module = _load_module("low_ccr_per_method_lambda.py")
    cross_map_df = pd.DataFrame(
        [
            {
                "experiment": "fixed_interval_k2__lambda_24p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k2",
                "mean_crosstrack_max": 0.50,
                "mean_crosstrack_rmse": 0.10,
                "mean_cloud_call_rate": 0.50,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
            },
            {
                "experiment": "self_norm_tau1p0_n3__lambda_24p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p0_n3",
                "mean_crosstrack_max": 0.504,
                "mean_crosstrack_rmse": 0.09,
                "mean_cloud_call_rate": 0.18,
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
            },
            {
                "experiment": "deterministic_t0p30__lambda_24p0",
                "strategy": "deterministic",
                "base_experiment": "deterministic_t0p30",
                "mean_crosstrack_max": 0.504,
                "mean_crosstrack_rmse": 0.08,
                "mean_cloud_call_rate": 0.15,
                "mean_collision_rate": 0.2,
                "mean_step_cap_rate": 0.0,
            },
        ]
    )

    ranked, baseline_row = module.acceptable_by_k2(
        cross_map_df,
        baseline_experiment_name="fixed_interval_k2__lambda_24p0",
    )

    assert baseline_row["mean_crosstrack_max"] == 0.50
    assert ranked.loc[ranked["base_experiment"] == "self_norm_tau1p0_n3", "acceptable"].item()
    assert not ranked.loc[ranked["base_experiment"] == "deterministic_t0p30", "acceptable"].item()


def test_plot_low_ccr_outputs_render_figures(tmp_path: Path) -> None:
    """The low-CCR plotter should render the exploratory figure set."""
    module = _load_module("plot_low_ccr_per_method_lambda.py")
    cross_map_csv = tmp_path / "cross_map.csv"
    acceptable_csv = tmp_path / "acceptable.csv"
    wins_csv = tmp_path / "wins.csv"
    out_dir = tmp_path / "figs"

    pd.DataFrame(
        [
            {
                "experiment": "fixed_interval_k2__lambda_24p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k2",
                "mean_cloud_call_rate": 0.50,
                "mean_crosstrack_max": 0.50,
                "mean_crosstrack_rmse": 0.10,
                "ccr_reduction_vs_k2": 0.0,
                "max_cte_delta_vs_k2": 0.0,
                "k2_threshold": 0.505,
            },
            {
                "experiment": "self_norm_tau1p0_n3__lambda_24p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p0_n3",
                "mean_cloud_call_rate": 0.18,
                "mean_crosstrack_max": 0.504,
                "mean_crosstrack_rmse": 0.09,
                "ccr_reduction_vs_k2": 0.32,
                "max_cte_delta_vs_k2": 0.004,
                "k2_threshold": 0.505,
            },
        ]
    ).to_csv(cross_map_csv, index=False)
    pd.DataFrame(
        [
            {
                "experiment": "self_norm_tau1p0_n3__lambda_24p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p0_n3",
                "mean_cloud_call_rate": 0.18,
                "mean_crosstrack_max": 0.504,
                "mean_crosstrack_rmse": 0.09,
                "ccr_reduction_vs_k2": 0.32,
                "max_cte_delta_vs_k2": 0.004,
            }
        ]
    ).to_csv(acceptable_csv, index=False)
    pd.DataFrame(
        [
            {
                "experiment": "self_norm_tau1p0_n3__lambda_24p0",
                "strategy": "self_normalizing_momentum",
                "maps_won": 3,
            }
        ]
    ).to_csv(wins_csv, index=False)

    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_low_ccr_per_method_lambda.py",
            "--cross-map-csv",
            str(cross_map_csv),
            "--acceptable-csv",
            str(acceptable_csv),
            "--win-counts-csv",
            str(wins_csv),
            "--output-dir",
            str(out_dir),
            "--formats",
            "png",
        ]
        module.main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "low_ccr_tradeoff.png").exists()
    assert (out_dir / "acceptable_configs.png").exists()
    assert (out_dir / "win_counts.png").exists()
    assert (out_dir / "direct_comparison.png").exists()
