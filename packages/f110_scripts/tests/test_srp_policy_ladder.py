"""Tests for the focused SRP/SRPv2 policy ladder study scripts."""

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


def test_build_base_experiments_contains_interval_ladder_and_srp_families() -> None:
    """The focused study grid should cover the requested interval ladder and SRP families."""
    module = _load_module("benchmark_srp_policy_ladder.py")
    experiments = module.build_base_experiments(
        interval_grid=[2, 3, 4, 7, 10, 15],
        tau_grid=[0.5, 1.0],
        nmax_grid=[2, 3],
    )
    names = {exp.name for exp in experiments}
    strategies = {exp.strategy for exp in experiments}

    assert "fixed_interval_k2" in names
    assert "fixed_interval_k10" in names
    assert "fixed_interval_k15" in names
    assert "fixed_interval_k5" not in names
    assert "self_norm_tau0p5_n2" in names
    assert "srpv2_tau1p0_n3" in names
    assert strategies == {"fixed_interval", "self_normalizing_momentum", "srpv2"}


def test_select_family_shortlist_prefers_pareto_then_fills_to_minimum() -> None:
    """Shortlisting should keep frontier points and backfill until the minimum count is met."""
    module = _load_module("benchmark_srp_policy_ladder.py")
    cross_map_df = pd.DataFrame(
        [
            {
                "experiment": "self_norm_tau0p5_n2__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau0p5_n2",
                "age_decay_lambda": 4.0,
                "params_json": '{"tau": 0.5, "nmax": 2, "seed": 7, "age_decay_lambda": 4.0}',
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.100,
                "mean_crosstrack_max": 0.520,
                "mean_cloud_call_rate": 0.090,
            },
            {
                "experiment": "self_norm_tau0p75_n2__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau0p75_n2",
                "age_decay_lambda": 4.0,
                "params_json": '{"tau": 0.75, "nmax": 2, "seed": 7, "age_decay_lambda": 4.0}',
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.099,
                "mean_crosstrack_max": 0.522,
                "mean_cloud_call_rate": 0.085,
            },
            {
                "experiment": "self_norm_tau1p0_n2__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p0_n2",
                "age_decay_lambda": 4.0,
                "params_json": '{"tau": 1.0, "nmax": 2, "seed": 7, "age_decay_lambda": 4.0}',
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.101,
                "mean_crosstrack_max": 0.530,
                "mean_cloud_call_rate": 0.082,
            },
            {
                "experiment": "self_norm_tau1p25_n2__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p25_n2",
                "age_decay_lambda": 4.0,
                "params_json": '{"tau": 1.25, "nmax": 2, "seed": 7, "age_decay_lambda": 4.0}',
                "mean_collision_rate": 0.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.102,
                "mean_crosstrack_max": 0.525,
                "mean_cloud_call_rate": 0.095,
            },
        ]
    )

    shortlist = module.select_family_shortlist(
        cross_map_df,
        strategy="self_normalizing_momentum",
        min_shortlist=4,
        max_shortlist=6,
    )

    assert len(shortlist) == 4
    assert shortlist[0]["pareto_frontier"] is True
    assert shortlist[1]["pareto_frontier"] is True
    assert {row["base_experiment"] for row in shortlist} == {
        "self_norm_tau0p5_n2",
        "self_norm_tau0p75_n2",
        "self_norm_tau1p0_n2",
        "self_norm_tau1p25_n2",
    }


def test_build_eval_experiments_preserves_interval_ladder_and_shortlists() -> None:
    """Eval assembly should keep always/never, interval controls, and both family shortlists."""
    module = _load_module("benchmark_srp_policy_ladder.py")
    payload = {
        "interval_controls": [
            {
                "experiment_name": "fixed_interval_k2__lambda_4p0",
                "strategy": "fixed_interval",
                "params": {"interval": 2, "age_decay_lambda": 4.0},
            },
            {
                "experiment_name": "fixed_interval_k10__lambda_4p0",
                "strategy": "fixed_interval",
                "params": {"interval": 10, "age_decay_lambda": 4.0},
            },
            {
                "experiment_name": "fixed_interval_k15__lambda_4p0",
                "strategy": "fixed_interval",
                "params": {"interval": 15, "age_decay_lambda": 4.0},
            },
        ],
        "srp_shortlist": [
            {
                "experiment_name": "self_norm_tau1p0_n3__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "params": {"tau": 1.0, "nmax": 3, "seed": 7, "age_decay_lambda": 4.0},
            }
        ],
        "srpv2_shortlist": [
            {
                "experiment_name": "srpv2_tau0p75_n4__lambda_4p0",
                "strategy": "srpv2",
                "params": {"tau": 0.75, "nmax": 4, "seed": 7, "age_decay_lambda": 4.0},
            }
        ],
    }

    experiments = module.build_eval_experiments(payload)
    names = {exp.name for exp in experiments}

    assert "always_hit__lambda_0p0" in names
    assert "never_query__lambda_0p0" in names
    assert "fixed_interval_k2__lambda_4p0" in names
    assert "fixed_interval_k10__lambda_4p0" in names
    assert "fixed_interval_k15__lambda_4p0" in names
    assert "self_norm_tau1p0_n3__lambda_4p0" in names
    assert "srpv2_tau0p75_n4__lambda_4p0" in names


def test_plot_policy_ladder_outputs_render_figures(tmp_path: Path) -> None:
    """The policy ladder plotter should render the figure set and tables."""
    module = _load_module("plot_srp_policy_ladder.py")
    cross_map_csv = tmp_path / "cross_map.csv"
    budget_csv = tmp_path / "budget.csv"
    paired_csv = tmp_path / "paired.csv"
    per_map_csv = tmp_path / "per_map.csv"
    out_dir = tmp_path / "figs"

    pd.DataFrame(
        [
            {
                "experiment": "always_hit__lambda_0p0",
                "strategy": "always",
                "base_experiment": "always_hit",
                "display_strategy": "Always query",
                "params_json": '{"age_decay_lambda": 0.0}',
                "mean_cloud_call_rate": 1.0,
                "mean_crosstrack_max": 0.549,
                "mean_crosstrack_max_cm": 54.9,
                "mean_crosstrack_rmse": 0.131,
                "mean_crosstrack_rmse_cm": 13.1,
                "mean_collision_rate": 0.0,
                "acceptable": False,
                "pareto_frontier": False,
            },
            {
                "experiment": "never_query__lambda_0p0",
                "strategy": "never_query",
                "base_experiment": "never_query",
                "display_strategy": "Never query",
                "params_json": '{"age_decay_lambda": 0.0}',
                "mean_cloud_call_rate": 0.0,
                "mean_crosstrack_max": 0.568,
                "mean_crosstrack_max_cm": 56.8,
                "mean_crosstrack_rmse": 0.130,
                "mean_crosstrack_rmse_cm": 13.0,
                "mean_collision_rate": 0.0,
                "acceptable": False,
                "pareto_frontier": False,
            },
            {
                "experiment": "fixed_interval_k2__lambda_4p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k2",
                "display_strategy": "Fixed Interval",
                "params_json": '{"interval": 2, "age_decay_lambda": 4.0}',
                "mean_cloud_call_rate": 0.50,
                "mean_crosstrack_max": 0.515,
                "mean_crosstrack_max_cm": 51.5,
                "mean_crosstrack_rmse": 0.094,
                "mean_crosstrack_rmse_cm": 9.4,
                "mean_collision_rate": 0.0,
                "acceptable": True,
                "pareto_frontier": True,
            },
            {
                "experiment": "fixed_interval_k4__lambda_4p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k4",
                "display_strategy": "Fixed Interval",
                "params_json": '{"interval": 4, "age_decay_lambda": 4.0}',
                "mean_cloud_call_rate": 0.25,
                "mean_crosstrack_max": 0.523,
                "mean_crosstrack_max_cm": 52.3,
                "mean_crosstrack_rmse": 0.096,
                "mean_crosstrack_rmse_cm": 9.6,
                "mean_collision_rate": 0.0,
                "acceptable": False,
                "pareto_frontier": False,
            },
            {
                "experiment": "fixed_interval_k10__lambda_4p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k10",
                "display_strategy": "Fixed Interval",
                "params_json": '{"interval": 10, "age_decay_lambda": 4.0}',
                "mean_cloud_call_rate": 0.10,
                "mean_crosstrack_max": 0.526,
                "mean_crosstrack_max_cm": 52.6,
                "mean_crosstrack_rmse": 0.096,
                "mean_crosstrack_rmse_cm": 9.6,
                "mean_collision_rate": 0.0,
                "acceptable": False,
                "pareto_frontier": False,
            },
            {
                "experiment": "fixed_interval_k15__lambda_4p0",
                "strategy": "fixed_interval",
                "base_experiment": "fixed_interval_k15",
                "display_strategy": "Fixed Interval",
                "params_json": '{"interval": 15, "age_decay_lambda": 4.0}',
                "mean_cloud_call_rate": 0.067,
                "mean_crosstrack_max": 0.531,
                "mean_crosstrack_max_cm": 53.1,
                "mean_crosstrack_rmse": 0.097,
                "mean_crosstrack_rmse_cm": 9.7,
                "mean_collision_rate": 0.0,
                "acceptable": False,
                "pareto_frontier": False,
            },
            {
                "experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "strategy": "self_normalizing_momentum",
                "base_experiment": "self_norm_tau1p0_n3",
                "display_strategy": "SRP (Ours)",
                "params_json": '{"tau": 1.0, "nmax": 3, "seed": 7, "age_decay_lambda": 4.0}',
                "mean_cloud_call_rate": 0.086,
                "mean_crosstrack_max": 0.524,
                "mean_crosstrack_max_cm": 52.4,
                "mean_crosstrack_rmse": 0.096,
                "mean_crosstrack_rmse_cm": 9.6,
                "mean_collision_rate": 0.0,
                "acceptable": False,
                "pareto_frontier": True,
            },
            {
                "experiment": "srpv2_tau0p5_n4__lambda_4p0",
                "strategy": "srpv2",
                "base_experiment": "srpv2_tau0p5_n4",
                "display_strategy": "SRPv2",
                "params_json": '{"tau": 0.5, "nmax": 4, "seed": 7, "age_decay_lambda": 4.0}',
                "mean_cloud_call_rate": 0.089,
                "mean_crosstrack_max": 0.519,
                "mean_crosstrack_max_cm": 51.9,
                "mean_crosstrack_rmse": 0.096,
                "mean_crosstrack_rmse_cm": 9.6,
                "mean_collision_rate": 0.0,
                "acceptable": True,
                "pareto_frontier": True,
            },
        ]
    ).to_csv(cross_map_csv, index=False)

    pd.DataFrame(
        [
            {
                "display_strategy": "SRP (Ours)",
                "experiment": "self_norm_tau1p0_n3__lambda_4p0",
                "mean_cloud_call_rate": 0.086,
                "mean_crosstrack_max_cm": 52.4,
                "delta_max_cte_vs_interval_cm": -0.2,
                "matched_interval_k": 10,
            },
            {
                "display_strategy": "SRPv2",
                "experiment": "srpv2_tau0p5_n4__lambda_4p0",
                "mean_cloud_call_rate": 0.089,
                "mean_crosstrack_max_cm": 51.9,
                "delta_max_cte_vs_interval_cm": -0.7,
                "matched_interval_k": 10,
            },
        ]
    ).to_csv(budget_csv, index=False)

    pd.DataFrame(
        [
            {
                "panel": "Sochi",
                "label": "Never query",
                "mean_delta_max_cte_vs_always_cm": 2.0,
                "ci95_low_cm": 1.0,
                "ci95_high_cm": 3.0,
            },
            {
                "panel": "Sochi",
                "label": "Fixed Interval",
                "mean_delta_max_cte_vs_always_cm": 0.0,
                "ci95_low_cm": -1.0,
                "ci95_high_cm": 1.0,
            },
            {
                "panel": "Sochi",
                "label": "SRP (Ours)",
                "mean_delta_max_cte_vs_always_cm": -0.5,
                "ci95_low_cm": -1.5,
                "ci95_high_cm": 0.2,
            },
            {
                "panel": "Sochi",
                "label": "SRPv2",
                "mean_delta_max_cte_vs_always_cm": -0.8,
                "ci95_low_cm": -1.8,
                "ci95_high_cm": -0.1,
            },
            {
                "panel": "Spa",
                "label": "Never query",
                "mean_delta_max_cte_vs_always_cm": 2.5,
                "ci95_low_cm": 1.5,
                "ci95_high_cm": 3.5,
            },
            {
                "panel": "Spa",
                "label": "Fixed Interval",
                "mean_delta_max_cte_vs_always_cm": -0.2,
                "ci95_low_cm": -1.0,
                "ci95_high_cm": 0.6,
            },
            {
                "panel": "Spa",
                "label": "SRP (Ours)",
                "mean_delta_max_cte_vs_always_cm": -0.9,
                "ci95_low_cm": -1.8,
                "ci95_high_cm": -0.1,
            },
            {
                "panel": "Spa",
                "label": "SRPv2",
                "mean_delta_max_cte_vs_always_cm": -1.2,
                "ci95_low_cm": -2.0,
                "ci95_high_cm": -0.3,
            },
            {
                "panel": "Aggregate",
                "label": "Never query",
                "mean_delta_max_cte_vs_always_cm": 2.2,
                "ci95_low_cm": 1.4,
                "ci95_high_cm": 3.0,
            },
            {
                "panel": "Aggregate",
                "label": "Fixed Interval",
                "mean_delta_max_cte_vs_always_cm": -0.1,
                "ci95_low_cm": -0.8,
                "ci95_high_cm": 0.5,
            },
            {
                "panel": "Aggregate",
                "label": "SRP (Ours)",
                "mean_delta_max_cte_vs_always_cm": -0.7,
                "ci95_low_cm": -1.4,
                "ci95_high_cm": -0.1,
            },
            {
                "panel": "Aggregate",
                "label": "SRPv2",
                "mean_delta_max_cte_vs_always_cm": -1.0,
                "ci95_low_cm": -1.7,
                "ci95_high_cm": -0.2,
            },
        ]
    ).to_csv(paired_csv, index=False)

    pd.DataFrame(
        [
            {
                "family": "SRP (Ours)",
                "map_name": "Sochi",
                "interval_max_cte_cm": 52.6,
                "target_max_cte_cm": 52.4,
            },
            {
                "family": "SRP (Ours)",
                "map_name": "Spa",
                "interval_max_cte_cm": 52.5,
                "target_max_cte_cm": 52.2,
            },
            {
                "family": "SRPv2",
                "map_name": "Sochi",
                "interval_max_cte_cm": 52.6,
                "target_max_cte_cm": 51.9,
            },
            {
                "family": "SRPv2",
                "map_name": "Spa",
                "interval_max_cte_cm": 52.5,
                "target_max_cte_cm": 51.8,
            },
        ]
    ).to_csv(per_map_csv, index=False)

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "plot_srp_policy_ladder.py",
            "--cross-map-csv",
            str(cross_map_csv),
            "--budget-matches-csv",
            str(budget_csv),
            "--paired-delta-csv",
            str(paired_csv),
            "--per-map-csv",
            str(per_map_csv),
            "--output-dir",
            str(out_dir),
            "--formats",
            "png",
        ]
        module.main()
    finally:
        sys.argv = original_argv

    assert (out_dir / "max_cte_vs_cloud_call_rate.png").exists()
    assert (out_dir / "pareto_frontier_max_cte.png").exists()
    assert (out_dir / "budget_matched_vs_interval.png").exists()
    assert (out_dir / "interval_ladder_only.png").exists()
    assert (out_dir / "per_map_srpx_vs_interval.png").exists()
    assert (out_dir / "paired_delta_vs_always.png").exists()
    assert (out_dir / "headline_performance_table.csv").exists()
    assert (out_dir / "budget_match_table.csv").exists()
