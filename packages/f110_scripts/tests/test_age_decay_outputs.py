"""Smoke tests for anchored age-decay plotting and report generation."""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


_BENCHMARKS_DIR: str | None = None


def _load_module(script_name: str):
    """Load a benchmark helper as a module."""
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


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_plot_age_decay_outputs_renders_figures(tmp_path: Path) -> None:
    """The age-decay plotter should render the full figure set from small CSVs."""
    mod = _load_module("plot_age_decay_paper_figures.py")
    lambda_summary = tmp_path / "lambda_summary.csv"
    lambda_raw = tmp_path / "lambda_raw.csv"
    static_summary = tmp_path / "static_summary.csv"
    lambda_heldout = tmp_path / "lambda_heldout.csv"
    best_summary = tmp_path / "best_summary.csv"
    selected_json = tmp_path / "selected.json"
    out_dir = tmp_path / "figs"

    _write_summary(
        lambda_summary,
        [
            {
                "age_decay_lambda": 0.0,
                "representative_rows": 4,
                "mean_collision_rate": 0.0,
                "mean_collision_free_rate": 1.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.11,
                "std_crosstrack_rmse": 0.01,
                "mean_cloud_call_rate": 0.59,
                "std_cloud_call_rate": 0.02,
                "frac_in_target_band": 1.0,
                "mean_abs_ccr_distance": 0.01,
                "collision_feasible": True,
            },
            {
                "age_decay_lambda": 1.0,
                "representative_rows": 4,
                "mean_collision_rate": 0.0,
                "mean_collision_free_rate": 1.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.09,
                "std_crosstrack_rmse": 0.01,
                "mean_cloud_call_rate": 0.60,
                "std_cloud_call_rate": 0.02,
                "frac_in_target_band": 1.0,
                "mean_abs_ccr_distance": 0.0,
                "collision_feasible": True,
            },
        ],
    )
    _write_summary(
        lambda_raw,
        [
            {
                "age_decay_lambda": value,
                "experiment": experiment,
                "strategy": strategy,
                "run_idx": run_idx,
                "crosstrack_rmse_m": rmse,
            }
            for value, experiment, strategy, rmse in [
                (0.0, "always_hit", "always", 0.12),
                (0.0, "fixed_interval_k5", "fixed_interval", 0.15),
                (0.0, "fixed_bernoulli_p60", "fixed_bernoulli", 0.11),
                (0.0, "bernoulli_max_miss_p60_m5", "bernoulli_max_miss", 0.1),
                (1.0, "always_hit", "always", 0.11),
                (1.0, "fixed_interval_k5", "fixed_interval", 0.14),
                (1.0, "fixed_bernoulli_p60", "fixed_bernoulli", 0.09),
                (1.0, "bernoulli_max_miss_p60_m5", "bernoulli_max_miss", 0.08),
            ]
            for run_idx in range(3)
        ],
    )

    base_rows = [
        {
            "map_name": "Monza",
            "cloud_latency": 5,
            "experiment": "fixed_bernoulli_p60",
            "strategy": "fixed_bernoulli",
            "trials": 5,
            "collision_rate": 0.0,
            "collision_free_rate": 1.0,
            "cloud_call_rate_mean": 0.60,
            "cloud_call_rate_ci95": 0.03,
            "cloud_call_rate_std": 0.02,
            "crosstrack_rmse_m_mean": 0.10,
            "crosstrack_rmse_m_ci95": 0.01,
            "crosstrack_rmse_m_std": 0.01,
            "in_target_ccr_band": True,
            "rank": 1,
            "rmse_delta_vs_always_pct": -5.0,
        },
        {
            "map_name": "Spa",
            "cloud_latency": 5,
            "experiment": "bernoulli_max_miss_p60_m5",
            "strategy": "bernoulli_max_miss",
            "trials": 5,
            "collision_rate": 0.0,
            "collision_free_rate": 1.0,
            "cloud_call_rate_mean": 0.58,
            "cloud_call_rate_ci95": 0.02,
            "cloud_call_rate_std": 0.02,
            "crosstrack_rmse_m_mean": 0.12,
            "crosstrack_rmse_m_ci95": 0.01,
            "crosstrack_rmse_m_std": 0.01,
            "in_target_ccr_band": True,
            "rank": 1,
            "rmse_delta_vs_always_pct": -3.0,
        },
    ]
    _write_summary(static_summary, base_rows)
    lambda_rows = [dict(row, rmse_delta_vs_always_pct=row["rmse_delta_vs_always_pct"] - 1.0) for row in base_rows]
    _write_summary(lambda_heldout, lambda_rows)
    _write_summary(best_summary, lambda_rows)
    selected_json.write_text(json.dumps({"selected_lambda": 1.0}))

    mod.main = mod.main  # touch module for lint sanity
    argv = [
        "plot_age_decay_paper_figures.py",
        "--lambda-summary-csv",
        str(lambda_summary),
        "--lambda-per-config-csv",
        str(lambda_raw),
        "--static-summary-csv",
        str(static_summary),
        "--lambda-summary-heldout-csv",
        str(lambda_heldout),
        "--best-config-summary-csv",
        str(best_summary),
        "--selected-lambda-json",
        str(selected_json),
        "--output-dir",
        str(out_dir),
        "--formats",
        "png",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "lambda_train_sweep.png").exists()
    assert (out_dir / "static_vs_lambda_target_band.png").exists()
    assert (out_dir / "strategy_family_tradeoff.png").exists()
    assert (out_dir / "strategy_win_summary.png").exists()
    assert (out_dir / "alpha_decay_curves.png").exists()


def test_write_age_decay_reports_emits_markdown_and_tex(tmp_path: Path) -> None:
    """The report writer should emit both markdown and LaTeX source files."""
    mod = _load_module("write_age_decay_reports.py")
    selected_json = tmp_path / "selected.json"
    lambda_summary = tmp_path / "lambda_summary.csv"
    static_summary = tmp_path / "static_summary.csv"
    lambda_heldout = tmp_path / "lambda_heldout.csv"
    best_configs = tmp_path / "best_configs.json"
    best_summary = tmp_path / "best_summary.csv"
    methodology_md = tmp_path / "methodology.md"
    results_md = tmp_path / "results.md"
    methodology_tex = tmp_path / "methodology.tex"
    results_tex = tmp_path / "results.tex"

    selected_json.write_text(json.dumps({"selected_lambda": 1.0, "lambda_grid": [0.0, 1.0]}))
    _write_summary(
        lambda_summary,
        [
            {
                "age_decay_lambda": 1.0,
                "mean_collision_rate": 0.0,
                "mean_collision_free_rate": 1.0,
                "mean_step_cap_rate": 0.0,
                "mean_crosstrack_rmse": 0.1,
                "mean_cloud_call_rate": 0.6,
                "frac_in_target_band": 1.0,
            }
        ],
    )
    base_summary = [
        {
            "map_name": "Monza",
            "experiment": "fixed_bernoulli_p60",
            "strategy": "fixed_bernoulli",
            "rank": 1,
            "in_target_ccr_band": True,
            "cloud_call_rate_mean": 0.60,
            "crosstrack_rmse_m_mean": 0.10,
            "collision_free_rate": 1.0,
        }
    ]
    _write_summary(static_summary, base_summary)
    _write_summary(lambda_heldout, base_summary)
    _write_summary(best_summary, base_summary)
    best_configs.write_text(
        json.dumps(
            {
                "best_configs": {
                    "fixed_bernoulli": {
                        "experiment_name": "fixed_bernoulli_p60",
                        "strategy": "fixed_bernoulli",
                        "params": {"p": 0.6},
                        "train_metrics": {
                            "mean_crosstrack_rmse": 0.1,
                            "mean_cloud_call_rate": 0.6,
                            "mean_collision_rate": 0.0,
                        },
                    }
                }
            }
        )
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "write_age_decay_reports.py",
            "--selected-lambda-json",
            str(selected_json),
            "--lambda-summary-csv",
            str(lambda_summary),
            "--static-summary-csv",
            str(static_summary),
            "--lambda-summary-heldout-csv",
            str(lambda_heldout),
            "--best-configs-json",
            str(best_configs),
            "--best-config-summary-csv",
            str(best_summary),
            "--methodology-md",
            str(methodology_md),
            "--results-md",
            str(results_md),
            "--methodology-tex",
            str(methodology_tex),
            "--results-tex",
            str(results_tex),
        ]
        mod.main()
    finally:
        sys.argv = old_argv

    assert methodology_md.exists()
    assert results_md.exists()
    assert methodology_tex.exists()
    assert results_tex.exists()
