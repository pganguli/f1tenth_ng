"""Tests for the paper-ready 10-map plotting script."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_plot_test_data import write_plot_fixture_csvs


def _load_plot_module():
    root = Path(__file__).resolve().parents[3]
    script_path = root / "scripts/benchmarks/plot_paper_results_10maps.py"
    spec = spec_from_file_location("plot_paper_results_10maps", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_single_summary_computes_baseline_deltas(tmp_path: Path) -> None:
    """The single-tier loader should compute delta columns against always-hit."""
    module = _load_plot_module()
    csv_path = tmp_path / "single_summary.csv"
    pd.DataFrame(
        [
            {
                "map_name": "Monza",
                "cloud_latency": 5.0,
                "experiment": "always_hit",
                "strategy": "always",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 1.0,
                "crosstrack_rmse_m_mean": 0.100,
                "in_target_ccr_band": False,
                "rank": 2,
            },
            {
                "map_name": "Monza",
                "cloud_latency": 5.0,
                "experiment": "fixed_bernoulli_p60",
                "strategy": "fixed_bernoulli",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 0.60,
                "crosstrack_rmse_m_mean": 0.095,
                "in_target_ccr_band": True,
                "rank": 1,
            },
        ]
    ).to_csv(csv_path, index=False)

    summary = module.load_single_summary(str(csv_path), 0.55, 0.65)
    row = summary[summary["strategy"] == "fixed_bernoulli"].iloc[0]

    assert row["always_rmse"] == pytest.approx(0.100)
    assert row["always_ccr"] == pytest.approx(1.0)
    assert row["rmse_delta_vs_always_pct"] == pytest.approx(-5.0)
    assert row["ccr_delta_vs_always"] == pytest.approx(-0.40)
    assert bool(row["in_target_ccr_band"]) is True


def test_strategy_win_counts_split_overall_and_target_band(tmp_path: Path) -> None:
    """Win-count helpers should distinguish best-overall from best target-band winners."""
    module = _load_plot_module()
    csv_path = tmp_path / "single_summary.csv"
    pd.DataFrame(
        [
            {
                "map_name": "Monza",
                "cloud_latency": 5.0,
                "experiment": "always_hit",
                "strategy": "always",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 1.0,
                "crosstrack_rmse_m_mean": 0.10,
                "in_target_ccr_band": False,
                "rank": 2,
            },
            {
                "map_name": "Monza",
                "cloud_latency": 5.0,
                "experiment": "fixed_bernoulli_p60",
                "strategy": "fixed_bernoulli",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 0.60,
                "crosstrack_rmse_m_mean": 0.11,
                "in_target_ccr_band": True,
                "rank": 1,
            },
            {
                "map_name": "Spa",
                "cloud_latency": 5.0,
                "experiment": "always_hit",
                "strategy": "always",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 1.0,
                "crosstrack_rmse_m_mean": 0.09,
                "in_target_ccr_band": False,
                "rank": 2,
            },
            {
                "map_name": "Spa",
                "cloud_latency": 5.0,
                "experiment": "deterministic_t0p03",
                "strategy": "deterministic",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 0.97,
                "crosstrack_rmse_m_mean": 0.085,
                "in_target_ccr_band": False,
                "rank": 1,
            },
            {
                "map_name": "Spa",
                "cloud_latency": 5.0,
                "experiment": "fixed_bernoulli_p60",
                "strategy": "fixed_bernoulli",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 0.60,
                "crosstrack_rmse_m_mean": 0.087,
                "in_target_ccr_band": True,
                "rank": 2,
            },
        ]
    ).to_csv(csv_path, index=False)

    summary = module.load_single_summary(str(csv_path), 0.55, 0.65)
    overall = module.strategy_win_counts(summary, target_only=False)
    target = module.strategy_win_counts(summary, target_only=True)

    assert overall["deterministic"] == 1
    assert overall["fixed_bernoulli"] == 1
    assert target["fixed_bernoulli"] == 2
    assert target["deterministic"] == 0


def test_load_single_summary_requires_always_hit_rows(tmp_path: Path) -> None:
    """The loader should fail clearly if a map has no always-hit baseline."""
    module = _load_plot_module()
    csv_path = tmp_path / "invalid_summary.csv"
    pd.DataFrame(
        [
            {
                "map_name": "Monza",
                "cloud_latency": 5.0,
                "experiment": "fixed_bernoulli_p60",
                "strategy": "fixed_bernoulli",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 0.60,
                "crosstrack_rmse_m_mean": 0.095,
                "in_target_ccr_band": True,
                "rank": 1,
            }
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="always-hit"):
        module.load_single_summary(str(csv_path), 0.55, 0.65)


def test_attach_oracle_baseline_computes_competitive_ratio(tmp_path: Path) -> None:
    """Oracle attachment should compute competitive-ratio columns against the latency-0 floor."""
    module = _load_plot_module()
    csv_path = tmp_path / "single_summary.csv"
    oracle_path = tmp_path / "oracle_summary.csv"

    pd.DataFrame(
        [
            {
                "map_name": "Monza",
                "cloud_latency": 5.0,
                "experiment": "always_hit",
                "strategy": "always",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 1.0,
                "crosstrack_rmse_m_mean": 0.100,
                "in_target_ccr_band": False,
                "rank": 2,
            },
            {
                "map_name": "Monza",
                "cloud_latency": 5.0,
                "experiment": "fixed_bernoulli_p60",
                "strategy": "fixed_bernoulli",
                "collision_rate": 0.0,
                "collision_free_rate": 1.0,
                "cloud_call_rate_mean": 0.60,
                "crosstrack_rmse_m_mean": 0.095,
                "in_target_ccr_band": True,
                "rank": 1,
            },
        ]
    ).to_csv(csv_path, index=False)
    pd.DataFrame(
        [
            {
                "map_name": "Monza",
                "crosstrack_rmse_m_mean": 0.090,
            }
        ]
    ).to_csv(oracle_path, index=False)

    summary = module.load_single_summary(str(csv_path), 0.55, 0.65)
    oracle = module.load_oracle_summary(str(oracle_path))
    enriched = module.attach_oracle_baseline(summary, oracle)
    row = enriched[enriched["strategy"] == "fixed_bernoulli"].iloc[0]

    assert row["oracle_rmse"] == pytest.approx(0.090)
    assert row["oracle_gap"] == pytest.approx(0.005)
    assert row["always_gap"] == pytest.approx(0.010)
    assert row["oracle_competitive_ratio"] == pytest.approx(0.095 / 0.090)
    assert row["always_oracle_ratio"] == pytest.approx(0.100 / 0.090)
    assert row["normalized_gap_vs_oracle"] == pytest.approx(0.5)


def test_paper_plot_smoke_generates_png_and_pdf(tmp_path: Path) -> None:
    """The paper plot script should render the full corrected 10-map figure suite."""
    module = _load_plot_module()
    fixture_paths = write_plot_fixture_csvs(tmp_path)
    single_summary = pd.read_csv(fixture_paths["single"])
    always = (
        single_summary[single_summary["strategy"] == "always"][
            ["map_name", "crosstrack_rmse_m_mean"]
        ]
        .groupby("map_name", as_index=False)
        .first()
    )
    oracle_path = tmp_path / "oracle_summary.csv"
    always["crosstrack_rmse_m_mean"] = always["crosstrack_rmse_m_mean"] * 0.97
    always.to_csv(oracle_path, index=False)

    outputs = module.run(
        single_summary_csv=str(fixture_paths["single"]),
        single_target_csv=str(fixture_paths["target"]),
        multi_summary_csv=str(fixture_paths["multi"]),
        oracle_summary_csv=str(oracle_path),
        output_dir=str(tmp_path),
        dpi=72,
        formats=["png", "pdf"],
        target_low=0.55,
        target_high=0.65,
    )

    expected_stems = {
        "figure1_pareto_rmse",
        "figure2_pareto_safety",
        "figure3_target_band_winners",
        "figure4_overall_vs_target",
        "figure5_strategy_wins",
        "figure6_multi_tier_tradeoff",
        "normalized_family_frontiers",
        "oracle_roofline_dumbbell",
        "target_band_family_matrix",
    }
    produced = {(path.stem, path.suffix) for path in outputs}

    for stem in expected_stems:
        assert (stem, ".png") in produced
        assert (stem, ".pdf") in produced
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()
