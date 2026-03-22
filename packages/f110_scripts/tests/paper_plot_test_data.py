"""Synthetic CSV fixtures for paper-plot smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MAPS = [
    "Austin",
    "BrandsHatch",
    "Hockenheim",
    "MexicoCity",
    "Montreal",
    "Monza",
    "Oschersleben",
    "Shanghai",
    "Spa",
    "Spielberg",
]


def _single_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_rmse = {
        "Austin": 0.1114,
        "BrandsHatch": 0.0920,
        "Hockenheim": 0.0987,
        "MexicoCity": 0.1126,
        "Montreal": 0.1156,
        "Monza": 0.0753,
        "Oschersleben": 0.1170,
        "Shanghai": 0.1299,
        "Spa": 0.0871,
        "Spielberg": 0.0854,
    }
    target_winner = {
        "Austin": "bernoulli_max_miss",
        "BrandsHatch": "fixed_bernoulli",
        "Hockenheim": "bernoulli_max_miss",
        "MexicoCity": "bernoulli_max_miss",
        "Montreal": "bernoulli_max_miss",
        "Monza": "fixed_bernoulli",
        "Oschersleben": "bernoulli_max_miss",
        "Shanghai": "bernoulli_max_miss",
        "Spa": "fixed_bernoulli",
        "Spielberg": "fixed_bernoulli",
    }
    overall_winner = {
        "Austin": "bernoulli_max_miss",
        "BrandsHatch": "deterministic",
        "Hockenheim": "deterministic",
        "MexicoCity": "fixed_bernoulli",
        "Montreal": "exponential",
        "Monza": "fixed_bernoulli",
        "Oschersleben": "bernoulli_max_miss",
        "Shanghai": "logistic",
        "Spa": "logistic",
        "Spielberg": "deterministic",
    }

    for map_name in MAPS:
        always_rmse = base_rmse[map_name]
        target = target_winner[map_name]
        overall = overall_winner[map_name]
        rows.extend(
            [
                {
                    "map_name": map_name,
                    "cloud_latency": 5.0,
                    "experiment": "always_hit",
                    "strategy": "always",
                    "collision_rate": 0.0,
                    "collision_free_rate": 1.0,
                    "cloud_call_rate_mean": 1.0,
                    "crosstrack_rmse_m_mean": always_rmse,
                    "in_target_ccr_band": False,
                    "rank": 4 if overall != "always" else 1,
                },
                {
                    "map_name": map_name,
                    "cloud_latency": 5.0,
                    "experiment": "fixed_bernoulli_p60",
                    "strategy": "fixed_bernoulli",
                    "collision_rate": 0.0,
                    "collision_free_rate": 1.0,
                    "cloud_call_rate_mean": 0.598,
                    "crosstrack_rmse_m_mean": always_rmse - (0.0010 if target == "fixed_bernoulli" else 0.0003),
                    "in_target_ccr_band": True,
                    "rank": 1 if overall == "fixed_bernoulli" else 2,
                },
                {
                    "map_name": map_name,
                    "cloud_latency": 5.0,
                    "experiment": "bernoulli_max_miss_p55_m5",
                    "strategy": "bernoulli_max_miss",
                    "collision_rate": 0.0,
                    "collision_free_rate": 1.0,
                    "cloud_call_rate_mean": 0.556,
                    "crosstrack_rmse_m_mean": always_rmse - (0.0011 if target == "bernoulli_max_miss" else 0.0002),
                    "in_target_ccr_band": True,
                    "rank": 1 if overall == "bernoulli_max_miss" else 2,
                },
                {
                    "map_name": map_name,
                    "cloud_latency": 5.0,
                    "experiment": "logistic_c0p02_s30",
                    "strategy": "logistic",
                    "collision_rate": 0.0,
                    "collision_free_rate": 1.0,
                    "cloud_call_rate_mean": 0.965,
                    "crosstrack_rmse_m_mean": always_rmse - (0.0012 if overall == "logistic" else 0.0001),
                    "in_target_ccr_band": False,
                    "rank": 1 if overall == "logistic" else 3,
                },
                {
                    "map_name": map_name,
                    "cloud_latency": 5.0,
                    "experiment": "exponential_c0p02_r15",
                    "strategy": "exponential",
                    "collision_rate": 0.0,
                    "collision_free_rate": 1.0,
                    "cloud_call_rate_mean": 0.935,
                    "crosstrack_rmse_m_mean": always_rmse - (0.00115 if overall == "exponential" else 0.00005),
                    "in_target_ccr_band": False,
                    "rank": 1 if overall == "exponential" else 3,
                },
                {
                    "map_name": map_name,
                    "cloud_latency": 5.0,
                    "experiment": "deterministic_t0p05",
                    "strategy": "deterministic",
                    "collision_rate": 0.0,
                    "collision_free_rate": 1.0,
                    "cloud_call_rate_mean": 0.972,
                    "crosstrack_rmse_m_mean": always_rmse - (0.00125 if overall == "deterministic" else 0.00008),
                    "in_target_ccr_band": False,
                    "rank": 1 if overall == "deterministic" else 3,
                },
            ]
        )
    return pd.DataFrame(rows)


def _target_summary(single_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        single_summary[single_summary["in_target_ccr_band"]]
        .sort_values(["map_name", "rank"], kind="stable")
        .groupby("map_name", as_index=False)
        .first()
    )


def _multi_summary() -> pd.DataFrame:
    rows = []
    experiments = [
        "prob_balanced",
        "threshold_large_sparing",
        "prob_medium_favoring",
        "edge_only_arch2",
    ]
    for index, map_name in enumerate(MAPS):
        rows.append(
            {
                "map_name": map_name,
                "experiment": experiments[index % len(experiments)],
                "collision": 0,
                "crosstrack_rmse_delta_pct_vs_large_always": -0.5 - 0.6 * (index % 4),
                "large_call_reduction_pct_vs_large_always": 72.0 + 2.5 * index,
            }
        )
    return pd.DataFrame(rows)


def write_plot_fixture_csvs(tmp_path: Path) -> dict[str, Path]:
    """Write a small but representative set of plotting CSV fixtures."""
    single = _single_summary()
    target = _target_summary(single)
    multi = _multi_summary()

    single_path = tmp_path / "single_summary.csv"
    target_path = tmp_path / "single_target.csv"
    multi_path = tmp_path / "multi_summary.csv"

    single.to_csv(single_path, index=False)
    target.to_csv(target_path, index=False)
    multi.to_csv(multi_path, index=False)

    return {
        "single": single_path,
        "target": target_path,
        "multi": multi_path,
    }
