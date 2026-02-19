"""
Unit tests for f110_planning.metric_callbacks.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

import math
from typing import Any

import numpy as np
import pytest

from f110_planning.base import Action
from f110_planning.metric_callbacks import (
    BaseMetric,
    CrossTrackErrorMetric,
    HeadingErrorMetric,
    LapTimeMetric,
    MetricAggregator,
    SmoothnessMetric,
    SpeedMetric,
    WallProximityMetric,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    x: float = 0.0,
    y: float = 0.0,
    theta: float = 0.0,
    vx: float = 1.0,
    collision: float = 0.0,
    lap_count: float = 0.0,
    scan: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a minimal observation dict for a single agent."""
    if scan is None:
        scan = np.full(1080, 5.0)  # 5 m everywhere
    return {
        "poses_x": np.array([x]),
        "poses_y": np.array([y]),
        "poses_theta": np.array([theta]),
        "linear_vels_x": np.array([vx]),
        "linear_vels_y": np.array([0.0]),
        "ang_vels_z": np.array([0.0]),
        "steering_angles": np.array([0.0]),
        "collisions": np.array([collision]),
        "lap_times": np.array([0.0]),
        "lap_counts": np.array([lap_count]),
        "scans": np.array([scan]),
    }


def _make_waypoints_line(n: int = 100, spacing: float = 0.5) -> np.ndarray:
    """Straight-line waypoints along the +x axis."""
    x = np.arange(n) * spacing
    y = np.zeros(n)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# BaseMetric is abstract
# ---------------------------------------------------------------------------


class TestBaseMetric:  # pylint: disable=too-few-public-methods
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseMetric()  # pylint: disable=abstract-class-instantiated


# ---------------------------------------------------------------------------
# LapTimeMetric
# ---------------------------------------------------------------------------


class TestLapTimeMetric:
    def test_basic_accumulation(self) -> None:
        m = LapTimeMetric()
        obs0 = _make_obs()
        m.on_reset(obs0)

        dt = 0.01
        for _ in range(100):
            m.on_step(obs0, Action(0.0, 1.0), dt)

        report = m.report()
        assert report["lap_time_s"] == pytest.approx(1.0, abs=1e-6)
        assert report["steps"] == 100.0
        assert report["collision"] == 0.0

    def test_collision_flag(self) -> None:
        m = LapTimeMetric()
        m.on_reset(_make_obs())
        m.on_step(_make_obs(collision=1.0), Action(0.0, 1.0), 0.01)
        assert m.report()["collision"] == 1.0

    def test_lap_count(self) -> None:
        m = LapTimeMetric()
        m.on_reset(_make_obs())
        m.on_step(_make_obs(lap_count=2.0), Action(0.0, 1.0), 0.01)
        assert m.report()["laps_completed"] == 2.0


# ---------------------------------------------------------------------------
# CrossTrackErrorMetric
# ---------------------------------------------------------------------------


class TestCrossTrackErrorMetric:
    def test_requires_waypoints(self) -> None:
        m = CrossTrackErrorMetric()
        with pytest.raises(ValueError, match="requires waypoints"):
            m.on_reset(_make_obs(), waypoints=None)

    def test_zero_error_on_path(self) -> None:
        wpts = _make_waypoints_line()
        m = CrossTrackErrorMetric()
        m.on_reset(_make_obs(), waypoints=wpts)

        for i in range(10):
            obs = _make_obs(x=i * 0.5, y=0.0)
            m.on_step(obs, Action(0.0, 1.0), 0.01)

        report = m.report()
        assert report["crosstrack_rmse_m"] == pytest.approx(0.0, abs=1e-3)
        assert report["crosstrack_max_m"] == pytest.approx(0.0, abs=1e-3)

    def test_nonzero_error_off_path(self) -> None:
        wpts = _make_waypoints_line()
        m = CrossTrackErrorMetric()
        m.on_reset(_make_obs(), waypoints=wpts)

        # Drive at y=0.5 (0.5 m lateral offset)
        for i in range(10):
            obs = _make_obs(x=i * 0.5, y=0.5)
            m.on_step(obs, Action(0.0, 1.0), 0.01)

        report = m.report()
        assert report["crosstrack_mean_m"] == pytest.approx(0.5, abs=0.01)
        assert report["crosstrack_std_m"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# HeadingErrorMetric
# ---------------------------------------------------------------------------


class TestHeadingErrorMetric:
    def test_requires_waypoints(self) -> None:
        m = HeadingErrorMetric()
        with pytest.raises(ValueError, match="requires waypoints"):
            m.on_reset(_make_obs(), waypoints=None)

    def test_zero_heading_error_aligned(self) -> None:
        wpts = _make_waypoints_line()
        m = HeadingErrorMetric()
        m.on_reset(_make_obs(), waypoints=wpts)

        # Heading theta=0 along +x axis matches waypoints
        for i in range(10):
            obs = _make_obs(x=i * 0.5, y=0.0, theta=0.0)
            m.on_step(obs, Action(0.0, 1.0), 0.01)

        report = m.report()
        assert report["heading_error_mean_deg"] == pytest.approx(0.0, abs=0.5)

    def test_nonzero_heading_error(self) -> None:
        wpts = _make_waypoints_line()
        m = HeadingErrorMetric()
        m.on_reset(_make_obs(), waypoints=wpts)

        # Heading at 45 degrees while waypoints go along +x
        for i in range(10):
            obs = _make_obs(x=i * 0.5, y=0.0, theta=math.pi / 4)
            m.on_step(obs, Action(0.0, 1.0), 0.01)

        report = m.report()
        assert report["heading_error_mean_deg"] == pytest.approx(45.0, abs=1.0)


# ---------------------------------------------------------------------------
# WallProximityMetric
# ---------------------------------------------------------------------------


class TestWallProximityMetric:
    def test_uniform_scan(self) -> None:
        m = WallProximityMetric()
        m.on_reset(_make_obs())

        scan = np.full(1080, 3.0)
        m.on_step(_make_obs(scan=scan), Action(0.0, 1.0), 0.01)

        report = m.report()
        assert report["wall_min_distance_m"] == pytest.approx(3.0, abs=0.1)

    def test_tracks_minimum(self) -> None:
        m = WallProximityMetric()
        m.on_reset(_make_obs())

        # Step 1: walls at 3 m
        m.on_step(_make_obs(scan=np.full(1080, 3.0)), Action(0.0, 1.0), 0.01)
        # Step 2: walls at 0.5 m
        m.on_step(_make_obs(scan=np.full(1080, 0.5)), Action(0.0, 1.0), 0.01)
        # Step 3: walls at 2 m
        m.on_step(_make_obs(scan=np.full(1080, 2.0)), Action(0.0, 1.0), 0.01)

        report = m.report()
        assert report["wall_min_distance_m"] == pytest.approx(0.5, abs=0.1)


# ---------------------------------------------------------------------------
# SmoothnessMetric
# ---------------------------------------------------------------------------


class TestSmoothnessMetric:
    def test_constant_steering_zero_rate(self) -> None:
        m = SmoothnessMetric()
        m.on_reset(_make_obs())

        for _ in range(10):
            m.on_step(_make_obs(), Action(0.1, 1.0), 0.01)

        report = m.report()
        assert report["steering_rate_mean_rad_s"] == pytest.approx(0.0, abs=1e-6)

    def test_changing_steering(self) -> None:
        m = SmoothnessMetric()
        m.on_reset(_make_obs())
        dt = 0.01

        # Alternating steering
        m.on_step(_make_obs(), Action(0.0, 1.0), dt)
        m.on_step(_make_obs(), Action(0.1, 1.0), dt)

        report = m.report()
        expected_rate = 0.1 / dt  # 10 rad/s
        assert report["steering_rate_mean_rad_s"] == pytest.approx(
            expected_rate, abs=0.1
        )


# ---------------------------------------------------------------------------
# SpeedMetric
# ---------------------------------------------------------------------------


class TestSpeedMetric:
    def test_constant_speed(self) -> None:
        m = SpeedMetric()
        m.on_reset(_make_obs())

        for _ in range(10):
            m.on_step(_make_obs(vx=5.0), Action(0.0, 5.0), 0.01)

        report = m.report()
        assert report["speed_mean_m_s"] == pytest.approx(5.0, abs=1e-6)
        assert report["speed_std_m_s"] == pytest.approx(0.0, abs=1e-6)

    def test_varying_speed(self) -> None:
        m = SpeedMetric()
        m.on_reset(_make_obs())

        m.on_step(_make_obs(vx=2.0), Action(0.0, 2.0), 0.01)
        m.on_step(_make_obs(vx=4.0), Action(0.0, 4.0), 0.01)

        report = m.report()
        assert report["speed_mean_m_s"] == pytest.approx(3.0, abs=1e-6)
        assert report["speed_max_m_s"] == pytest.approx(4.0, abs=1e-6)


# ---------------------------------------------------------------------------
# MetricAggregator
# ---------------------------------------------------------------------------


class TestMetricAggregator:
    def test_create_default_with_waypoints(self) -> None:
        wpts = _make_waypoints_line()
        agg = MetricAggregator.create_default(waypoints=wpts)
        # Should contain all 6 metrics
        assert agg.metric_count == 6

    def test_create_default_without_waypoints(self) -> None:
        agg = MetricAggregator.create_default(waypoints=None)
        # Should contain 4 metrics (no cross-track, no heading)
        assert agg.metric_count == 4

    def test_full_lifecycle(self) -> None:
        wpts = _make_waypoints_line()
        agg = MetricAggregator.create_default(waypoints=wpts)
        obs0 = _make_obs()
        agg.on_reset(obs0, waypoints=wpts)

        for i in range(20):
            obs = _make_obs(x=i * 0.5, y=0.0, vx=3.0)
            agg.on_step(obs, Action(0.0, 3.0), 0.01)

        report = agg.report()

        # Check that keys from all 6 metrics are present
        assert "lap_time_s" in report
        assert "crosstrack_rmse_m" in report
        assert "heading_error_mean_deg" in report
        assert "wall_min_distance_m" in report
        assert "steering_rate_mean_rad_s" in report
        assert "speed_mean_m_s" in report

    def test_report_without_waypoints(self) -> None:
        agg = MetricAggregator.create_default(waypoints=None)
        obs0 = _make_obs()
        agg.on_reset(obs0)

        for _ in range(5):
            agg.on_step(obs0, Action(0.0, 1.0), 0.01)

        report = agg.report()
        assert "lap_time_s" in report
        assert "crosstrack_rmse_m" not in report
        assert "speed_mean_m_s" in report
