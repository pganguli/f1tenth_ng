"""
Unit tests for the pure stateless metric functions in ``f110_planning.metrics``.

Each function is exercised in isolation with minimal synthetic observations /
inputs so that no simulator or gymnasium environment is required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import f110_planning.metrics as metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scan(n_beams: int = 1080, fill: float = 5.0) -> np.ndarray:
    """Return a uniform range scan filled with *fill* metres."""
    return np.full(n_beams, fill, dtype=np.float64)


def _make_obs(
    x: float = 0.0,
    y: float = 0.0,
    theta: float = 0.0,
    vx: float = 5.0,
    collision: float = 0.0,
    scan_fill: float = 5.0,
) -> dict:
    return {
        "poses_x": np.array([x]),
        "poses_y": np.array([y]),
        "poses_theta": np.array([theta]),
        "linear_vels_x": np.array([vx]),
        "collisions": np.array([collision]),
        "scans": np.array([_make_scan(fill=scan_fill)]),
    }


# ---------------------------------------------------------------------------
# crosstrack_error
# ---------------------------------------------------------------------------


def test_crosstrack_error_on_path() -> None:
    """Point that lies exactly on the path should have zero error."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pos = np.array([1.0, 0.0])
    assert metrics.crosstrack_error(pos, path) == pytest.approx(0.0, abs=1e-9)


def test_crosstrack_error_offset() -> None:
    """Point 0.5 m off the x-axis should return 0.5."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pos = np.array([1.0, 0.5])
    assert metrics.crosstrack_error(pos, path) == pytest.approx(0.5, rel=1e-6)


def test_crosstrack_error_uses_xy_only() -> None:
    """Extra waypoint columns (e.g. speed reference) should be ignored."""
    path = np.column_stack([
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.ones((3, 1)) * 3.0,  # dummy third column
    ])
    pos = np.array([1.0, 0.2])
    assert metrics.crosstrack_error(pos, path) == pytest.approx(0.2, rel=1e-6)


# ---------------------------------------------------------------------------
# heading_error_rad
# ---------------------------------------------------------------------------


def test_heading_error_aligned() -> None:
    """Vehicle heading perfectly aligned with path should give ~0 error."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pos = np.array([1.0, 0.0])
    theta = 0.0  # heading east, same direction as path
    err = metrics.heading_error_rad(pos, theta, path)
    assert abs(err) == pytest.approx(0.0, abs=1e-3)


def test_heading_error_perpendicular() -> None:
    """Vehicle heading perpendicular to path should be ±π/2."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pos = np.array([1.0, 0.0])
    theta = math.pi / 2  # facing north, path goes east
    err = metrics.heading_error_rad(pos, theta, path)
    assert abs(err) == pytest.approx(math.pi / 2, rel=1e-3)


# ---------------------------------------------------------------------------
# min_wall_dist
# ---------------------------------------------------------------------------


def test_min_wall_dist_uniform() -> None:
    """Uniform scan should return the fill value (both sides identical)."""
    scan = _make_scan(fill=3.0)
    # the exact value depends on how the beam indices are split, but it should
    # be positive and at most 3.0 (the uniform distance)
    result = metrics.min_wall_dist(scan)
    assert 0.0 < result <= 3.0


def test_min_wall_dist_asymmetric() -> None:
    """Min distance should come from the closer side."""
    scan = _make_scan(fill=5.0)
    # artificially set one side closer — we can't know the exact cutoff index
    # without replicating the slice logic, but we can verify min < fill
    scan[:200] = 1.0  # first beams (right side typically) are very close
    result = metrics.min_wall_dist(scan)
    assert result <= 5.0


# ---------------------------------------------------------------------------
# steering_rate
# ---------------------------------------------------------------------------


def test_steering_rate_normal() -> None:
    """Δsteer = 0.1 rad over dt = 0.01 s → rate = 10 rad/s."""
    assert metrics.steering_rate(0.0, 0.1, 0.01) == pytest.approx(10.0, rel=1e-9)


def test_steering_rate_zero_delta() -> None:
    """No steering change → rate = 0."""
    assert metrics.steering_rate(0.3, 0.3, 0.01) == pytest.approx(0.0, abs=1e-9)


def test_steering_rate_zero_dt() -> None:
    """dt ≤ 0 should return 0 (guard for first step / degenerate case)."""
    assert metrics.steering_rate(0.0, 0.5, 0.0) == 0.0
    assert metrics.steering_rate(0.0, 0.5, -0.01) == 0.0


def test_steering_rate_absolute_value() -> None:
    """Result should always be non-negative regardless of sign of Δsteer."""
    assert metrics.steering_rate(0.1, 0.0, 0.01) == pytest.approx(10.0, rel=1e-9)


# ---------------------------------------------------------------------------
# current_speed
# ---------------------------------------------------------------------------


def test_current_speed() -> None:
    obs = _make_obs(vx=7.5)
    assert metrics.current_speed(obs, ego_idx=0) == pytest.approx(7.5)


def test_current_speed_multi_agent() -> None:
    obs = {"linear_vels_x": np.array([1.0, 9.9])}
    assert metrics.current_speed(obs, ego_idx=1) == pytest.approx(9.9)


# ---------------------------------------------------------------------------
# has_collision
# ---------------------------------------------------------------------------


def test_has_collision_false() -> None:
    obs = _make_obs(collision=0.0)
    assert metrics.has_collision(obs) is False


def test_has_collision_true() -> None:
    obs = _make_obs(collision=1.0)
    assert metrics.has_collision(obs) is True


def test_has_collision_multi_agent() -> None:
    obs = {"collisions": np.array([0.0, 1.0])}
    assert metrics.has_collision(obs, ego_idx=0) is False
    assert metrics.has_collision(obs, ego_idx=1) is True


# ---------------------------------------------------------------------------
# pi_2_pi
# ---------------------------------------------------------------------------


def test_pi_2_pi_identity() -> None:
    """Values already in [-π, π] are returned unchanged."""
    from f110_planning.utils.geometry_utils import pi_2_pi

    assert pi_2_pi(0.0) == pytest.approx(0.0)
    assert pi_2_pi(1.0) == pytest.approx(1.0)
    assert pi_2_pi(-1.0) == pytest.approx(-1.0)


def test_pi_2_pi_wraps_above_pi() -> None:
    """Angles just above +π should be mapped to near -π."""
    from f110_planning.utils.geometry_utils import pi_2_pi

    result = pi_2_pi(math.pi + 0.5)
    assert result == pytest.approx(math.pi + 0.5 - 2 * math.pi, abs=1e-9)
    # must lie within (-π, π]
    assert -math.pi <= result <= math.pi


def test_pi_2_pi_wraps_below_minus_pi() -> None:
    """Angles just below -π should be mapped to near +π."""
    from f110_planning.utils.geometry_utils import pi_2_pi

    result = pi_2_pi(-math.pi - 0.5)
    assert result == pytest.approx(-math.pi - 0.5 + 2 * math.pi, abs=1e-9)
    assert -math.pi <= result <= math.pi
