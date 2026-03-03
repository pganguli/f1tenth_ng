"""
Unit tests for planning utility functions.
"""

import os

import numpy as np
import pytest

from f110_planning.utils import get_vehicle_state, load_waypoints, nearest_point


def test_load_waypoints() -> None:
    """Test loading waypoints from a TSV file."""
    # Test loading a real file if it exists
    path = "data/maps/F1/Oschersleben/Oschersleben_centerline.tsv"
    if os.path.exists(path):
        waypoints = load_waypoints(path)
        assert waypoints.ndim == 2
        assert waypoints.shape[1] >= 2


def test_get_vehicle_state() -> None:
    """Test extracting vehicle state from observation dictionary."""
    obs = {
        "poses_x": np.array([1.0, 2.0]),
        "poses_y": np.array([3.0, 4.0]),
        "poses_theta": np.array([0.5, 0.6]),
        "linear_vels_x": np.array([10.0, 11.0]),
        "linear_vels_y": np.array([0.0, 0.0]),
        "ang_vels_z": np.array([0.1, 0.2]),
    }
    state = get_vehicle_state(obs, ego_idx=1)
    # Expected [x, y, theta, v, ...] depends on implementation but usually starts with x, y, theta
    assert state[0] == 2.0
    assert state[1] == 4.0
    assert state[2] == 0.6


def test_nearest_point() -> None:
    """Test finding the nearest point on a path."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    point = np.array([0.5, 0.1])
    p, _, _, i = nearest_point(point, path)
    assert np.allclose(p, [0.5, 0.0])
    assert i == 0  # nearest segment starts at index 0


def test_nearest_point_with_duplicate() -> None:
    """Zero-length segments should be ignored and not raise an error."""
    # insert a duplicate waypoint in the middle of the path
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    point = np.array([0.5, 0.1])
    # this would previously trigger a SystemError inside the njit
    p, _, _, i = nearest_point(point, path)
    assert np.allclose(p, [0.5, 0.0])
    assert i == 0


def test_nearest_point_endpoint_clamping() -> None:
    """A query point far past the last waypoint should project onto the last segment end."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    point = np.array([100.0, 0.0])  # far beyond path end
    p, dist, t, _ = nearest_point(point, path)
    # The projection parameter t is clamped to 1.0 on the last segment, so the
    # projected point should be the last waypoint [2.0, 0.0].
    assert np.allclose(p, [2.0, 0.0]), f"Expected [2, 0] but got {p}"
    assert dist == pytest.approx(98.0, abs=1e-6)


def test_get_vehicle_state_includes_velocity() -> None:
    """get_vehicle_state should also extract linear_vels_x (index 3)."""
    obs = {
        "poses_x": np.array([5.0, 6.0]),
        "poses_y": np.array([7.0, 8.0]),
        "poses_theta": np.array([1.0, 2.0]),
        "linear_vels_x": np.array([3.5, 9.1]),
        "linear_vels_y": np.array([0.0, 0.0]),
        "ang_vels_z": np.array([0.0, 0.0]),
    }
    state = get_vehicle_state(obs, ego_idx=0)
    assert state[3] == pytest.approx(3.5), f"velocity at index 3 should be 3.5, got {state[3]}"

    state1 = get_vehicle_state(obs, ego_idx=1)
    assert state1[3] == pytest.approx(9.1)


def test_load_waypoints_bad_path() -> None:
    """Passing a non-existent file should raise FileNotFoundError or return an empty array."""
    try:
        result = load_waypoints("/nonexistent/path/to/waypoints.tsv")
        # If a fallback is implemented it must be either None or an empty array
        assert result is None or (hasattr(result, "size") and result.size == 0)
    except (FileNotFoundError, OSError):
        pass  # also acceptable behaviour
