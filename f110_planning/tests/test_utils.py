"""
Unit tests for planning utility functions.
"""

import os

import numpy as np

from f110_planning.utils import get_vehicle_state, load_waypoints, nearest_point


def test_load_waypoints():
    """Test loading waypoints from a TSV file."""
    # Test loading a real file if it exists
    path = "data/maps/F1/Oschersleben/Oschersleben_centerline.tsv"
    if os.path.exists(path):
        waypoints = load_waypoints(path)
        assert waypoints.ndim == 2
        assert waypoints.shape[1] >= 2


def test_get_vehicle_state():
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


def test_nearest_point():
    """Test finding the nearest point on a path."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    point = np.array([0.5, 0.1])
    p, _, _, i = nearest_point(point, path)
    assert np.allclose(p, [0.5, 0.0])
    assert i == 0  # nearest segment starts at index 0
