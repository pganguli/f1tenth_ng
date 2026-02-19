"""
Unit tests for waypoint data generation script.
"""

from unittest.mock import MagicMock

import numpy as np
from f110_planning.tracking import PurePursuitPlanner

from scripts.datagen.waypoint_datagen import (
    _gather_step_data,
    create_planner,
    _apply_steering_noise,
)


def test_create_planner() -> None:
    """Test creation of tracking planners from strings."""
    waypoints = np.array([[0, 0], [1, 0]])
    planner = create_planner("pure_pursuit", waypoints)
    assert isinstance(planner, PurePursuitPlanner)
    assert np.all(planner.waypoints == waypoints)


def test_gather_step_data() -> None:
    """Test extraction of training features (scans, distances, errors) from observations."""
    obs = {
        "scans": [np.ones(1080) * 5.0],
        "poses_x": [0.0],
        "poses_y": [0.0],
        "poses_theta": [0.0],
    }
    waypoints = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)

    scan, l_dist, r_dist, h_err = _gather_step_data(obs, waypoints)

    assert scan.shape == (1080,)
    assert l_dist > 0
    assert r_dist > 0
    # On a straight line facing the right direction, heading error should be 0
    assert abs(h_err) < 1e-3


def test_apply_steering_noise() -> None:
    """Test application of random steering noise for data augmentation."""
    # pylint: disable=redefined-outer-name
    args = MagicMock()
    args.steering_noise = 0.1
    args.drift_prob = 0.0

    steer, active, _ = _apply_steering_noise(0.0, 0, 0.0, args)
    assert active == 0
    assert abs(steer) > 0.0  # mostly non-zero due to noise
