"""Unit tests for waypoint data generation script."""

from unittest.mock import MagicMock

import numpy as np

from f110_scripts.datagen.waypoint_datagen import (
    _apply_steering_noise,
    _gather_step_data,
    create_planner,
)


def test_create_planner_stores_waypoints() -> None:
    """create_planner should embed the supplied waypoints into the planner."""
    waypoints = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    planner = create_planner("pure_pursuit", waypoints)
    # The planner should remember the waypoints it was given
    assert np.allclose(planner.waypoints[:, :2], waypoints)


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


def test_apply_steering_noise_with_drift() -> None:
    """With drift_prob=1.0 the drift branch is activated every call."""
    args = MagicMock()
    args.steering_noise = 0.0   # no Gaussian noise so only drift counts
    args.drift_prob = 1.0       # always trigger drift
    args.drift_magnitude = 0.3  # correct attribute name used by _apply_steering_noise

    # Call several times from a known initial state; at least some should
    # produce a non-zero steer (the drift accumulates while active==1).
    results = []
    active = 0
    steer = 0.0
    for _ in range(10):
        steer, active, _ = _apply_steering_noise(steer, active, 0.0, args)
        results.append(steer)

    # After multiple calls with drift always on, the steering offset must
    # have been non-zero at some point.
    assert any(abs(np.asarray(s).item()) > 0.0 for s in results), (
        "Expected non-zero steer values when drift_prob=1.0"
    )
