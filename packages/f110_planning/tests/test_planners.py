"""
Unit tests for waypoint tracking planners.
"""

# pylint: disable=redefined-outer-name

from typing import Any

import numpy as np
import pytest

from f110_planning.base import Action
from f110_planning.tracking import LQRPlanner, PurePursuitPlanner, StanleyPlanner


@pytest.fixture
def dummy_waypoints() -> np.ndarray:
    """Provides a set of dummy waypoints forming a straight line."""
    # Straight line along x-axis
    return np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [2.0, 0.0, 5.0],
            [3.0, 0.0, 5.0],
        ]
    )


def test_pure_pursuit_init(dummy_waypoints: np.ndarray) -> None:
    """Test initialization of the Pure Pursuit planner."""
    planner = PurePursuitPlanner(waypoints=dummy_waypoints)
    assert planner.wheelbase == 0.33
    assert len(planner.waypoints) == 4


def test_pure_pursuit_plan(dummy_waypoints: np.ndarray, dummy_obs: dict[str, Any]) -> None:
    """Test that Pure Pursuit generates correct steering commands."""
    planner = PurePursuitPlanner(waypoints=dummy_waypoints)
    action = planner.plan(dummy_obs)
    assert isinstance(action, Action)
    # On a straight line along X, with car at (0,0) facing X, steering should be ~0
    assert abs(action.steer) < 1e-2
    assert action.speed > 0.0

    # If car is at (0, 0.5) facing X, it should steer right (negative angle) to return to path y=0
    dummy_obs["poses_y"] = np.array([0.5])
    action = planner.plan(dummy_obs)
    assert action.steer < 0.0


def test_lqr_convergence(dummy_waypoints: np.ndarray, dummy_obs: dict[str, Any]) -> None:
    """Test that LQR correctly steers toward the path."""
    planner = LQRPlanner(waypoints=dummy_waypoints)
    # Start with lateral error
    dummy_obs["poses_y"] = np.array([0.2])
    action = planner.plan(dummy_obs)
    # LQR should steer to correct error
    assert action.steer < 0.0


def test_stanley_convergence(dummy_waypoints: np.ndarray, dummy_obs: dict[str, Any]) -> None:
    """Test that Stanley correctly steers toward the path."""
    planner = StanleyPlanner(waypoints=dummy_waypoints)
    # Start with lateral error
    dummy_obs["poses_y"] = np.array([0.2])
    action = planner.plan(dummy_obs)
    # Stanley should steer to correct error
    assert action.steer < 0.0
