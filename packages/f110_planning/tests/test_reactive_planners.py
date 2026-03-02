"""
Unit tests for reactive obstacle avoidance planners.
"""

# pylint: disable=redefined-outer-name

from typing import Any

import numpy as np
import pytest

from f110_planning.reactive import (
    BubblePlanner,
    DisparityExtenderPlanner,
    GapFollowerPlanner,
)


@pytest.fixture
def reactive_obs(dummy_obs: dict[str, Any]) -> dict[str, Any]:
    """Provides a dummy observation with LiDAR scans."""
    obs = dummy_obs.copy()
    obs["scans"] = np.random.rand(1, 1080)
    return obs


def test_gap_follower_with_obstacle(reactive_obs: dict[str, Any]) -> None:
    """Test that Gap Follower steers away from a nearby obstacle."""
    planner = GapFollowerPlanner()

    # Create a scan where everything is far (10m) except for a block on the left
    scan = np.ones(1080) * 10.0
    # Left is roughly indices 700-1080 for a 270 FOV (center is 540)
    # Let's put a close obstacle on the left (e.g., 0.5m)
    scan[700:900] = 0.5
    reactive_obs["scans"][0] = scan

    action = planner.plan(reactive_obs)
    # Should steer right (negative angle)
    assert action.steer < 0.0


def test_bubble_planner_safety(reactive_obs: dict[str, Any]) -> None:
    """Test that Bubble Planner steers away from nearby obstacles."""
    planner = BubblePlanner(safety_radius=1.0)
    # Obstacle slightly to the right of center
    scan = np.ones(1080) * 10.0
    scan[400:450] = 0.5
    reactive_obs["scans"][0] = scan

    action = planner.plan(reactive_obs)
    # Should steer away from the right-side obstacle (steer > 0)
    assert action.steer > 0.1


def test_disparity_extender_planner(reactive_obs: dict[str, Any]) -> None:
    """Test initialization and basic planning of Disparity Extender."""
    planner = DisparityExtenderPlanner()
    action = planner.plan(reactive_obs)
    assert hasattr(action, "steer")
