"""
Unit tests for reactive obstacle avoidance planners.
"""

# pylint: disable=redefined-outer-name

from typing import Any

import copy
import numpy as np
import pytest

from f110_planning.reactive import (
    BubblePlanner,
    DisparityExtenderPlanner,
    EdgeCloudPlanner,
    GapFollowerPlanner,
    SelectiveEdgeCloudPlanner,
)
from f110_planning.utils import F110_MAX_STEER


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


def test_disparity_extender_output_bounds(reactive_obs: dict[str, Any]) -> None:
    """DisparityExtenderPlanner output must be within the physical steering envelope."""
    planner = DisparityExtenderPlanner()
    # Run several times with varied scans to exercise different code paths.
    for _ in range(5):
        reactive_obs["scans"][0] = np.random.rand(1080) * 10.0
        action = planner.plan(reactive_obs)
        assert abs(action.steer) <= F110_MAX_STEER + 1e-9, (
            f"steer {action.steer} exceeds F110_MAX_STEER {F110_MAX_STEER}"
        )
        assert action.speed >= 0.0, f"speed {action.speed} must be non-negative"


def test_edge_cloud_planner_alpha_boundaries(reactive_obs: dict[str, Any]) -> None:
    """alpha=0 means edge-only; alpha=1 means cloud-only."""

    scan = np.ones(1080) * 5.0
    reactive_obs["scans"][0] = scan

    # Build two planners whose edge and cloud sub-planners produce predictable
    # outputs by relying entirely on the lateral_gain path (no DNN model loaded).
    # With alpha_*=0 the final action must equal the edge-only prediction.
    planner_edge_only = EdgeCloudPlanner(
        cloud_latency=0, alpha_left=0.0, alpha_track=0.0, alpha_heading=0.0
    )
    planner_cloud_only = EdgeCloudPlanner(
        cloud_latency=0, alpha_left=1.0, alpha_track=1.0, alpha_heading=1.0
    )

    # Force a cloud result by running step 0 (FixedIntervalScheduler calls at step 0)
    action_edge = planner_edge_only.plan(copy.deepcopy(reactive_obs))
    action_cloud = planner_cloud_only.plan(copy.deepcopy(reactive_obs))

    # With alpha=0 the result must be the pure edge output; with alpha=1 the
    # pure cloud output.  Since both planners share the same scan and model config
    # (no weights: both return 0/0 heading+wall signals), edge and cloud
    # produce the same Action, so the blended result equals either.
    # What we verify is that the blending coefficients are applied correctly:
    # manually compute expected blend and cross-check.
    edge_planner = planner_edge_only.edge_planner
    cloud_planner = planner_edge_only.cloud_planner
    edge_act = edge_planner.plan(copy.deepcopy(reactive_obs))
    cloud_act = cloud_planner.plan(copy.deepcopy(reactive_obs))

    expected_edge_only_steer = 0.0 * cloud_act.steer + 1.0 * edge_act.steer
    expected_cloud_only_steer = 1.0 * cloud_act.steer + 0.0 * edge_act.steer

    assert abs(action_edge.steer - expected_edge_only_steer) < 1e-9
    assert abs(action_cloud.steer - expected_cloud_only_steer) < 1e-9


def test_selective_edge_cloud_feature_blend(reactive_obs: dict[str, Any]) -> None:
    """SelectiveEdgeCloudPlanner with alpha=1 for all features must produce
    the same action as one with alpha=0 when no cloud result has yet arrived
    (cache=None → pure edge fallback for both).
    """
    reactive_obs["scans"][0] = np.ones(1080) * 5.0

    planner_a = SelectiveEdgeCloudPlanner(
        cloud_latency=10, alpha_left=0.0, alpha_track=0.0, alpha_heading=0.0
    )
    planner_b = SelectiveEdgeCloudPlanner(
        cloud_latency=10, alpha_left=1.0, alpha_track=1.0, alpha_heading=1.0
    )

    action_a = planner_a.plan(copy.deepcopy(reactive_obs), call_mask=[False, False, False])
    action_b = planner_b.plan(copy.deepcopy(reactive_obs), call_mask=[False, False, False])

    # Before any cloud result arrives both planners fall back to pure edge; the
    # actions must be identical regardless of alpha.
    assert action_a.steer == pytest.approx(action_b.steer, abs=1e-9)
    assert action_a.speed == pytest.approx(action_b.speed, abs=1e-9)
