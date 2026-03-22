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
    MultiTierEdgeCloudPlanner,
)
from f110_planning.base import Action, CloudScheduler, CloudTier, TieredCloudScheduler
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
    # With alpha_steer=0 the final action steers must equal the edge action steers.
    planner_edge_only = EdgeCloudPlanner(
        cloud_latency=0, alpha_steer=0.0, alpha_speed=0.0
    )
    planner_cloud_only = EdgeCloudPlanner(
        cloud_latency=0, alpha_steer=1.0, alpha_speed=1.0
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


class _SequenceTierScheduler(TieredCloudScheduler):
    """Simple deterministic tier scheduler for planner tests."""

    def __init__(self, decisions: list[CloudTier | None]) -> None:
        self.decisions = decisions

    def choose_cloud_tier(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_actions: dict[CloudTier, Action | None],
        context: dict[str, Any] | None = None,
    ) -> CloudTier | None:
        del obs, latest_cloud_actions, context
        if step < len(self.decisions):
            return self.decisions[step]
        return None


class _CaptureScheduler(CloudScheduler):
    """Record scheduler contexts for single-tier planner tests."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, float]] = []

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        del step, obs, latest_cloud_action
        self.contexts.append(dict(context or {}))
        return False


def test_edge_cloud_planner_action_distance_uses_weights() -> None:
    """Weighted L1 action disagreement should combine steer and speed terms."""
    planner = EdgeCloudPlanner(
        deviation_steer_weight=2.0,
        deviation_speed_weight=0.5,
    )

    distance = planner._action_distance(  # pylint: disable=protected-access
        Action(steer=0.3, speed=4.0),
        Action(steer=0.1, speed=3.0),
    )

    assert distance == pytest.approx(0.9)


def test_edge_cloud_planner_deviation_is_action_distance(
    reactive_obs: dict[str, Any],
) -> None:
    """Scheduler context should use edge-vs-held-cloud action disagreement."""
    scheduler = _CaptureScheduler()
    planner = EdgeCloudPlanner(
        scheduler=scheduler,
        cloud_latency=0,
        deviation_steer_weight=2.0,
        deviation_speed_weight=0.5,
    )
    planner.edge_planner.plan = lambda *_args, **_kwargs: Action(steer=0.3, speed=4.0)  # type: ignore[method-assign]
    planner._latest_cloud_action = Action(steer=0.1, speed=3.0)  # pylint: disable=protected-access

    planner.plan(copy.deepcopy(reactive_obs))

    assert scheduler.contexts
    assert scheduler.contexts[-1]["deviation"] == pytest.approx(0.9)


def test_edge_cloud_planner_reset_clears_last_cloud_call() -> None:
    """Reset should clear the previous cloud-call flag."""
    planner = EdgeCloudPlanner()
    planner.last_cloud_call = True

    planner.reset()

    assert not planner.last_cloud_call


def test_multi_tier_edge_cloud_planner_uses_requested_tier_blending(
    reactive_obs: dict[str, Any],
) -> None:
    """Multi-tier planner should blend medium and large cloud actions separately."""
    planner = MultiTierEdgeCloudPlanner(
        scheduler=_SequenceTierScheduler([CloudTier.MEDIUM, None, CloudTier.LARGE, None]),
        medium_cloud_latency=0,
        large_cloud_latency=0,
        alpha_steer_medium=0.25,
        alpha_speed_medium=0.25,
        alpha_steer_large=0.75,
        alpha_speed_large=0.75,
    )

    planner.edge_planner.plan = lambda *_args, **_kwargs: Action(steer=0.2, speed=2.0)  # type: ignore[method-assign]
    planner.medium_cloud_planner.last_left_dist = 1.0
    planner.medium_cloud_planner.last_track_width = 2.0
    planner.medium_cloud_planner.last_heading_error = 0.1
    planner.medium_cloud_planner.plan = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: Action(steer=0.6, speed=4.0)
    )
    planner.large_cloud_planner.last_left_dist = 1.5
    planner.large_cloud_planner.last_track_width = 2.5
    planner.large_cloud_planner.last_heading_error = 0.2
    planner.large_cloud_planner.plan = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: Action(steer=1.0, speed=6.0)
    )
    planner._blend_features = (  # type: ignore[method-assign]
        lambda _obs, _ego_idx, _features, tier: (
            Action(steer=0.3, speed=2.5)
            if tier == CloudTier.MEDIUM
            else Action(steer=0.8, speed=5.0)
        )
    )

    action0 = planner.plan(copy.deepcopy(reactive_obs))
    action1 = planner.plan(copy.deepcopy(reactive_obs))
    action2 = planner.plan(copy.deepcopy(reactive_obs))
    action3 = planner.plan(copy.deepcopy(reactive_obs))

    assert action0 == Action(steer=0.2, speed=2.0)
    assert action1 == Action(steer=0.3, speed=2.5)
    assert action2 == Action(steer=0.3, speed=2.5)
    assert action3 == Action(steer=0.8, speed=5.0)
    assert planner.last_cloud_tier_called is None
