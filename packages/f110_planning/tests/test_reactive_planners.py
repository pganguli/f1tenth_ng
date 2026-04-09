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
        self.contexts: list[dict[str, Any]] = []

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


class _SequenceCallScheduler(CloudScheduler):
    """Record contexts while returning a fixed sequence of cloud decisions."""

    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = decisions
        self.contexts: list[dict[str, Any]] = []

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        del step, obs, latest_cloud_action
        self.contexts.append(dict(context or {}))
        index = len(self.contexts) - 1
        if index < len(self.decisions):
            return self.decisions[index]
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


def test_edge_cloud_planner_context_exposes_edge_and_cloud_runtime_state(
    reactive_obs: dict[str, Any],
) -> None:
    """Scheduler context should include edge features/action and cloud state."""
    scheduler = _CaptureScheduler()
    planner = EdgeCloudPlanner(
        scheduler=scheduler,
        cloud_latency=5,
    )

    def edge_plan(*_args, **_kwargs) -> Action:
        planner.edge_planner.last_left_dist = 1.0
        planner.edge_planner.last_track_width = 2.0
        planner.edge_planner.last_heading_error = 0.3
        return Action(steer=0.2, speed=3.0)

    planner.edge_planner.plan = edge_plan  # type: ignore[method-assign]

    planner.plan(copy.deepcopy(reactive_obs))

    assert scheduler.contexts
    context = scheduler.contexts[-1]
    assert context["edge_features"] == pytest.approx((1.0, 2.0, 0.3))
    assert context["edge_action"] == pytest.approx((0.2, 3.0))
    assert context["cloud_age"] == 999
    assert context["cloud_in_flight"] is False
    assert context["cloud_queue_depth"] == 0
    assert context["cloud_last_updated_step"] == -1


def test_edge_cloud_planner_reset_clears_last_cloud_call() -> None:
    """Reset should clear the previous cloud-call flag."""
    planner = EdgeCloudPlanner()
    planner.last_cloud_call = True

    planner.reset()

    assert not planner.last_cloud_call


def test_edge_cloud_planner_latency_zero_uses_same_step_cloud_features(
    reactive_obs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-latency cloud calls should be consumed on the same planning step."""
    planner = EdgeCloudPlanner(cloud_latency=0, alpha_steer=1.0, alpha_speed=1.0)

    def edge_plan(*_args, **_kwargs) -> Action:
        planner.edge_planner.last_left_dist = 1.0
        planner.edge_planner.last_track_width = 2.0
        planner.edge_planner.last_heading_error = 0.1
        return Action(steer=0.0, speed=1.0)

    def cloud_plan(*_args, **_kwargs) -> Action:
        planner.cloud_planner.last_left_dist = 4.0
        planner.cloud_planner.last_track_width = 6.0
        planner.cloud_planner.last_heading_error = 0.3
        return Action(steer=0.4, speed=3.0)

    planner.edge_planner.plan = edge_plan  # type: ignore[method-assign]
    planner.cloud_planner.plan = cloud_plan  # type: ignore[method-assign]

    def fake_reactive_action(*_args, **kwargs) -> Action:
        assert kwargs["left_dist"] == pytest.approx(4.0)
        assert kwargs["right_dist"] == pytest.approx(2.0)
        assert kwargs["heading_error"] == pytest.approx(0.3)
        return Action(steer=0.25, speed=2.5)

    monkeypatch.setattr(
        "f110_planning.reactive.edge_cloud_planner.get_reactive_action",
        fake_reactive_action,
    )

    action = planner.plan(copy.deepcopy(reactive_obs))

    assert planner.last_cloud_call
    assert planner._latest_cloud_features == pytest.approx((4.0, 6.0, 0.3))  # pylint: disable=protected-access
    assert action.steer == pytest.approx(0.25)
    assert action.speed == pytest.approx(2.5)


def test_edge_cloud_planner_alpha_age_decreases_with_age() -> None:
    """Sigma-based age decay should reduce cloud weight as held data gets older."""
    se2, sc2, sp2 = 0.011366, 0.000989, 0.044961**2
    alpha_0 = EdgeCloudPlanner._alpha_age(0, se2, sc2, sp2)  # pylint: disable=protected-access
    alpha_5 = EdgeCloudPlanner._alpha_age(5, se2, sc2, sp2)  # pylint: disable=protected-access
    alpha_20 = EdgeCloudPlanner._alpha_age(20, se2, sc2, sp2)  # pylint: disable=protected-access

    assert alpha_0 > alpha_5 > alpha_20


def test_edge_cloud_planner_age_zero_keeps_static_feature_alphas() -> None:
    """Anchored decay should preserve the tuned static alpha at cloud age zero."""
    planner = EdgeCloudPlanner(
        alpha_left=0.2,
        alpha_track=0.2,
        alpha_heading=0.7,
        sigma_proc_left=0.044961,
        sigma_proc_track=0.067937,
        sigma_proc_heading=0.033182,
        age_decay_lambda=4.0,
    )

    assert planner._resolved_alphas(0) == pytest.approx((0.2, 0.2, 0.7))  # pylint: disable=protected-access


def test_edge_cloud_planner_age_decay_lambda_zero_keeps_static_behavior() -> None:
    """Disabling anchored decay should return the static feature alphas for any age."""
    planner = EdgeCloudPlanner(
        alpha_left=0.2,
        alpha_track=0.2,
        alpha_heading=0.7,
        sigma_proc_left=0.044961,
        sigma_proc_track=0.067937,
        sigma_proc_heading=0.033182,
        age_decay_lambda=0.0,
    )

    assert planner._resolved_alphas(5) == pytest.approx((0.2, 0.2, 0.7))  # pylint: disable=protected-access


def test_edge_cloud_planner_anchored_alpha_decreases_with_age_and_lambda() -> None:
    """Anchored decay should reduce cloud weights as age or lambda increases."""
    planner = EdgeCloudPlanner(
        alpha_left=0.2,
        alpha_track=0.2,
        alpha_heading=0.7,
        sigma_proc_left=0.044961,
        sigma_proc_track=0.067937,
        sigma_proc_heading=0.033182,
        age_decay_lambda=1.0,
    )
    low_age = planner._resolved_alphas(1)  # pylint: disable=protected-access
    high_age = planner._resolved_alphas(5)  # pylint: disable=protected-access

    faster_decay = EdgeCloudPlanner(
        alpha_left=0.2,
        alpha_track=0.2,
        alpha_heading=0.7,
        sigma_proc_left=0.044961,
        sigma_proc_track=0.067937,
        sigma_proc_heading=0.033182,
        age_decay_lambda=4.0,
    )._resolved_alphas(5)  # pylint: disable=protected-access

    assert high_age[0] < low_age[0] < 0.2
    assert high_age[1] < low_age[1] < 0.2
    assert high_age[2] < low_age[2] < 0.7
    assert faster_decay[0] < high_age[0]
    assert faster_decay[1] < high_age[1]
    assert faster_decay[2] < high_age[2]


def test_edge_cloud_planner_missing_sigma_proc_ignores_lambda() -> None:
    """Missing process-noise sigmas should disable anchored age decay."""
    planner = EdgeCloudPlanner(
        alpha_left=0.2,
        alpha_track=0.2,
        alpha_heading=0.7,
        sigma_proc_left=None,
        sigma_proc_track=None,
        sigma_proc_heading=None,
        age_decay_lambda=8.0,
    )

    assert planner._resolved_alphas(12) == pytest.approx((0.2, 0.2, 0.7))  # pylint: disable=protected-access


def test_edge_cloud_planner_latency_correction_uses_current_edge_features(
    reactive_obs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrived cloud features should be edge-delta corrected using current edge features."""
    planner = EdgeCloudPlanner(cloud_latency=1, alpha_steer=1.0, alpha_speed=1.0)

    edge_features = iter(((1.0, 2.0, 0.1), (1.5, 2.5, 0.15)))

    def edge_plan(*_args, **_kwargs) -> Action:
        left, track, heading = next(edge_features)
        planner.edge_planner.last_left_dist = left
        planner.edge_planner.last_track_width = track
        planner.edge_planner.last_heading_error = heading
        return Action(steer=0.0, speed=1.0)

    def cloud_plan(*_args, **_kwargs) -> Action:
        planner.cloud_planner.last_left_dist = 4.0
        planner.cloud_planner.last_track_width = 6.0
        planner.cloud_planner.last_heading_error = 0.3
        return Action(steer=0.4, speed=3.0)

    planner.edge_planner.plan = edge_plan  # type: ignore[method-assign]
    planner.cloud_planner.plan = cloud_plan  # type: ignore[method-assign]

    observed = []

    def fake_reactive_action(*_args, **kwargs) -> Action:
        observed.append(
            (
                kwargs["left_dist"],
                kwargs["right_dist"],
                kwargs["heading_error"],
            )
        )
        return Action(steer=0.25, speed=2.5)

    monkeypatch.setattr(
        "f110_planning.reactive.edge_cloud_planner.get_reactive_action",
        fake_reactive_action,
    )

    planner.plan(copy.deepcopy(reactive_obs))
    planner.plan(copy.deepcopy(reactive_obs))

    assert planner._latest_cloud_features == pytest.approx((4.5, 6.5, 0.35))  # pylint: disable=protected-access
    assert observed[-1][0] == pytest.approx(4.5)
    assert observed[-1][1] == pytest.approx(2.0)
    assert observed[-1][2] == pytest.approx(0.35)


def test_edge_cloud_planner_delayed_context_uses_final_returned_command(
    reactive_obs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delayed-cloud scheduler context should expose the final returned action stream."""
    scheduler = _SequenceCallScheduler([True, False])
    planner = EdgeCloudPlanner(
        scheduler=scheduler,
        cloud_latency=1,
        alpha_left=1.0,
        alpha_track=1.0,
        alpha_heading=1.0,
    )

    edge_features = iter(((1.0, 2.0, 0.1), (1.5, 2.5, 0.15)))

    def edge_plan(*_args, **_kwargs) -> Action:
        left, track, heading = next(edge_features)
        planner.edge_planner.last_left_dist = left
        planner.edge_planner.last_track_width = track
        planner.edge_planner.last_heading_error = heading
        return Action(steer=0.1, speed=1.0)

    def cloud_plan(*_args, **_kwargs) -> Action:
        planner.cloud_planner.last_left_dist = 4.0
        planner.cloud_planner.last_track_width = 6.0
        planner.cloud_planner.last_heading_error = 0.3
        return Action(steer=0.4, speed=3.0)

    planner.edge_planner.plan = edge_plan  # type: ignore[method-assign]
    planner.cloud_planner.plan = cloud_plan  # type: ignore[method-assign]

    monkeypatch.setattr(
        "f110_planning.reactive.edge_cloud_planner.get_reactive_action",
        lambda *_args, **_kwargs: Action(steer=0.25, speed=2.5),
    )

    first_action = planner.plan(copy.deepcopy(reactive_obs))
    second_action = planner.plan(copy.deepcopy(reactive_obs))

    assert first_action.steer == pytest.approx(0.1)
    assert first_action.speed == pytest.approx(1.0)
    assert second_action.steer == pytest.approx(0.25)
    assert second_action.speed == pytest.approx(2.5)
    assert len(scheduler.contexts) == 2

    first_context = scheduler.contexts[0]
    assert first_context["current_command"] == pytest.approx((0.1, 1.0))
    assert first_context["prev_command"] is None
    assert first_context["cloud_received"] is False

    second_context = scheduler.contexts[1]
    assert second_context["prev_edge_features"] == pytest.approx((1.0, 2.0, 0.1))
    assert second_context["prev_command"] == pytest.approx((0.1, 1.0))
    assert second_context["current_command"] == pytest.approx((0.25, 2.5))
    assert second_context["cloud_received"] is True


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
