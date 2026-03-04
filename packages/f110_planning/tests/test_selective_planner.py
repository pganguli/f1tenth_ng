"""Unit tests for SelectiveEdgeCloudPlanner."""

from typing import Any

import numpy as np
import pytest

from f110_planning.reactive import SelectiveEdgeCloudPlanner


@pytest.fixture
def obs() -> dict[str, Any]:
    """Minimal observation dict with a flat 1080-ray scan."""
    return {
        "poses_x": np.array([0.0]),
        "poses_y": np.array([0.0]),
        "poses_theta": np.array([0.0]),
        "linear_vels_x": np.array([1.0]),
        "linear_vels_y": np.array([0.0]),
        "ang_vels_z": np.array([0.0]),
        "scans": np.ones((1, 1080)) * 5.0,
    }


# ---------------------------------------------------------------------------
# Queue mechanics
# ---------------------------------------------------------------------------

def test_only_selected_dnn_queued(obs: dict[str, Any]) -> None:
    """Requesting DNN 0 should enqueue to queue[0] only."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=5)
    planner.plan(obs, call_mask=[True, False, False])
    assert len(planner._queues[0]) == 1  # pylint: disable=protected-access
    assert len(planner._queues[1]) == 0  # pylint: disable=protected-access
    assert len(planner._queues[2]) == 0  # pylint: disable=protected-access


def test_all_queues_populated_when_all_called(obs: dict[str, Any]) -> None:
    """All three queues should each receive exactly one entry."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=5)
    planner.plan(obs, call_mask=[True, True, True])
    for i in range(SelectiveEdgeCloudPlanner.NUM_DNNS):
        assert len(planner._queues[i]) == 1  # pylint: disable=protected-access


def test_no_queue_entry_when_call_mask_false(obs: dict[str, Any]) -> None:
    """No call_mask=False should leave all queues empty."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=5)
    planner.plan(obs, call_mask=[False, False, False])
    for q in planner._queues:  # pylint: disable=protected-access
        assert len(q) == 0


# ---------------------------------------------------------------------------
# Edge-fallback / hold-last behaviour
# ---------------------------------------------------------------------------

def test_edge_fallback_before_cloud_arrives(obs: dict[str, Any]) -> None:
    """Before any cloud result arrives (cache=None), resolved cloud values
    must equal the corresponding edge outputs regardless of call_mask."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=10)
    planner.plan(obs, call_mask=[False, False, False])

    # caches should still be None
    assert planner._cloud_cache == [None, None, None]  # pylint: disable=protected-access
    # resolved values fall back to current edge outputs
    assert planner.last_cloud_left == pytest.approx(planner.last_edge_left)
    assert planner.last_cloud_track == pytest.approx(planner.last_edge_track)
    assert planner.last_cloud_heading == pytest.approx(planner.last_edge_heading)


def test_stale_cache_not_used_for_uncalled_dnn(obs: dict[str, Any]) -> None:
    """A populated cloud cache for a DNN must NOT be used when that DNN is
    absent from the current call_mask.  Fresh edge output must be used instead.

    This prevents arbitrarily-stale cloud values from poisoning the reactive
    controller at extreme bends when top_k=1 keeps cycling through only one
    DNN.
    """
    planner = SelectiveEdgeCloudPlanner(cloud_latency=10)

    # Step 0: run with no calls to seed last_edge_* values.
    planner.plan(obs, call_mask=[False, False, False])

    # Inject sentinel values directly into all three caches (simulating stale
    # cloud results received in a distant past step).
    planner._cloud_cache[SelectiveEdgeCloudPlanner.LEFT] = 1111.0    # pylint: disable=protected-access
    planner._cloud_cache[SelectiveEdgeCloudPlanner.TRACK] = 2222.0   # pylint: disable=protected-access
    planner._cloud_cache[SelectiveEdgeCloudPlanner.HEADING] = 3333.0 # pylint: disable=protected-access

    # Step 1: call ONLY DNN 1 (TRACK) — LEFT and HEADING must NOT use sentinels.
    # With latency=10 the TRACK result has not arrived yet either, so TRACK
    # uses its injected stale value (call_mask[TRACK]=True, cache is not None).
    planner.plan(obs, call_mask=[False, True, False])

    fresh_edge_left = planner.last_edge_left
    fresh_edge_heading = planner.last_edge_heading

    assert planner.last_cloud_left == pytest.approx(fresh_edge_left), (
        f"Stale LEFT sentinel (1111.0) leaked; expected fresh edge {fresh_edge_left}"
    )
    assert planner.last_cloud_heading == pytest.approx(fresh_edge_heading), (
        f"Stale HEADING sentinel (3333.0) leaked; expected fresh edge {fresh_edge_heading}"
    )
    # TRACK was called and cache is populated → should use the stale sentinel.
    assert planner.last_cloud_track == pytest.approx(2222.0)


def test_cloud_cache_consumed_after_latency(obs: dict[str, Any]) -> None:
    """After cloud_latency steps the in-flight request must be consumed."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=2)

    # Step 0: enqueue DNN 0; arrival step = 0 + 2 = 2
    planner.plan(obs, call_mask=[True, False, False])
    assert len(planner._queues[0]) == 1  # pylint: disable=protected-access

    # Step 1: no new request; queue not yet popped (arrival at step 2)
    planner.plan(obs, call_mask=[False, False, False])
    assert len(planner._queues[0]) == 1  # pylint: disable=protected-access

    # Step 2: now step == arrival step → queue entry popped and cache updated
    planner.plan(obs, call_mask=[False, False, False])
    assert len(planner._queues[0]) == 0  # pylint: disable=protected-access


def test_cloud_cache_only_updated_for_called_dnn(obs: dict[str, Any]) -> None:
    """Only the called DNN's cache should be populated; others remain None."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=0)  # immediate arrival
    planner.plan(obs, call_mask=[True, False, False])

    # With latency=0 the result arrives in the same step (step >=0 condition)
    # Queues are processed at arrival <= current step, so cache[0] should be set
    # (or None if the model returns None — both are acceptable non-crash outcomes)
    assert planner._cloud_cache[1] is None  # pylint: disable=protected-access
    assert planner._cloud_cache[2] is None  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Default call_mask and last_call_mask attribute
# ---------------------------------------------------------------------------

def test_default_call_mask_is_no_call(obs: dict[str, Any]) -> None:
    """plan() without call_mask argument must default to all-False."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=5)
    planner.plan(obs)
    assert planner.last_call_mask == [False, False, False]


def test_last_call_mask_reflects_input(obs: dict[str, Any]) -> None:
    """last_call_mask should mirror the mask passed to plan()."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=5)
    planner.plan(obs, call_mask=[False, True, True])
    assert planner.last_call_mask == [False, True, True]


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_queues_and_cache(obs: dict[str, Any]) -> None:
    """reset() must wipe per-DNN queues, cloud cache, and step counter."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=5)
    planner.plan(obs, call_mask=[True, True, True])
    planner.reset()

    assert planner._step == 0  # pylint: disable=protected-access
    for q in planner._queues:  # pylint: disable=protected-access
        assert len(q) == 0
    assert planner._cloud_cache == [None, None, None]  # pylint: disable=protected-access
    assert planner.last_action is None
    assert planner.last_call_mask == [False, False, False]


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------

def test_plan_returns_finite_action(obs: dict[str, Any]) -> None:
    """plan() must return an Action with finite steer and non-negative speed."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=5)
    action = planner.plan(obs, call_mask=[True, False, False])
    assert np.isfinite(action.steer)
    assert np.isfinite(action.speed)
    assert action.speed >= 0.0
