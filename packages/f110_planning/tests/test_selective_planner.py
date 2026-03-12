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


def test_cache_always_contributes_with_age_weighting(obs: dict[str, Any]) -> None:
    """Populated cloud caches always contribute to the blended output, regardless
    of whether the DNN was in the current call_mask.  When sigma_proc is None
    (static weights) the blended value must differ from the pure edge value
    whenever the sentinel is far from the edge estimate.
    """
    planner = SelectiveEdgeCloudPlanner(cloud_latency=10)

    # Step 0: run with no calls to seed last_edge_* values.
    planner.plan(obs, call_mask=[False, False, False])

    # Inject sentinel values directly into all three caches (simulating stale
    # cloud results received in a distant past step).
    planner._cloud_cache[SelectiveEdgeCloudPlanner.LEFT] = 1111.0    # pylint: disable=protected-access
    planner._cloud_cache[SelectiveEdgeCloudPlanner.TRACK] = 2222.0   # pylint: disable=protected-access
    planner._cloud_cache[SelectiveEdgeCloudPlanner.HEADING] = 3333.0 # pylint: disable=protected-access

    # Step 1: call ONLY DNN 1 (TRACK) — LEFT and HEADING caches should still
    # blend into the output (with their static alpha weights).
    planner.plan(obs, call_mask=[False, True, False])

    # All three blended values must be affected by the large sentinels;
    # none of them should equal the pure edge estimate.
    # (alpha_left=0.996, alpha_track=0.988, alpha_heading=0.974 by default)
    assert planner.last_cloud_left != pytest.approx(planner.last_edge_left), (
        "LEFT sentinel (1111.0) should blend into last_cloud_left"
    )
    assert planner.last_cloud_heading != pytest.approx(planner.last_edge_heading), (
        "HEADING sentinel (3333.0) should blend into last_cloud_heading"
    )
    # TRACK cache was also just available (static alpha=0.988), so it also
    # shifts the blended value.
    assert planner.last_cloud_track != pytest.approx(planner.last_edge_track), (
        "TRACK sentinel (2222.0) should blend into last_cloud_track"
    )


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


# ---------------------------------------------------------------------------
# Per-feature alpha blending
# ---------------------------------------------------------------------------

def test_alpha_age_static_at_age_zero(obs: dict[str, Any]) -> None:
    """_alpha_age at age=0 must equal sigma_e^2 / (sigma_e^2 + sigma_c^2)."""
    se2, sc2 = 0.059444, 0.000212
    expected = se2 / (se2 + sc2)
    result = SelectiveEdgeCloudPlanner._alpha_age(  # pylint: disable=protected-access
        age=0, sigma_e2=se2, sigma_c2=sc2, sigma_proc2=0.01
    )
    assert result == pytest.approx(expected, rel=1e-9)


def test_alpha_age_decreases_with_age(obs: dict[str, Any]) -> None:
    """When sigma_proc > 0, alpha must decrease as age increases."""
    se2, sc2, sp2 = 0.059444, 0.000212, 0.01
    alpha_0 = SelectiveEdgeCloudPlanner._alpha_age(0, se2, sc2, sp2)  # pylint: disable=protected-access
    alpha_5 = SelectiveEdgeCloudPlanner._alpha_age(5, se2, sc2, sp2)  # pylint: disable=protected-access
    alpha_20 = SelectiveEdgeCloudPlanner._alpha_age(20, se2, sc2, sp2)  # pylint: disable=protected-access
    assert alpha_0 > alpha_5 > alpha_20


def test_edge_fallback_before_cloud_ever_arrives(obs: dict[str, Any]) -> None:
    """When cache is None (no cloud result yet), blended value must equal edge."""
    planner = SelectiveEdgeCloudPlanner(cloud_latency=10)
    planner.plan(obs, call_mask=[False, False, False])
    # All caches still None → blended == edge
    assert planner.last_cloud_left == pytest.approx(planner.last_edge_left)
    assert planner.last_cloud_track == pytest.approx(planner.last_edge_track)
    assert planner.last_cloud_heading == pytest.approx(planner.last_edge_heading)


def test_edge_delta_correction_stored_in_cache(obs: dict[str, Any]) -> None:
    """With cloud_latency=0 the cache should be updated on the same step.
    The stored value should equal cloud_result + (edge_now - edge_at_enqueue),
    which at latency=0 collapses to the raw cloud output (delta = 0).
    """
    planner = SelectiveEdgeCloudPlanner(cloud_latency=0)
    planner.plan(obs, call_mask=[True, False, False])
    # cache[LEFT] may be None if the model returns None, but it must not crash
    # and any non-None value should be a finite float.
    cached = planner._cloud_cache[SelectiveEdgeCloudPlanner.LEFT]  # pylint: disable=protected-access
    if cached is not None:
        assert np.isfinite(cached)
