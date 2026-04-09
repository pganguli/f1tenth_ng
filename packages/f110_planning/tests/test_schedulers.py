"""Tests for cloud scheduler implementations."""

import numpy as np
import pytest

from f110_planning.schedulers import (
    DualSignalPeriodicConfig,
    DualSignalPeriodicScheduler,
    FixedIntervalScheduler,
    NeverCloudScheduler,
    RLScheduler,
    RoundRobinScheduler,
    SelfNormalizingMomentumConfig,
    SelfNormalizingMomentumScheduler,
    SensitivityProportionalScheduler,
)
from f110_planning.base import Action


def test_fixed_interval() -> None:
    """FixedIntervalScheduler should trigger on multiples of the given interval."""
    sched = FixedIntervalScheduler(interval=3)
    # should call on step 0,3,6,...
    calls = [sched.should_call_cloud(i, {}, None) for i in range(10)]
    expected = [i % 3 == 0 for i in range(10)]
    assert calls == expected


def test_rl_scheduler_basic() -> None:
    """RLScheduler should default to False and respect set_action / reset."""
    sched = RLScheduler()
    # no action set -> defaults to False
    assert not sched.should_call_cloud(0, {}, None)
    # set action to True
    sched.set_action(True)
    assert sched.should_call_cloud(5, {}, None)
    # clearing via reset
    sched.reset()
    assert not sched.should_call_cloud(0, {}, None)


def test_legacy_scheduler_exports_remain_available() -> None:
    """Older scheduler imports should remain available from the package root."""
    assert RoundRobinScheduler is not None
    assert SensitivityProportionalScheduler is not None


def test_self_normalizing_scheduler_bootstraps_only_once_before_first_receipt() -> None:
    """SRP should issue exactly one bootstrap request before the first receipt."""
    sched = SelfNormalizingMomentumScheduler(
        SelfNormalizingMomentumConfig(cloud_latency=5)
    )

    assert sched.should_call_cloud(
        0,
        {},
        None,
        context={
            "cloud_in_flight": False,
        },
    )
    assert not sched.should_call_cloud(
        1,
        {},
        None,
        context={
            "cloud_in_flight": False,
        },
    )
    state = sched.debug_state()
    assert state["bootstrap_issued"]
    assert not state["has_received_cloud"]
    assert state["last_probability"] == pytest.approx(1.0)


def test_self_normalizing_scheduler_blocks_while_in_flight() -> None:
    """Outstanding cloud work should suppress duplicate requests."""
    sched = SelfNormalizingMomentumScheduler(
        SelfNormalizingMomentumConfig(cloud_latency=5)
    )
    assert not sched.should_call_cloud(
        0,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (0.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_received": True,
            "cloud_in_flight": False,
        },
    )

    assert not sched.should_call_cloud(
        1,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (1.0, 1.0, 1.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (1.0, 1.0),
            "prev_command": (0.0, 0.0),
            "cloud_in_flight": True,
        },
    )


def test_self_normalizing_scheduler_uses_returned_command_stream() -> None:
    """The command-side SRP term should come from the returned command stream."""
    sched = SelfNormalizingMomentumScheduler(
        SelfNormalizingMomentumConfig(cloud_latency=5, tau=10.0, seed=7)
    )
    assert not sched.should_call_cloud(
        0,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (0.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_received": True,
            "cloud_in_flight": False,
        },
    )
    sched.should_call_cloud(
        1,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (1.0, 1.0, 1.0),
            "edge_action": (0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (2.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_in_flight": False,
        },
    )
    state = sched.debug_state()
    assert state["last_instability"] == pytest.approx(5.0)
    assert state["last_effort"] == pytest.approx(np.sqrt(12.5))
    assert state["last_score"] == pytest.approx(5.0 * np.sqrt(12.5))


def test_self_normalizing_scheduler_updates_scale_and_baseline_before_ratio() -> None:
    """SRP should compute rho against the updated per-step baseline."""
    sched = SelfNormalizingMomentumScheduler(
        SelfNormalizingMomentumConfig(cloud_latency=5, tau=10.0, seed=7)
    )
    assert not sched.should_call_cloud(
        0,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (0.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_received": True,
            "cloud_in_flight": False,
        },
    )
    sched.should_call_cloud(
        1,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (1.0, 1.0, 1.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (2.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_in_flight": False,
        },
    )
    state = sched.debug_state()
    assert state["last_ratio"] == pytest.approx(5.0)
    assert state["last_probability"] == pytest.approx(0.4)


def test_self_normalizing_scheduler_stale_fallback_triggers() -> None:
    """The stale override should first fire strictly after nmax * latency."""
    sched = SelfNormalizingMomentumScheduler(
        SelfNormalizingMomentumConfig(cloud_latency=5, nmax=5, seed=7)
    )
    assert not sched.should_call_cloud(
        0,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (0.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_received": True,
            "cloud_in_flight": False,
        },
    )

    assert not sched.should_call_cloud(
        25,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (0.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_in_flight": False,
        },
    )
    assert sched.should_call_cloud(
        26,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (0.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_in_flight": False,
        },
    )
    assert sched.debug_state()["last_stale_override"]


def test_self_normalizing_scheduler_quiet_regime_does_not_call() -> None:
    """Zero delta after the first receipt should collapse the SRP score to zero."""
    sched = SelfNormalizingMomentumScheduler(
        SelfNormalizingMomentumConfig(cloud_latency=5, seed=7)
    )
    assert not sched.should_call_cloud(
        0,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (0.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_received": True,
            "cloud_in_flight": False,
        },
    )

    assert not sched.should_call_cloud(
        1,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (0.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_in_flight": False,
        },
    )
    assert sched.debug_state()["last_probability"] == pytest.approx(0.0)


def test_self_normalizing_scheduler_seeded_draws_are_reproducible() -> None:
    """Identical seeded SRP schedulers should make the same probabilistic choice."""
    config = SelfNormalizingMomentumConfig(cloud_latency=5, tau=0.5, seed=17)
    sched_a = SelfNormalizingMomentumScheduler(config)
    sched_b = SelfNormalizingMomentumScheduler(config)
    for sched in (sched_a, sched_b):
        sched._sigma_e = np.ones(3)  # pylint: disable=protected-access
        sched._sigma_u = np.ones(2)  # pylint: disable=protected-access
        sched._baseline = 0.8  # pylint: disable=protected-access
        sched._has_received_cloud = True  # pylint: disable=protected-access
        sched._last_cloud_receipt_step = 0  # pylint: disable=protected-access

    context = {
        "edge_features": (2.0, 0.0, 0.0),
        "prev_edge_features": (0.0, 0.0, 0.0),
        "current_command": (2.0, 0.0),
        "prev_command": (0.0, 0.0),
        "cloud_in_flight": False,
    }
    latest = Action(0.0, 1.0)

    decision_a = sched_a.should_call_cloud(1, {}, latest, context=context)
    decision_b = sched_b.should_call_cloud(
        1,
        {},
        latest,
        context=context,
    )
    assert 0.0 < sched_a.last_probability < 1.0
    assert decision_a == decision_b


def test_self_normalizing_scheduler_srpv2_uses_pre_update_scales_and_baseline() -> None:
    """SRPv2 should score the current event against history before absorbing it."""
    sched = SelfNormalizingMomentumScheduler(
        SelfNormalizingMomentumConfig(
            cloud_latency=5,
            tau=1.0,
            causal_feature_scales=True,
            causal_baseline=True,
            seed=7,
        )
    )
    sched._sigma_e = np.ones(3)  # pylint: disable=protected-access
    sched._sigma_u = np.ones(2)  # pylint: disable=protected-access
    sched._baseline = 1.0  # pylint: disable=protected-access
    sched._has_received_cloud = True  # pylint: disable=protected-access
    sched._last_cloud_receipt_step = 0  # pylint: disable=protected-access

    assert not sched.should_call_cloud(
        1,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (2.0, 0.0, 0.0),
            "prev_edge_features": (0.0, 0.0, 0.0),
            "current_command": (2.0, 0.0),
            "prev_command": (0.0, 0.0),
            "cloud_in_flight": True,
        },
    )

    state = sched.debug_state()
    assert state["causal_feature_scales"]
    assert state["causal_baseline"]
    assert state["last_ratio"] == pytest.approx(1.632993161855452)
    assert state["last_probability"] == pytest.approx(0.6329931618554521)
    assert state["pressure_baseline"] == pytest.approx(1.1265986323710904)


def test_never_cloud_scheduler_never_calls() -> None:
    """The never-query control should always remain edge-only."""
    sched = NeverCloudScheduler()
    assert not sched.should_call_cloud(0, {}, None)
    assert not sched.should_call_cloud(
        10,
        {},
        Action(0.0, 1.0),
        context={"cloud_received": True},
    )


def test_dual_signal_scheduler_bootstrap_and_backbone_ignore_queue_depth() -> None:
    """Bootstrap/backbone refreshes should fire on cadence even with queued work."""
    sched = DualSignalPeriodicScheduler(
        DualSignalPeriodicConfig(cloud_latency=5, base_interval=3)
    )

    assert sched.should_call_cloud(
        0,
        {},
        None,
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "cloud_age": 999,
            "cloud_queue_depth": 0,
        },
    )
    assert sched.last_call_reason == "bootstrap"
    assert not sched.should_call_cloud(
        1,
        {},
        None,
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "cloud_age": 999,
            "cloud_queue_depth": 2,
        },
    )
    assert sched.should_call_cloud(
        3,
        {},
        None,
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "cloud_age": 999,
            "cloud_queue_depth": 3,
        },
    )
    assert sched.last_call_reason == "bootstrap"


def test_dual_signal_scheduler_bursts_on_high_deviation() -> None:
    """High deviation should trigger an off-backbone burst call."""
    sched = DualSignalPeriodicScheduler(
        DualSignalPeriodicConfig(cloud_latency=5, base_interval=4, burst_threshold=0.6)
    )
    sched.should_call_cloud(
        0,
        {},
        None,
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "cloud_age": 999,
            "cloud_queue_depth": 0,
        },
    )

    assert sched.should_call_cloud(
        1,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "deviation": 0.2,
            "cloud_age": 1,
            "cloud_queue_depth": 0,
        },
    )
    assert sched.last_call_reason == "burst"


def test_dual_signal_scheduler_bursts_when_momentum_lifts_moderate_deviation() -> None:
    """Momentum should help a moderate deviation cross the burst threshold."""
    sched = DualSignalPeriodicScheduler(
        DualSignalPeriodicConfig(
            cloud_latency=5,
            base_interval=10,
            burst_threshold=0.5,
            age_weight=0.15,
            deviation_weight=0.55,
            momentum_weight=0.30,
        )
    )
    sched._prev_edge_features = np.zeros(3)  # pylint: disable=protected-access
    sched._prev_edge_action = np.zeros(2)  # pylint: disable=protected-access
    sched._sigma_e = np.ones(3)  # pylint: disable=protected-access
    sched._sigma_u = np.ones(2)  # pylint: disable=protected-access
    sched._momentum = 0.5  # pylint: disable=protected-access

    assert sched.should_call_cloud(
        1,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (2.0, 0.0, 0.0),
            "edge_action": (2.0, 0.0),
            "deviation": 0.035,
            "cloud_age": 1,
            "cloud_queue_depth": 0,
        },
    )
    assert sched.last_call_reason == "burst"


def test_dual_signal_scheduler_force_age_and_queue_cap() -> None:
    """Force-age should refresh stale cloud, while queue cap suppresses bursts."""
    sched = DualSignalPeriodicScheduler(
        DualSignalPeriodicConfig(
            cloud_latency=5,
            base_interval=10,
            burst_threshold=0.6,
            burst_queue_cap=1,
            force_age_multiplier=3,
        )
    )
    sched.should_call_cloud(
        0,
        {},
        None,
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "cloud_age": 999,
            "cloud_queue_depth": 0,
        },
    )

    assert sched.should_call_cloud(
        1,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "deviation": 0.2,
            "cloud_age": 16,
            "cloud_queue_depth": 0,
        },
    )
    assert sched.last_call_reason == "force_age"

    assert not sched.should_call_cloud(
        2,
        {},
        Action(0.0, 1.0),
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "deviation": 0.2,
            "cloud_age": 1,
            "cloud_queue_depth": 1,
        },
    )
    assert sched.last_call_reason == "none"


def test_dual_signal_scheduler_normalizes_invalid_weights_and_reset_clears_state() -> None:
    """Invalid weights should fall back and reset should zero counters."""
    sched = DualSignalPeriodicScheduler(
        DualSignalPeriodicConfig(
            cloud_latency=5,
            age_weight=-1.0,
            deviation_weight=0.0,
            momentum_weight=0.0,
        )
    )

    assert sched._weights == pytest.approx((0.20, 0.60, 0.20))  # pylint: disable=protected-access
    sched.should_call_cloud(
        0,
        {},
        None,
        context={
            "edge_features": (0.0, 0.0, 0.0),
            "edge_action": (0.0, 0.0),
            "cloud_age": 999,
            "cloud_queue_depth": 0,
        },
    )
    sched.reset()

    assert sched.debug_state()["call_reason_counts"] == {
        "bootstrap": 0,
        "backbone": 0,
        "burst": 0,
        "force_age": 0,
        "none": 0,
    }
    assert sched.last_call_reason == "none"
