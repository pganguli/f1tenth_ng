"""Tests for policy-driven cloud scheduling."""

import pytest

from f110_planning.base import Action
from f110_planning.schedulers import (
    BernoulliMaxMissConfig,
    BernoulliMaxMissPolicy,
    DeterministicDeviationConfig,
    DeterministicDeviationPolicy,
    ExponentialRiskConfig,
    ExponentialRiskPolicy,
    FixedBernoulliConfig,
    FixedBernoulliPolicy,
    LogisticRiskConfig,
    LogisticRiskPolicy,
    PiecewiseLinearRampConfig,
    PiecewiseLinearRampPolicy,
    PolicyDrivenScheduler,
    SchedulingSignals,
)


def test_deterministic_policy_respects_min_interval() -> None:
    """Policy should gate calls according to min interval."""
    policy = DeterministicDeviationPolicy(
        DeterministicDeviationConfig(
            threshold=0.5,
            min_interval=2,
            warmup_steps=1,
            fallback_interval=0,
        )
    )

    assert policy.should_call(0, SchedulingSignals(deviation=0.0), False)
    # min interval blocks this step even though deviation is high
    assert not policy.should_call(1, SchedulingSignals(deviation=0.8), True)
    # interval elapsed -> high deviation triggers call
    assert policy.should_call(2, SchedulingSignals(deviation=0.8), True)


def test_deterministic_policy_fallback_interval() -> None:
    """Policy should eventually call cloud even with low deviation."""
    policy = DeterministicDeviationPolicy(
        DeterministicDeviationConfig(
            threshold=10.0,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=3,
        )
    )

    assert policy.should_call(0, SchedulingSignals(deviation=0.0), False)
    assert not policy.should_call(1, SchedulingSignals(deviation=0.0), True)
    assert not policy.should_call(2, SchedulingSignals(deviation=0.0), True)
    assert policy.should_call(3, SchedulingSignals(deviation=0.0), True)


def test_policy_driven_scheduler_prefers_context_deviation() -> None:
    """Scheduler should prefer explicit context deviation over observation fallbacks."""
    policy = DeterministicDeviationPolicy(
        DeterministicDeviationConfig(
            threshold=0.2,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
        )
    )
    scheduler = PolicyDrivenScheduler(policy)

    obs = {
        "crosstrack_dist": [0.0],
        "linear_vels_x": [1.0],
    }

    latest_cloud_action = Action(steer=0.0, speed=1.0)
    assert scheduler.should_call_cloud(
        1,
        obs,
        latest_cloud_action,
        context={"deviation": 0.3},
    )
    assert not scheduler.should_call_cloud(
        2,
        obs,
        latest_cloud_action,
        context={"deviation": 0.1},
    )


def test_fixed_bernoulli_policy_constant_probability() -> None:
    """Fixed Bernoulli scheduling should ignore deviation and honor p extremes."""
    never_call = FixedBernoulliPolicy(
        FixedBernoulliConfig(
            p=0.0,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=7,
        )
    )
    always_call = FixedBernoulliPolicy(
        FixedBernoulliConfig(
            p=1.0,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=7,
        )
    )

    assert not never_call.should_call(1, SchedulingSignals(deviation=0.0), True)
    assert not never_call.should_call(2, SchedulingSignals(deviation=1.0), True)
    assert always_call.should_call(1, SchedulingSignals(deviation=0.0), True)
    assert always_call.should_call(2, SchedulingSignals(deviation=1.0), True)


def test_bernoulli_max_miss_forces_after_m_misses() -> None:
    """Bernoulli+guard should force a call after the configured miss streak."""
    policy = BernoulliMaxMissPolicy(
        BernoulliMaxMissConfig(
            p=0.0,
            max_miss=3,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=7,
        )
    )

    assert policy.should_call(0, SchedulingSignals(deviation=0.0), False)
    assert not policy.should_call(1, SchedulingSignals(deviation=0.0), True)
    assert not policy.should_call(2, SchedulingSignals(deviation=0.0), True)
    assert not policy.should_call(3, SchedulingSignals(deviation=0.0), True)
    assert policy.should_call(4, SchedulingSignals(deviation=0.0), True)



def test_logistic_policy_probability_bounds() -> None:
    """Logistic strategy should map low/high deviation near p_min/p_max."""
    policy = LogisticRiskPolicy(
        LogisticRiskConfig(
            center=0.5,
            slope=100.0,
            p_min=0.1,
            p_max=0.9,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=1,
        )
    )
    p_low = policy.probability(SchedulingSignals(deviation=0.0))
    p_high = policy.probability(SchedulingSignals(deviation=1.0))
    assert p_low < 0.15
    assert p_high > 0.85


def test_exponential_policy_probability_monotonicity() -> None:
    """Exponential strategy probability should rise with deviation."""
    policy = ExponentialRiskPolicy(
        ExponentialRiskConfig(
            center=0.2,
            rate=10.0,
            p_min=0.0,
            p_max=1.0,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=1,
        )
    )
    p1 = policy.probability(SchedulingSignals(deviation=0.2))
    p2 = policy.probability(SchedulingSignals(deviation=0.4))
    p3 = policy.probability(SchedulingSignals(deviation=1.0))
    assert 0.0 <= p1 <= p2 <= p3 <= 1.0


def test_piecewise_linear_ramp_probability_bounds() -> None:
    """Piecewise ramp should clamp outside thresholds and interpolate inside."""
    policy = PiecewiseLinearRampPolicy(
        PiecewiseLinearRampConfig(
            d_low=0.01,
            d_high=0.05,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=7,
        )
    )

    assert policy.probability(SchedulingSignals(deviation=0.005)) == 0.0
    assert policy.probability(SchedulingSignals(deviation=0.03)) == pytest.approx(0.5)
    assert policy.probability(SchedulingSignals(deviation=0.06)) == 1.0


def test_piecewise_linear_ramp_invalid_config() -> None:
    """Piecewise ramp should reject inverted or degenerate threshold ranges."""
    with pytest.raises(ValueError, match="d_high must be greater than d_low"):
        PiecewiseLinearRampPolicy(PiecewiseLinearRampConfig(d_low=0.05, d_high=0.05))
