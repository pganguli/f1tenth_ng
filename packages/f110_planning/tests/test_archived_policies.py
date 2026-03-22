"""Tests for archived (legacy) scheduling policies.

These policies have been superseded by paper-accurate implementations
but are kept for reference. These tests verify the archived code still works.
"""

from f110_planning.schedulers._archived_policies import (
    BernoulliDeviationConfig,
    BernoulliDeviationPolicy,
    MaxMissDeterministicConfig,
    MaxMissDeterministicPolicy,
)
from f110_planning.schedulers.safety_policies import SchedulingSignals


def test_max_miss_policy_forces_refresh() -> None:
    """Max-miss policy should force a cloud call after bounded skips."""
    policy = MaxMissDeterministicPolicy(
        MaxMissDeterministicConfig(
            threshold=10.0,
            max_miss=2,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
        )
    )
    assert policy.should_call(0, SchedulingSignals(deviation=0.0), False)
    assert not policy.should_call(1, SchedulingSignals(deviation=0.0), True)
    assert not policy.should_call(2, SchedulingSignals(deviation=0.0), True)
    assert policy.should_call(3, SchedulingSignals(deviation=0.0), True)


def test_bernoulli_policy_extreme_probabilities() -> None:
    """Bernoulli policy should be deterministic at p=0 and p=1."""
    never_call = BernoulliDeviationPolicy(
        BernoulliDeviationConfig(
            threshold=100.0,
            base_prob=0.0,
            risk_gain=0.0,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=7,
        )
    )
    always_call = BernoulliDeviationPolicy(
        BernoulliDeviationConfig(
            threshold=0.0,
            base_prob=1.0,
            risk_gain=0.0,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=7,
        )
    )
    assert not never_call.should_call(1, SchedulingSignals(deviation=1.0), True)
    assert always_call.should_call(1, SchedulingSignals(deviation=0.0), True)
