"""Tests for tier-aware cloud scheduling."""

from f110_planning.base import Action, CloudTier
from f110_planning.schedulers import (
    SchedulingSignals,
    TieredPolicyDrivenScheduler,
    TieredProbabilisticRiskConfig,
    TieredProbabilisticRiskPolicy,
    TieredThresholdConfig,
    TieredThresholdPolicy,
)


def test_tiered_threshold_policy_routes_between_tiers() -> None:
    """Threshold policy should map disagreement into edge, medium, or large."""
    policy = TieredThresholdPolicy(
        TieredThresholdConfig(
            medium_threshold=0.2,
            large_threshold=0.5,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
        )
    )

    assert policy.choose_tier(0, SchedulingSignals(deviation=0.1), True) is None
    assert policy.choose_tier(1, SchedulingSignals(deviation=0.3), True) == CloudTier.MEDIUM
    assert policy.choose_tier(2, SchedulingSignals(deviation=0.7), True) == CloudTier.LARGE


def test_tiered_threshold_policy_supports_relative_quantiles() -> None:
    """Relative threshold mode should route using rolling disagreement quantiles."""
    policy = TieredThresholdPolicy(
        TieredThresholdConfig(
            medium_threshold=0.1,
            large_threshold=0.2,
            use_relative_thresholds=True,
            history_window=5,
            min_history=3,
            medium_quantile=0.5,
            large_quantile=0.9,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
        )
    )

    assert policy.choose_tier(0, SchedulingSignals(deviation=0.05), True) is None
    assert policy.choose_tier(1, SchedulingSignals(deviation=0.15), True) is CloudTier.MEDIUM
    assert policy.choose_tier(2, SchedulingSignals(deviation=0.25), True) is CloudTier.LARGE
    # history is now [0.05, 0.15, 0.25], so the adaptive large threshold should
    # remain above 0.2 and the adaptive medium threshold above 0.1.
    assert policy.choose_tier(3, SchedulingSignals(deviation=0.12), True) is None
    assert policy.choose_tier(4, SchedulingSignals(deviation=0.16), True) is CloudTier.MEDIUM
    assert policy.choose_tier(5, SchedulingSignals(deviation=0.24), True) is CloudTier.LARGE


def test_tiered_probabilistic_policy_extreme_probabilities() -> None:
    """Probability extremes should make the tiered policy deterministic."""
    medium_policy = TieredProbabilisticRiskPolicy(
        TieredProbabilisticRiskConfig(
            p_medium=1.0,
            p_large=0.0,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=3,
        )
    )
    large_policy = TieredProbabilisticRiskPolicy(
        TieredProbabilisticRiskConfig(
            p_medium=0.0,
            p_large=1.0,
            min_interval=1,
            warmup_steps=0,
            fallback_interval=0,
            seed=3,
        )
    )

    assert medium_policy.choose_tier(0, SchedulingSignals(deviation=0.0), True) == CloudTier.MEDIUM
    assert large_policy.choose_tier(0, SchedulingSignals(deviation=0.0), True) == CloudTier.LARGE


def test_tiered_scheduler_uses_context_deviation() -> None:
    """Tiered scheduler should route using the extracted deviation signal."""
    scheduler = TieredPolicyDrivenScheduler(
        TieredThresholdPolicy(
            TieredThresholdConfig(
                medium_threshold=0.2,
                large_threshold=0.5,
                min_interval=1,
                warmup_steps=0,
                fallback_interval=0,
            )
        )
    )
    latest_cloud_actions = {
        CloudTier.MEDIUM: Action(0.0, 1.0),
        CloudTier.LARGE: None,
    }
    obs = {
        "crosstrack_dist": [0.0],
        "linear_vels_x": [1.0],
    }

    assert (
        scheduler.choose_cloud_tier(
            1,
            obs,
            latest_cloud_actions,
            context={"deviation": 0.3},
        )
        == CloudTier.MEDIUM
    )
