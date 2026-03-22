"""
Adapter from tiered safety policy objects to the TieredCloudScheduler interface.
"""

from typing import Any, Callable

from ..base import Action, CloudTier, TieredCloudScheduler
from .safety_policies import TieredSchedulingPolicy
from .safety_signals import SchedulingSignals, build_signals


SignalExtractor = Callable[[dict[str, Any], dict[str, Any] | None], SchedulingSignals]


class TieredPolicyDrivenScheduler(TieredCloudScheduler):  # pylint: disable=too-few-public-methods
    """
    Wrap a `TieredSchedulingPolicy` for use with `MultiTierEdgeCloudPlanner`.
    """

    def __init__(
        self,
        policy: TieredSchedulingPolicy,
        signal_extractor: SignalExtractor | None = None,
    ) -> None:
        self.policy = policy
        self.signal_extractor = signal_extractor or build_signals

    def choose_cloud_tier(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_actions: dict[CloudTier, Action | None],
        context: dict[str, Any] | None = None,
    ) -> CloudTier | None:
        signals = self.signal_extractor(obs, context)
        latest_cloud_action_available = any(
            latest_cloud_actions.get(tier) is not None for tier in (CloudTier.MEDIUM, CloudTier.LARGE)
        )
        return self.policy.choose_tier(
            step=step,
            signals=signals,
            latest_cloud_action_available=latest_cloud_action_available,
        )
