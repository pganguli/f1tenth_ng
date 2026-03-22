"""
Adapter from safety policy objects to the CloudScheduler interface.
"""

from typing import Any, Callable

from ..base import Action, CloudScheduler
from .safety_policies import SchedulingPolicy
from .safety_signals import SchedulingSignals, build_signals


SignalExtractor = Callable[[dict[str, Any], dict[str, Any] | None], SchedulingSignals]


class PolicyDrivenScheduler(CloudScheduler):  # pylint: disable=too-few-public-methods
    """
    Wrap a `SchedulingPolicy` for use with `EdgeCloudPlanner`.
    """

    def __init__(
        self,
        policy: SchedulingPolicy,
        signal_extractor: SignalExtractor | None = None,
    ) -> None:
        self.policy = policy
        self.signal_extractor = signal_extractor or build_signals

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        signals = self.signal_extractor(obs, context)
        return self.policy.should_call(
            step=step,
            signals=signals,
            latest_cloud_action_available=latest_cloud_action is not None,
        )
