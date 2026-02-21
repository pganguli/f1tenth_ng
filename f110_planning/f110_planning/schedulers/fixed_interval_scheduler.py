"""
Fixed-interval cloud scheduler.
"""

from typing import Any

from ..base import Action, CloudScheduler


class FixedIntervalScheduler(CloudScheduler):  # pylint: disable=too-few-public-methods
    """
    Calls the cloud every *interval* steps.

    Parameters
    ----------
    interval : int
        Number of steps between successive cloud requests.
    """

    def __init__(self, interval: int = 10) -> None:
        self.interval = interval

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
    ) -> bool:
        return step % self.interval == 0
