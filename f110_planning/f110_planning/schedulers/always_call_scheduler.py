"""
Always-call cloud scheduler.
"""

from typing import Any

from ..base import Action, CloudScheduler


class AlwaysCallScheduler(CloudScheduler):  # pylint: disable=too-few-public-methods
    """Calls the cloud on every single step (the default behaviour)."""

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
    ) -> bool:
        return True
