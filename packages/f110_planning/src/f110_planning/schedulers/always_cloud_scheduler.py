"""
Always-on cloud scheduler.
"""

from typing import Any

from ..base import Action, CloudScheduler


class AlwaysCloudScheduler(CloudScheduler):  # pylint: disable=too-few-public-methods
    """Requests cloud inference at every simulation step."""

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        del step, obs, latest_cloud_action, context
        return True

