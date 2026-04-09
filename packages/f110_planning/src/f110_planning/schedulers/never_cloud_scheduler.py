"""Scheduler that never issues cloud requests."""

from __future__ import annotations

from typing import Any

from ..base import Action, CloudScheduler


class NeverCloudScheduler(CloudScheduler):
    """Force edge-only operation by never requesting cloud inference."""

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        del step, obs, latest_cloud_action, context
        return False
