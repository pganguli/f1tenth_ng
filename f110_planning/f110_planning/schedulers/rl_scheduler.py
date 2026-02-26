"""RL-friendly cloud scheduler for imitation/learning.

This scheduler simply stores the most recent action decision provided by an
external agent.  The RL environment sets the action each step using
:meth:`set_action` and the :meth:`should_call_cloud` method returns the
currently stored value.  We separate the two so that the usual planner
interface is preserved while allowing an external controller to override the
scheduling decision.
"""

from __future__ import annotations

from typing import Any

from .fixed_interval_scheduler import Action, CloudScheduler


class RLScheduler(CloudScheduler):
    """Scheduler whose behaviour is driven by an external RL agent.

    The environment calls :meth:`set_action` once per step to convey whether a
    cloud request should be issued.  ``should_call_cloud`` simply returns the
    stored flag (default ``False`` until an action is received).
    """

    def __init__(self) -> None:
        self._call_next: bool | None = None

    # public API -------------------------------------------------------------
    def set_action(self, call_cloud: bool) -> None:
        """Tells the scheduler what decision the agent made for the upcoming step.

        Args:
            call_cloud: ``True`` to request a cloud inference; ``False``
                otherwise.
        """
        self._call_next = bool(call_cloud)

    # CloudScheduler interface ------------------------------------------------
    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
    ) -> bool:
        # ``_call_next`` may be ``None`` if the env never set an action
        # (e.g. before reset).  Treat that as ``False``.
        return bool(self._call_next)

    def reset(self) -> None:  # type: ignore[override]
        """Clear any stored action state."""
        self._call_next = None
