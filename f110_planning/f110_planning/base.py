"""
Base classes and common types for F1TENTH Planning.
"""

from abc import ABC, abstractmethod
from typing import Any, NamedTuple


class Action(NamedTuple):
    """
    Represents the control action for a vehicle.

    Attributes:
        steer (float): Steering angle in radians.
        speed (float): Requested longitudinal speed in meters per second.
    """

    steer: float
    speed: float


class BasePlanner(ABC):  # pylint: disable=too-few-public-methods
    """
    Abstract base class for all vehicle planners.
    """

    @abstractmethod
    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Computes the next control action based on the observation.

        Args:
            obs: A dictionary containing simulation observations.
            ego_idx: The index of the agent being controlled.

        Returns:
            Action: The computed steering and speed commands.
        """


class CloudScheduler(ABC):  # pylint: disable=too-few-public-methods
    """
    Decides **when** to issue a cloud inference request.

    Subclass this to implement arbitrary scheduling policies – fixed
    interval, adaptive, learned (RL), etc.  The
    :class:`~f110_planning.reactive.EdgeCloudPlanner` calls
    :meth:`should_call_cloud` once per simulation step.
    """

    @abstractmethod
    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Return ``True`` to issue a cloud request on this step.

        Args:
            step: The current simulation step (0-based).
            obs: The current observation dict.
            latest_cloud_action: The most recent cloud action received
                (``None`` if no cloud result has arrived yet).
            context: Optional runtime context from the edge planner
                (e.g., uncertainty estimates).

        Returns:
            Whether to send a new cloud inference request.
        """


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
        context: dict[str, Any] | None = None,
    ) -> bool:
        return step % self.interval == 0


class AlwaysCallScheduler(CloudScheduler):  # pylint: disable=too-few-public-methods
    """Calls the cloud on every single step (the default behaviour)."""

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        return True


class UncertaintyThresholdScheduler(CloudScheduler):  # pylint: disable=too-few-public-methods
    """
    Calls cloud when edge uncertainty exceeds a threshold.

    Expects ``context["edge_uncertainty"]`` from the caller.
    """

    def __init__(
        self,
        threshold: float = 0.03,
        min_interval: int = 1,
        warmup_steps: int = 1,
        fallback_interval: int = 0,
    ) -> None:
        self.threshold = max(0.0, float(threshold))
        self.min_interval = max(1, int(min_interval))
        self.warmup_steps = max(0, int(warmup_steps))
        self.fallback_interval = max(0, int(fallback_interval))
        self._last_call_step: int | None = None

    def _can_call(self, step: int) -> bool:
        if self._last_call_step is None:
            return True
        return (step - self._last_call_step) >= self.min_interval

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        del obs  # unused by this scheduler
        if not self._can_call(step):
            return False

        should_call = False
        if latest_cloud_action is None or step < self.warmup_steps:
            should_call = True
        else:
            uncertainty = None if context is None else context.get("edge_uncertainty")
            if isinstance(uncertainty, (int, float)):
                should_call = float(uncertainty) >= self.threshold
            if not should_call and self.fallback_interval > 0 and self._last_call_step is not None:
                should_call = (step - self._last_call_step) >= self.fallback_interval

        if should_call:
            self._last_call_step = step
        return should_call
