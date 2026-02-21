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
    ) -> bool:
        """
        Return ``True`` to issue a cloud request on this step.

        Args:
            step: The current simulation step (0-based).
            obs: The current observation dict.
            latest_cloud_action: The most recent cloud action received
                (``None`` if no cloud result has arrived yet).

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
    ) -> bool:
        return step % self.interval == 0


class AlwaysCallScheduler(CloudScheduler):  # pylint: disable=too-few-public-methods
    """Calls the cloud on every single step (the default behaviour)."""

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Action | None,
    ) -> bool:
        return True
