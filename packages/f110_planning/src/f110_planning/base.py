"""
Base classes and common types for F1TENTH Planning.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Mapping, NamedTuple


class Action(NamedTuple):
    """
    Represents the control action for a vehicle.

    Attributes:
        steer (float): Steering angle in radians.
        speed (float): Requested longitudinal speed in meters per second.
    """

    steer: float
    speed: float


class CloudTier(str, Enum):
    """Available cloud tiers for multi-tier edge/cloud planning."""

    MEDIUM = "medium"
    LARGE = "large"


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


class TieredCloudScheduler(ABC):  # pylint: disable=too-few-public-methods
    """
    Decide which cloud tier to invoke for multi-tier cloud planning.
    """

    @abstractmethod
    def choose_cloud_tier(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_actions: Mapping[CloudTier, Action | None],
        context: dict[str, Any] | None = None,
    ) -> CloudTier | None:
        """
        Return the cloud tier to request on this step, or ``None`` for edge-only.

        Args:
            step: The current simulation step (0-based).
            obs: The current observation dict.
            latest_cloud_actions: Latest available action for each cloud tier.
            context: Optional runtime context from the planner.

        Returns:
            Requested cloud tier or ``None`` if no cloud request should be issued.
        """
