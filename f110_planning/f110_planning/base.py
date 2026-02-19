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
