"""
Random planner module.
"""

import random
from typing import Any

from ..base import Action, BasePlanner


class RandomPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    Experimental planner that yields stochastic control actions.

    Used primarily for environment stress-testing and data diversity generation.
    """

    def __init__(
        self,
        s_min: float = -0.4189,
        s_max: float = 0.4189,
        v_min: float = 0.5,
        v_max: float = 5.0,
    ):
        """
        Initializes the random ranges for control.

        Args:
            s_min: Minimum steering angle in radians.
            s_max: Maximum steering angle in radians.
            v_min: Minimum longitudinal velocity.
            v_max: Maximum longitudinal velocity.
        """
        self.s_min = s_min
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Generates a random steering and speed command.
        """
        speed = random.uniform(self.v_min, self.v_max)
        steer = random.uniform(self.s_min, self.s_max)
        return Action(steer=steer, speed=speed)
