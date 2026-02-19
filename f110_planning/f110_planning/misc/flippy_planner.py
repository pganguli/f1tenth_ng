"""
Flippy planner module.
"""

from typing import Any

from ..base import Action, BasePlanner


class FlippyPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    Planner designed to exploit integration methods and dynamics.
    For testing only. To observe this error, use single track dynamics for all velocities >0.1
    """

    def __init__(
        self,
        flip_every: int = 1,
        steer: float = 2,
        speed: float = 1,
    ) -> None:
        self.flip_every = flip_every
        self.steer = steer
        self.speed = speed
        self.counter = 0

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        self.counter += 1
        if self.counter % self.flip_every == 0:
            self.counter = 0
            self.steer *= -1
        return Action(steer=self.steer, speed=self.speed)
