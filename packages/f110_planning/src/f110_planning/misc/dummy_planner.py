"""
Dummy planner module.
"""

from typing import Any

import numpy as np

from ..base import Action, BasePlanner


class DummyPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    A dummy planner that always returns a constant action.
    """

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        return Action(steer=np.pi / 2, speed=1)
