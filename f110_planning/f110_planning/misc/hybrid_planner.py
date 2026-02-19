"""
Hybrid planner module combining manual and autonomous control.
"""

from typing import Any

from pyglet import window as pyg_window

from ..base import Action, BasePlanner
from .manual_planner import ManualPlanner


class HybridPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    Arbitrator that delegates control based on human input.

    The Hybrid Planner monitors the keyboard state. If any movement keys
    ('W', 'A', 'S', 'D') are active, it hands over control to the human
    operator. Otherwise, it defaults to the provided autonomous planner.
    """

    def __init__(self, manual_planner: ManualPlanner, auto_planner: BasePlanner) -> None:
        """
        Initializes the hybrid control system.

        Args:
            manual_planner: An initialized ManualPlanner instance.
            auto_planner: Any autonomous planner implementing BasePlanner.
        """
        self.manual = manual_planner
        self.auto = auto_planner

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Switches between human and autonomous control in real-time.
        """
        keys = self.manual.keys
        # Check if the human is providing any steering or throttle input
        manual_override = any(
            keys[k]
            for k in [
                pyg_window.key.W,
                pyg_window.key.A,
                pyg_window.key.S,
                pyg_window.key.D,
            ]
        )

        if manual_override:
            return self.manual.plan(obs, ego_idx)
        return self.auto.plan(obs, ego_idx)
