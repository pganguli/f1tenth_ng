"""
Lap time and completion metric.
"""

from typing import Any

import numpy as np

from f110_planning.base import Action

from .base import BaseMetric


class LapTimeMetric(BaseMetric):
    """
    Tracks simulated lap time, collision status, and lap completion.

    Lap time is accumulated from the per-step reward (which equals
    ``env.timestep``, typically 0.01 s).
    """

    def __init__(self) -> None:
        self._elapsed: float = 0.0
        self._collision: bool = False
        self._laps_completed: int = 0
        self._steps: int = 0

    @property
    def name(self) -> str:
        return "Lap Time"

    def on_reset(
        self,
        obs: dict[str, Any],
        waypoints: np.ndarray | None = None,
    ) -> None:
        self._elapsed = 0.0
        self._collision = False
        self._laps_completed = 0
        self._steps = 0

    def on_step(
        self,
        obs: dict[str, Any],
        action: Action,
        reward: float,
        ego_idx: int = 0,
    ) -> None:
        self._elapsed += reward
        self._steps += 1

        if obs["collisions"][ego_idx] > 0:
            self._collision = True

        self._laps_completed = int(obs["lap_counts"][ego_idx])

    def report(self) -> dict[str, float]:
        return {
            "lap_time_s": round(self._elapsed, 4),
            "steps": float(self._steps),
            "collision": float(self._collision),
            "laps_completed": float(self._laps_completed),
        }
