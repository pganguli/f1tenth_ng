"""
Steering smoothness metric.
"""

from typing import Any

import numpy as np

from f110_planning.base import Action

from .base import BaseMetric


class SmoothnessMetric(BaseMetric):
    """
    Evaluates control smoothness by analysing steering rate of change.

    The steering rate is ``|Δsteer / Δt|`` between consecutive steps.
    High values indicate jerky, oscillatory control which is undesirable
    both for passenger comfort and tyre wear.
    """

    def __init__(self) -> None:
        self._prev_steer: float | None = None
        self._steering_rates: list[float] = []
        self._timestep: float = 0.0

    @property
    def name(self) -> str:
        return "Smoothness"

    def on_reset(
        self,
        obs: dict[str, Any],
        waypoints: np.ndarray | None = None,
    ) -> None:
        self._prev_steer = None
        self._steering_rates = []
        self._timestep = 0.0

    def on_step(
        self,
        obs: dict[str, Any],
        action: Action,
        reward: float,
        ego_idx: int = 0,
    ) -> None:
        dt = reward  # reward == env.timestep
        if dt <= 0:
            return

        self._timestep = dt
        steer = float(action.steer)

        if self._prev_steer is not None:
            rate = abs(steer - self._prev_steer) / dt
            self._steering_rates.append(rate)

        self._prev_steer = steer

    def report(self) -> dict[str, float]:
        arr = np.array(self._steering_rates)
        if arr.size == 0:
            return {
                "steering_rate_mean_rad_s": 0.0,
                "steering_rate_max_rad_s": 0.0,
                "steering_rate_std_rad_s": 0.0,
            }
        return {
            "steering_rate_mean_rad_s": round(float(np.mean(arr)), 4),
            "steering_rate_max_rad_s": round(float(np.max(arr)), 4),
            "steering_rate_std_rad_s": round(float(np.std(arr)), 4),
        }
