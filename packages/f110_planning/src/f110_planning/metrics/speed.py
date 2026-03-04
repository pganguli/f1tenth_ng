"""
Speed profile metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from f110_planning.base import Action

from .base import BaseMetric


def current_speed(obs: dict[str, Any], ego_idx: int = 0) -> float:
    """Return the longitudinal speed of the ego vehicle from an observation.

    Args:
        obs: Observation dictionary produced by ``F110Env.step()`` /
            ``F110Env.reset()``.
        ego_idx: Index of the ego vehicle in multi-agent observations.

    Returns:
        Longitudinal speed in m/s (``obs["linear_vels_x"][ego_idx]``).
    """
    return float(obs["linear_vels_x"][ego_idx])


class SpeedMetric(BaseMetric):
    """
    Records the vehicle's *actual* longitudinal speed throughout an episode.

    Uses ``obs["linear_vels_x"]`` (measured speed, not commanded) so that
    the statistics reflect real vehicle dynamics including slip and
    acceleration limits.
    """

    def __init__(self) -> None:
        self._speeds: list[float] = []

    @property
    def name(self) -> str:
        return "Speed"

    def on_reset(
        self,
        obs: dict[str, Any],
        waypoints: np.ndarray | None = None,
    ) -> None:
        self._speeds = []

    def on_step(
        self,
        obs: dict[str, Any],
        action: Action,
        reward: float,
        ego_idx: int = 0,
    ) -> None:
        self._speeds.append(current_speed(obs, ego_idx))

    def report(self) -> dict[str, float]:
        arr = np.array(self._speeds)
        if arr.size == 0:
            return {
                "speed_mean_m_s": 0.0,
                "speed_max_m_s": 0.0,
                "speed_std_m_s": 0.0,
            }
        return {
            "speed_mean_m_s": round(float(np.mean(arr)), 4),
            "speed_max_m_s": round(float(np.max(arr)), 4),
            "speed_std_m_s": round(float(np.std(arr)), 4),
        }
