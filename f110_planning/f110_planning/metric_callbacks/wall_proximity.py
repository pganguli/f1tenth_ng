"""
Wall proximity (safety) metric.
"""

from typing import Any

import numpy as np

from f110_planning.base import Action
from f110_planning.utils import get_side_distances

from .base import BaseMetric


class WallProximityMetric(BaseMetric):
    """
    Tracks how close the vehicle comes to walls during an episode.

    At every step the minimum of left and right wall distances (from LiDAR)
    is recorded.  The global minimum over the episode answers *"how close
    did we get to crashing?"* while the mean captures overall centering
    quality.
    """

    def __init__(self) -> None:
        self._closest_per_step: list[float] = []

    @property
    def name(self) -> str:
        return "Wall Proximity"

    def on_reset(
        self,
        obs: dict[str, Any],
        waypoints: np.ndarray | None = None,
    ) -> None:
        self._closest_per_step = []

    def on_step(
        self,
        obs: dict[str, Any],
        action: Action,
        reward: float,
        ego_idx: int = 0,
    ) -> None:
        scan = obs["scans"][ego_idx]
        left_dist, right_dist = get_side_distances(scan)
        self._closest_per_step.append(min(float(left_dist), float(right_dist)))

    def report(self) -> dict[str, float]:
        arr = np.array(self._closest_per_step)
        if arr.size == 0:
            return {
                "wall_min_distance_m": 0.0,
                "wall_mean_distance_m": 0.0,
                "wall_std_distance_m": 0.0,
            }
        return {
            "wall_min_distance_m": round(float(np.min(arr)), 4),
            "wall_mean_distance_m": round(float(np.mean(arr)), 4),
            "wall_std_distance_m": round(float(np.std(arr)), 4),
        }
