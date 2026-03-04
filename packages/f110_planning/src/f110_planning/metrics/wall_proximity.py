"""
Wall proximity (safety) metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from f110_planning.base import Action
from ..utils.lidar_utils import get_side_distances

from .base import BaseMetric


def min_wall_dist(scan: np.ndarray) -> float:
    """Return the minimum of the left and right wall distances from a LiDAR scan.

    Args:
        scan: ``(num_beams,)`` LiDAR range scan for a single agent.

    Returns:
        Minimum side-clearance in metres.
    """
    left_dist, right_dist = get_side_distances(scan)
    return float(min(left_dist, right_dist))


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
        self._closest_per_step.append(min_wall_dist(scan))

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
