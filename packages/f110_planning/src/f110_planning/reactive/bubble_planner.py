"""
Bubble planner for obstacle avoidance.

Logic: Local Repulsion.
Mechanism: This planner identifies the single closest obstacle within a defined
'safety_radius' from the LiDAR scan. It then calculates a steering angle
that points directly away from that obstacle (opposite direction).
"""

from typing import Any

import numpy as np
from numba import njit

from ..base import Action, BasePlanner
from ..utils import F110_MAX_STEER, index_to_angle


@njit(cache=True)
def detect_obstacles_jit(lidar_data: np.ndarray, radius: float) -> np.ndarray:
    """
    JIT-optimized obstacle detection.
    """
    num_beams = len(lidar_data)
    # Count obstacles first to pre-allocate
    count = 0
    for d in lidar_data:
        if d <= radius:
            count += 1

    obstacles = np.empty((count, 2))
    idx = 0
    for i, distance in enumerate(lidar_data):
        if distance <= radius:
            obstacles[idx, 0] = index_to_angle(i, num_beams)
            obstacles[idx, 1] = distance
            idx += 1
    return obstacles


class BubblePlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    A reactive planner that creates a 'bubble' around the car and avoids obstacles.
    """

    def __init__(self, safety_radius: float = 1.3, avoidance_speed: float = 2.0) -> None:
        self.safety_radius = safety_radius
        self.avoidance_speed = avoidance_speed

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Plans by moving away from the closest obstacle within the safety radius.
        """
        obstacles = detect_obstacles_jit(obs["scans"][ego_idx], self.safety_radius)
        if len(obstacles) == 0:
            return Action(steer=0.0, speed=self.avoidance_speed * 2)

        # Find the closest obstacle
        idx = np.argmin(obstacles[:, 1])
        closest_angle = obstacles[idx, 0]

        # Point away from obstacle (inverse direction)
        steer = -closest_angle
        steer = np.clip(steer, -F110_MAX_STEER, F110_MAX_STEER)

        return Action(steer=steer, speed=self.avoidance_speed)
