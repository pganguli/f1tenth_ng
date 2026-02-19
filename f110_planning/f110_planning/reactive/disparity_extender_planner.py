"""
Disparity extender planner module.

Logic: Obstacle Inflation.
Mechanism:
1. Searches for 'disparities'—sudden jumps in distance between adjacent LiDAR beams.
2. 'Blooms' or extends the nearer side of that disparity by the car's width to
   mask out 'corners' where the car would otherwise clip its side.
3. Picks the furthest remaining point in the masked scan to steer towards.
"""

from typing import Any

import numpy as np
from numba import njit

from ..base import Action, BasePlanner
from ..utils import (
    F110_MAX_STEER,
    F110_WIDTH,
    LIDAR_FOV,
    LIDAR_MIN_ANGLE,
)


@njit(cache=True)
def find_disparities_jit(values: np.ndarray, threshold: float) -> np.ndarray:
    """JIT-optimized disparity finding."""
    count = 0
    for i, val in enumerate(values[:-1]):
        if abs(val - values[i + 1]) >= threshold:
            count += 1

    disparities = np.empty(count, dtype=np.int32)
    idx = 0
    for i, val in enumerate(values[:-1]):
        if abs(val - values[i + 1]) >= threshold:
            disparities[idx] = i
            idx += 1
    return disparities


@njit(cache=True)
def extend_disparities_jit(  # pylint: disable=too-many-locals
    values: np.ndarray,
    disparities: np.ndarray,
    car_width: float,
    samples_per_radian: float,
) -> tuple[np.ndarray, np.ndarray]:
    """JIT-optimized disparity extension."""
    masked_disparities = np.copy(values)
    possible_disparity_indices = np.empty(len(disparities), dtype=np.int32)

    for i, d in enumerate(disparities):
        a = values[d]
        b = values[d + 1]
        nearer_value = a
        nearer_index = d
        extend_positive = True
        if b < a:
            extend_positive = False
            nearer_value = b
            nearer_index = d + 1

        # Calculate samples to extend
        radians_per_sample = 1.0 / samples_per_radian
        dist_between_samples = nearer_value * radians_per_sample
        samples_to_extend = int(np.ceil(car_width / dist_between_samples))

        current_index = nearer_index
        for _ in range(samples_to_extend):
            if current_index < 0:
                current_index = 0
                break
            if current_index >= len(masked_disparities):
                current_index = len(masked_disparities) - 1
                break
            if masked_disparities[current_index] > nearer_value:
                masked_disparities[current_index] = nearer_value

            if extend_positive:
                current_index += 1
            else:
                current_index -= 1
        possible_disparity_indices[i] = current_index

    return masked_disparities, possible_disparity_indices


@njit(cache=True)
def find_new_angle_jit(
    limited_values: np.ndarray, min_considered_angle: float, samples_per_radian: float
) -> tuple[float, float]:
    """JIT-optimized best angle finding."""
    max_distance = -1.0e10
    angle = 0.0
    found_distance = 0.0

    for i, distance in enumerate(limited_values):
        if distance > max_distance:
            angle = min_considered_angle + float(i) / samples_per_radian
            max_distance = distance
            found_distance = distance
    return found_distance, angle


class DisparityExtenderPlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """
    Reactive planner that uses the disparity extension algorithm to avoid obstacles.
    """

    def __init__(self):
        self.car_width = F110_WIDTH + 0.1
        self.disparity_threshold = 0.5
        self.scan_fov = LIDAR_FOV
        self.turn_clearance = 0.2
        self.max_turn_angle = F110_MAX_STEER
        self.min_speed = 3.0
        self.max_speed = 6.0
        self.absolute_max_speed = 8.0
        self.min_distance = 0.5
        self.no_obstacles_distance = 6.0
        self.min_considered_angle = -np.pi / 2
        self.max_considered_angle = np.pi / 2
        self.lidar_distances: np.ndarray = np.array([])
        self.masked_disparities: np.ndarray = np.array([])
        self.samples_per_radian: float = 0.0
        self.angle: float = 0.0
        self.velocity: float = 0.0

    def get_index(self, angle: float) -> int:
        """Returns the index in the LIDAR samples corresponding to the given
        angle in radians."""
        return int(
            ((angle - LIDAR_MIN_ANGLE) / LIDAR_FOV) * (len(self.lidar_distances) - 1)
        )

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """Processes LIDAR data and returns a steering angle and velocity."""
        self.lidar_distances = np.array(obs["scans"][ego_idx])
        num_beams = len(self.lidar_distances)
        self.samples_per_radian = num_beams / LIDAR_FOV

        # 1. Extend disparities
        disparities = find_disparities_jit(
            self.lidar_distances, self.disparity_threshold
        )
        self.masked_disparities, _ = extend_disparities_jit(
            self.lidar_distances,
            disparities,
            self.car_width,
            self.samples_per_radian,
        )

        # 2. Find best angle in considered range
        min_idx = self.get_index(self.min_considered_angle)
        max_idx = self.get_index(self.max_considered_angle)
        limited_values = self.masked_disparities[min_idx:max_idx]

        forward_dist, target_angle = find_new_angle_jit(
            limited_values, self.min_considered_angle, self.samples_per_radian
        )

        # 3. Corner cutting protection
        # If we are turning, check the side we are turning towards
        num_side_beams = int((45.0 / 180.0 * np.pi) * self.samples_per_radian)
        if target_angle > 0.1:  # Turning left
            side_samples = self.lidar_distances[-num_side_beams:]
            if np.min(side_samples) < self.turn_clearance:
                target_angle = 0.0
        elif target_angle < -0.1:  # Turning right
            side_samples = self.lidar_distances[:num_side_beams]
            if np.min(side_samples) < self.turn_clearance:
                target_angle = 0.0

        # 4. Speed logic
        if forward_dist <= self.min_distance:
            speed = 0.0
        elif forward_dist >= self.no_obstacles_distance:
            speed = self.absolute_max_speed
        else:
            dist_range = self.no_obstacles_distance - self.min_distance
            speed = self.min_speed + (self.max_speed - self.min_speed) * (
                (forward_dist - self.min_distance) / dist_range
            )

        self.angle = np.clip(target_angle, -F110_MAX_STEER, F110_MAX_STEER)
        self.velocity = speed

        return Action(steer=self.angle, speed=self.velocity)
