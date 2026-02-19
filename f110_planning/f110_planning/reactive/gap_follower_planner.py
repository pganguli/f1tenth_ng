"""
Gap Follower Planner (FGM) for F1TENTH.

Logic: Contiguous Free Space.
Mechanism:
1. Finds the closest obstacle and sets a 'bubble' of nearby LiDAR beams to zero.
2. Identifies the largest contiguous sequence (the 'gap') of non-zero LiDAR beams.
3. Finds the 'best point' (usually the midpoint or furthest point) within that
   specific gap and steers towards it.
"""

from typing import Any

import numpy as np
from numba import njit

from ..base import Action, BasePlanner
from ..utils import (
    F110_MAX_STEER,
    LIDAR_FOV,
    LIDAR_MIN_ANGLE,
    index_to_angle,
)

# Constants
LIDAR_MIN_CONSIDERED_ANGLE = -np.pi / 2
LIDAR_MAX_CONSIDERED_ANGLE = np.pi / 2
MAX_LIDAR_DIST = 10.0  # meters


@njit(cache=True)
def find_max_gap_jit(free_space_ranges: np.ndarray) -> tuple[int, int]:
    """
    Identifies the largest contiguous sequence of non-zero values in a scan.

    Args:
        free_space_ranges: Array of LiDAR ranges where obstacles have been masked.

    Returns:
        A tuple of (start_index, end_index) for the largest gap.
    """
    max_start = 0
    max_end = 0
    max_len = 0

    current_start = -1
    for i, val in enumerate(free_space_ranges):
        if val != 0:
            if current_start == -1:
                current_start = i
        else:
            if current_start != -1:
                current_len = i - current_start
                if current_len > max_len:
                    max_len = current_len
                    max_start = current_start
                    max_end = i
                current_start = -1

    if current_start != -1:
        current_len = len(free_space_ranges) - current_start
        if current_len > max_len:
            max_start = current_start
            max_end = len(free_space_ranges)

    return max_start, max_end


@njit(cache=True)
def find_best_point_jit(
    start_i: int, end_i: int, ranges: np.ndarray, best_point_conv_size: int
) -> int:
    """
    Finds the deepest point within a gap using a sliding window average.

    Args:
        start_i: Start index of the gap.
        end_i: End index of the gap.
        ranges: Original LiDAR ranges.
        best_point_conv_size: Size of the smoothing window.

    Returns:
        The index of the chosen target point.
    """
    sub_ranges = ranges[start_i:end_i]
    if len(sub_ranges) == 0:
        return start_i

    n = len(sub_ranges)
    k = best_point_conv_size

    if n < k:
        max_val = -1.0
        max_idx = 0
        for i in range(n):
            if sub_ranges[i] > max_val:
                max_val = sub_ranges[i]
                max_idx = i
        return max_idx + start_i

    current_sum = 0.0
    for i in range(k):
        current_sum += sub_ranges[i]

    max_sum = current_sum
    max_idx = k // 2

    for i in range(1, n - k + 1):
        current_sum = current_sum - sub_ranges[i - 1] + sub_ranges[i + k - 1]
        if current_sum > max_sum:
            max_sum = current_sum
            max_idx = i + k // 2

    return max_idx + start_i


class GapFollowerPlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """
    Reactive planner that steers toward the largest contiguous gap in LiDAR scans.

    The "Follow the Gap" (FGM) algorithm works by:
    1. Preprocessing the scan to ignore the area behind the vehicle.
    2. Placing a "virtual bubble" around the closest obstacle to avoid collisions.
    3. Identifying the largest sequence of obstacle-free beams (the gap).
    4. Navigating toward the deepest point within that gap.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        bubble_radius: int = 160,
        corners_speed: float = 4.0,
        straights_speed: float = 6.0,
        straights_steering_angle: float = np.pi / 18,
        preprocess_conv_size: int = 3,
        best_point_conv_size: int = 80,
        max_lidar_dist: float = MAX_LIDAR_DIST,
    ):
        """
        Initializes the Gap Follower with specific tuning parameters.

        Args:
            bubble_radius: Number of beams to zero-out around the closest obstacle.
            corners_speed: Velocity used when steering is significant.
            straights_speed: Velocity used when the path is relatively straight.
            straights_steering_angle: Threshold for switching between corner and straight speed.
            preprocess_conv_size: Window size for initial LiDAR scan smoothing.
            best_point_conv_size: Window size for finding the deepest part of a gap.
            max_lidar_dist: Clipping distance for LiDAR sensor data.
        """
        self.bubble_radius = bubble_radius
        self.corners_speed = corners_speed
        self.straights_speed = straights_speed
        self.straights_steering_angle = straights_steering_angle
        self.preprocess_conv_size = preprocess_conv_size
        self.best_point_conv_size = best_point_conv_size
        self.max_lidar_dist = max_lidar_dist

    def preprocess_lidar(self, ranges: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Slices the LiDAR scan to the front hemisphere and applies smoothing.
        """
        num_beams = len(ranges)
        idx_start = int(
            ((LIDAR_MIN_CONSIDERED_ANGLE - LIDAR_MIN_ANGLE) / LIDAR_FOV)
            * (num_beams - 1)
        )
        idx_end = int(
            ((LIDAR_MAX_CONSIDERED_ANGLE - LIDAR_MIN_ANGLE) / LIDAR_FOV)
            * (num_beams - 1)
        )

        proc_ranges = np.array(ranges[idx_start:idx_end])
        proc_ranges = (
            np.convolve(proc_ranges, np.ones(self.preprocess_conv_size), "same")
            / self.preprocess_conv_size
        )
        proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_dist)
        return proc_ranges, idx_start, num_beams

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:  # pylint: disable=too-many-locals
        """
        Computes the next steering and speed action based on the 'Follow the Gap' logic.
        """
        ranges = obs["scans"][ego_idx]
        proc_ranges, idx_offset, num_beams = self.preprocess_lidar(ranges)

        closest = proc_ranges.argmin()

        min_index = max(closest - self.bubble_radius, 0)
        max_index = min(closest + self.bubble_radius, len(proc_ranges) - 1)
        proc_ranges[min_index:max_index] = 0

        gap_start, gap_end = find_max_gap_jit(proc_ranges)

        best = find_best_point_jit(
            gap_start, gap_end, proc_ranges, self.best_point_conv_size
        )

        final_idx = best + idx_offset
        steering_angle = index_to_angle(final_idx, num_beams)
        steering_angle = np.clip(steering_angle, -F110_MAX_STEER, F110_MAX_STEER)

        mode_speed = (
            self.straights_speed
            if abs(steering_angle) < self.straights_steering_angle
            else self.corners_speed
        )

        return Action(steer=steering_angle, speed=mode_speed)
