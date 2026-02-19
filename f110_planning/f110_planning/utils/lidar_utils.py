"""
LiDAR and path tracking utility functions.
"""

import numpy as np
from numba import njit

from .geometry_utils import pi_2_pi
from .pure_pursuit_utils import nearest_point
from .reactive_utils import LIDAR_FOV, LIDAR_MIN_ANGLE


@njit(cache=True)
def index_to_angle(i: int, num_beams: int = 1080) -> float:
    """
    Convert a LiDAR scan index to its corresponding angle.

    Args:
        i: Scan index.
        num_beams: Total number of beams in the scan.

    Returns:
        float: Angle in radians.
    """
    return (i / (num_beams - 1)) * LIDAR_FOV + LIDAR_MIN_ANGLE


@njit(cache=True)
def get_side_distances(scan: np.ndarray) -> tuple[float, float]:
    """
    Extracts minimum distances to the left and right walls from raw LiDAR data.

    This uses a small window around the 90-degree and -90-degree points to
    robustly estimate the wall distance even if the car is slightly rotated.

    Args:
        scan: The 1D array of LiDAR distances.

    Returns:
        A tuple (left_min, right_min) in meters.
    """
    num_beams = len(scan)
    angle_increment = LIDAR_FOV / (num_beams - 1)
    lidar_min_angle = -LIDAR_FOV / 2

    left_angle = np.pi / 2
    left_idx = int((left_angle - lidar_min_angle) / angle_increment)

    right_angle = -np.pi / 2
    right_idx = int((right_angle - lidar_min_angle) / angle_increment)

    # Use a 10-degree window for robustness
    window_angle = 10 * np.pi / 180
    window_size = int(window_angle / angle_increment)

    left_min = np.min(
        scan[max(0, left_idx - window_size) : min(num_beams, left_idx + window_size)]
    )
    right_min = np.min(
        scan[max(0, right_idx - window_size) : min(num_beams, right_idx + window_size)]
    )

    return left_min, right_min


@njit(cache=True)
def get_heading_error(
    waypoints: np.ndarray, car_position: np.ndarray, car_theta: float
) -> float:
    """
    Computes the orientation error between the car's heading and the local path segment.

    Args:
        waypoints: Array of waypoints [N, 2+] containing at least [x, y].
        car_position: Current vehicle [x, y] position.
        car_theta: Current orientation of the vehicle in radians.

    Returns:
        Heading error in radians. Positive values indicate the path is to the
        car's left; negative to the right.
    """
    _, _, _, i = nearest_point(car_position, waypoints[:, 0:2])

    # Forming a segment with the next point handles circular tracks via modulo
    next_i = (i + 1) % len(waypoints)
    direction_vector = waypoints[next_i, 0:2] - waypoints[i, 0:2]
    intended_theta = np.arctan2(direction_vector[1], direction_vector[0])

    return pi_2_pi(intended_theta - car_theta)
