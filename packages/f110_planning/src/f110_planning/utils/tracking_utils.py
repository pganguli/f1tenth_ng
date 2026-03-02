"""
Utility functions for tracking planners.
"""

from typing import Any

import numpy as np
from numba import njit

from .geometry_utils import pi_2_pi
from .pure_pursuit_utils import nearest_point


def get_vehicle_state(obs: dict[str, Any], ego_idx: int) -> np.ndarray:
    """
    Extracts the vehicle's pose and velocity from a gym observation dictionary.

    Args:
        obs: The raw observation dictionary from the environment.
        ego_idx: The index of the vehicle to extract state for.

    Returns:
        A numpy array containing [x, y, theta, linear_vel_x].
    """
    return np.array(
        [
            obs["poses_x"][ego_idx],
            obs["poses_y"][ego_idx],
            obs["poses_theta"][ego_idx],
            obs["linear_vels_x"][ego_idx],
        ]
    )


@njit(cache=True)
def calculate_tracking_errors(  # pylint: disable=too-many-locals
    vehicle_state: np.ndarray, waypoints: np.ndarray, wheelbase: float
) -> tuple[float, float, int, float]:
    """
    Calculates lateral and heading errors relative to the front axle hub.

    This projection allows controllers like Stanley to compensate for the
    vehicle's non-holonomic constraints more effectively by tracking
    from the steering pivot.

    Args:
        vehicle_state: Current state [x, y, heading, velocity].
        waypoints: Reference path coordinates [x, y, ...].
        wheelbase: Physical distance between front and rear axles.

    Returns:
        tuple: (theta_e, ef, target_index, goal_velocity)
            - theta_e: Heading error in radians.
            - ef: Lateral crosstrack error at the front axle.
            - target_index: Index of the closest path segment.
            - goal_velocity: Target speed at that segment.
    """
    fx = vehicle_state[0] + wheelbase * np.cos(vehicle_state[2])
    fy = vehicle_state[1] + wheelbase * np.sin(vehicle_state[2])
    position_front_axle = np.array([fx, fy])
    nearest_point_front, _, _, target_index = nearest_point(
        position_front_axle, waypoints[:, 0:2]
    )
    vec_dist_nearest_point = position_front_axle - nearest_point_front

    front_axle_vec_rot_90 = np.array(
        [
            [np.cos(vehicle_state[2] - np.pi / 2.0)],
            [np.sin(vehicle_state[2] - np.pi / 2.0)],
        ]
    )
    ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

    next_index = (target_index + 1) % len(waypoints)
    dx = waypoints[next_index, 0] - waypoints[target_index, 0]
    dy = waypoints[next_index, 1] - waypoints[target_index, 1]
    theta_raceline = np.arctan2(dy, dx)
    theta_e = pi_2_pi(theta_raceline - vehicle_state[2])

    goal_velocity = vehicle_state[3]

    return theta_e, ef[0], target_index, goal_velocity
