"""
Utilities for reactive navigation, including coordinate conversions and LiDAR mappings.
"""

from typing import Any

import numpy as np

from .pure_pursuit_utils import get_actuation

# F1TENTH Vehicle Constants
F110_WIDTH = 0.31  # meters
F110_LENGTH = 0.58  # meters
F110_MAX_STEER = 0.4189  # approx 24 degrees in radians
F110_WHEELBASE = 0.33  # meters

# LiDAR Constants (standard f110_gym config)
LIDAR_FOV = 4.7  # radians (approx 270 degrees)
LIDAR_MIN_ANGLE = -LIDAR_FOV / 2  # -2.35 rad
LIDAR_MAX_ANGLE = LIDAR_FOV / 2  # 2.35 rad


def get_reactive_actuation(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    left_dist: float,
    right_dist: float,
    heading_error: float,
    car_position: np.ndarray,
    car_theta: float,
    current_speed: float,
    lookahead_gain: float,
    max_speed: float,
    wheelbase: float,
    lateral_gain: float,
) -> tuple[np.ndarray, float, float]:
    """
    Computes a dynamic waypoint and actuation using reactive features.

    This function centers the car between walls based on distance sensors and
    aligns it with predicted heading errors, effectively acting as a reactive
    controller that mimics human lane-keeping behavior.

    Args:
        left_dist: Predicted or sensed distance to the left wall.
        right_dist: Predicted or sensed distance to the right wall.
        heading_error: Predicted orientation error relative to the corridor.
        car_position: Current [x, y] coordinates of the vehicle.
        car_theta: Current orientation of the vehicle in radians.
        current_speed: Current longitudinal velocity.
        lookahead_gain: Scaling factor for the adaptive lookahead distance.
        max_speed: Maximum allow speed for the plan.
        wheelbase: Physical distance between front and rear axles.
        lateral_gain: Scaling factor for the lateral correction.

    Returns:
        tuple: (target_point_global, steering_angle, speed)
            - target_point_global: [x, y, target_speed] in the map frame.
            - steering_angle: Calculated steering command in radians.
            - speed: Calculated velocity command in m/s.
    """
    lookahead = max(0.8, (lookahead_gain / 5.0) * current_speed + 0.5)

    target_y_vehicle = lateral_gain * (
        (left_dist - right_dist) / 2.0 + np.sin(heading_error) * lookahead
    )

    target_point = np.array(
        [
            car_position[0]
            + lookahead * np.cos(car_theta)
            - target_y_vehicle * np.sin(car_theta),
            car_position[1]
            + lookahead * np.sin(car_theta)
            + target_y_vehicle * np.cos(car_theta),
            max(
                2.5,
                max_speed
                / (
                    1.0
                    + 2.0
                    * (
                        np.abs(heading_error)
                        + np.abs(left_dist - right_dist)
                        / (left_dist + right_dist + 1e-6)
                    )
                ),
            ),
        ]
    )

    _, steering_angle = get_actuation(
        car_theta,
        target_point,
        car_position,
        lookahead,
        wheelbase,
    )

    final_speed = target_point[2] * max(
        0.4,
        1.0 - (np.abs(steering_angle) * current_speed / (F110_MAX_STEER * max_speed)),
    )

    return target_point, steering_angle, final_speed


def get_reactive_action(planner: Any, **kwargs: Any) -> Any:
    """
    Wraps reactive actuation into a standard Action object for use in planners.

    Args:
        planner: High-level planner instance (e.g., LidarDNNPlanner).
        **kwargs: Additional arguments for get_reactive_actuation (dists, errors, position).

    Returns:
        An Action instance containing the calculated steering and speed.
    """
    from ..base import Action  # pylint: disable=import-outside-toplevel

    target_point, steering_angle, speed = get_reactive_actuation(
        **kwargs,
        lookahead_gain=planner.lookahead_distance,
        max_speed=planner.max_speed,
        wheelbase=planner.wheelbase,
        lateral_gain=planner.lateral_gain,
    )
    planner.last_target_point = target_point
    return Action(steer=steering_angle, speed=speed)
