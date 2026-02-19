"""
Dynamic Waypoint Planner

This planner dynamically computes an imaginary waypoint at each step based on:
- Left and right wall distances (for lateral centering)
- Heading error angle (for angular alignment)
The car follows this continuously updated waypoint to stay centered and aligned.
"""

from typing import Any

import numpy as np

from ..base import Action, BasePlanner
from ..utils import (
    get_heading_error,
    get_reactive_action,
    get_side_distances,
)


class DynamicWaypointPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    Adaptive Dynamic waypoint following planner for F1TENTH

    This planner creates an imaginary waypoint ahead of the car at each timestep,
    positioned to simultaneously:
    1. Center the car between left and right walls (Reactive centering)
    2. Align the car's heading with the track direction (Heading alignment)

    The lookahead distance and target speed are adjusted adaptively based on
    the car's current velocity and the local track curvature. By continuously
    tracking this dynamically computed waypoint, the planner achieves behavior
    similar to Pure Pursuit on actual raceline waypoints without needing a
    pre-planned global path.

    Args:
        waypoints (np.ndarray): Reference waypoints for heading error computation
        lookahead_distance (float, default=1.0): Tuning gain for adaptive lookahead.
            Scaling this increases/decreases the speed-based lookahead window.
        max_speed (float, default=5.0): Maximum speed on straight sections.
            Planner will automatically decelerate for turns and centering maneuvers.
        wheelbase (float, default=0.33): Vehicle wheelbase for steering computation.
        lateral_gain (float, default=1.0): Multiplier for lateral centering aggressiveness.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        waypoints: np.ndarray,
        lookahead_distance: float = 1.0,  # Tuning gain for adaptive velocity-based lookahead
        max_speed: float = 5.0,
        wheelbase: float = 0.33,
        lateral_gain: float = 1.0,
    ):
        self.waypoints = waypoints
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.wheelbase = wheelbase
        self.lateral_gain = lateral_gain
        self.last_target_point = None  # To be picked up by renderer

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:  # pylint: disable=too-many-locals
        """
        Plan action by computing dynamic waypoint from sensor data.

        Args:
            obs (dict): Observation dictionary containing scans and poses
            ego_idx (int): Index of ego vehicle

        Returns:
            Action: Steering angle and speed command
        """
        scan = obs["scans"][ego_idx]
        car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
        car_theta = obs["poses_theta"][ego_idx]
        current_speed = obs["linear_vels_x"][ego_idx]

        # Get sensor-based metrics
        left_dist, right_dist = get_side_distances(scan)
        heading_error = get_heading_error(self.waypoints, car_position, car_theta)

        # Compute dynamic waypoint and actuation using shared logic helper
        return get_reactive_action(
            self,
            left_dist=left_dist,
            right_dist=right_dist,
            heading_error=heading_error,
            car_position=car_position,
            car_theta=car_theta,
            current_speed=current_speed,
        )
