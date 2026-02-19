"""
Stanley waypoint tracker
"""

from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner
from ..utils import calculate_tracking_errors, get_vehicle_state


class StanleyPlanner(BasePlanner):
    """
    Front Wheel Feedback Controller (Stanley) for path tracking.

    The Stanley controller combines heading alignment and cross-track error
    compensation (from the perspective of the front axle) into a single
    steering law.

    Reference: Thrun et al, "Stanley: The Robot that Won the DARPA Grand Challenge", 2005.
    """

    def __init__(
        self,
        wheelbase: float = 0.33,
        waypoints: Optional[np.ndarray] = None,
        max_speed: float = 5.0,
        k_path: float = 5.0,
    ):
        """
        Initializes the Stanley planner.

        Args:
            wheelbase: Front-to-rear axle distance in meters.
            waypoints: Reference path [N, 2+].
            max_speed: Target longitudinal velocity.
            k_path: Gain for the cross-track error compensation ($k$).
        """
        self.wheelbase = wheelbase
        self.waypoints = waypoints if waypoints is not None else np.array([])
        self.max_speed = max_speed
        self.k_path = k_path

    def calc_theta_and_ef(
        self, vehicle_state: np.ndarray, waypoints: np.ndarray
    ) -> tuple[float, float, int, float]:
        """
        Computes the current tracking errors relative to the front axle hub.
        """
        theta_e, ef, target_index, _ = calculate_tracking_errors(
            vehicle_state, waypoints, self.wheelbase
        )

        return theta_e, ef, target_index, self.max_speed

    def controller(
        self, vehicle_state: np.ndarray, waypoints: np.ndarray, k_path: float
    ) -> tuple[float, float]:
        """
        Computes steering and speed using the Stanley control law.

        The steering angle delta is defined as:
        delta(t) = theta_e(t) + arctan((k * e(t)) / v(t))
        """
        theta_e, ef, _, goal_velocity = self.calc_theta_and_ef(vehicle_state, waypoints)

        # Non-linear gain inversely proportional to speed
        cte_front = np.atan2(k_path * ef, vehicle_state[3] + 1e-6)
        delta = cte_front + theta_e

        return delta, goal_velocity

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Determines the next control action for the vehicle.
        """
        if self.waypoints is None or len(self.waypoints) == 0:
            raise ValueError("Waypoints must be provided to the planner.")

        vehicle_state = get_vehicle_state(obs, ego_idx)
        steering_angle, speed = self.controller(
            vehicle_state, self.waypoints, self.k_path
        )

        return Action(steer=steering_angle, speed=speed)
