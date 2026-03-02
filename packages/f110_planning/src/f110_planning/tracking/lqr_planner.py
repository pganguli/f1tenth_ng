"""
LQR waypoint tracker
Implementation inspired by
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/lqr_steer_control/lqr_steer_control.py
"""

from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner
from ..utils import (
    calculate_tracking_errors,
    get_vehicle_state,
    solve_lqr,
    update_matrix,
)


class LQRPlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """
    Lateral Controller using Linear Quadratic Regulator (LQR).

    This planner linearizes the vehicle dynamics around a reference path and
    solves for an optimal gain matrix K that minimizes a cost function
    representing both tracking error and control effort.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        wheelbase: float = 0.33,
        waypoints: Optional[np.ndarray] = None,
        max_speed: float = 5.0,
        timestep: float = 0.01,
        matrix_q_1: float = 0.999,
        matrix_q_2: float = 0.0,
        matrix_q_3: float = 0.0066,
        matrix_q_4: float = 0.0,
        matrix_r: float = 0.75,
        iterations: int = 50,
        eps: float = 0.001,
    ) -> None:
        """
        Initializes the LQR planner with control weights and system parameters.

        Args:
            wheelbase: Front-to-rear axle distance.
            waypoints: Loaded path coordinates [N, 2+].
            max_speed: Target longitudinal velocity.
            timestep: Integration step for discretization.
            matrix_q_1..4: Diagonal elements of the state cost matrix Q.
            matrix_r: Element of the input cost matrix R.
            iterations: Maximum iterations for the DARE solver.
            eps: Convergence tolerance for the solver.
        """
        self.wheelbase = wheelbase
        self.waypoints = waypoints if waypoints is not None else np.array([])
        self.max_speed = max_speed
        self.vehicle_control_e_cog = 0.0
        self.vehicle_control_theta_e = 0.0
        self.timestep = timestep
        self.matrix_q_1 = matrix_q_1
        self.matrix_q_2 = matrix_q_2
        self.matrix_q_3 = matrix_q_3
        self.matrix_q_4 = matrix_q_4
        self.matrix_r = matrix_r
        self.iterations = iterations
        self.eps = eps

    def calc_control_points(
        self, vehicle_state: np.ndarray, waypoints: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        """
        Calculates the current state errors and reference curvature.
        """

        theta_e, ef, target_index, _ = calculate_tracking_errors(
            vehicle_state, waypoints, self.wheelbase
        )

        next_index = (target_index + 1) % len(waypoints)
        dx = waypoints[next_index, 0] - waypoints[target_index, 0]
        dy = waypoints[next_index, 1] - waypoints[target_index, 1]
        theta_raceline = np.arctan2(dy, dx)

        kappa_ref = 0.0
        goal_velocity = self.max_speed

        self.vehicle_control_e_cog = ef
        self.vehicle_control_theta_e = theta_e

        return theta_e, ef, theta_raceline, kappa_ref, goal_velocity

    def controller(self, vehicle_state: np.ndarray) -> tuple[float, float]:
        """
        Optimal lateral control calculation.

        This discrete-time LQR implementation computes steering based on a
        4-state error vector: [crosstrack, lateral_velocity, heading, heading_rate].
        """
        # Construct cost matrices from class attributes
        matrix_q = np.diag(
            [self.matrix_q_1, self.matrix_q_2, self.matrix_q_3, self.matrix_q_4]
        )
        matrix_r = np.diag([self.matrix_r])

        # Get tracking errors and reference values
        theta_e, e_cg, _, k_ref, v_ref = self.calc_control_points(
            vehicle_state, self.waypoints
        )

        # Linearize vehicle dynamics and solve the Riccati equation
        ad_mat, bd_mat = update_matrix(vehicle_state, 4, self.timestep, self.wheelbase)
        k_mat = solve_lqr(ad_mat, bd_mat, matrix_q, matrix_r, self.eps, self.iterations)

        # Build current state vector for feedback
        matrix_state = np.zeros((4, 1))
        matrix_state[0, 0] = e_cg
        matrix_state[1, 0] = (e_cg - self.vehicle_control_e_cog) / self.timestep
        matrix_state[2, 0] = theta_e
        matrix_state[3, 0] = (theta_e - self.vehicle_control_theta_e) / self.timestep

        # LQR control law: steering = feedback + feedforward
        steer_angle = (k_mat @ matrix_state)[0, 0] + (k_ref * self.wheelbase)

        return steer_angle, v_ref

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Compute lateral control command.
        """
        if self.waypoints is None or len(self.waypoints) == 0:
            raise ValueError(
                "Please set waypoints to track during planner instantiation or when calling plan()"
            )

        # Execute LQR control law
        vehicle_state = get_vehicle_state(obs, ego_idx)
        steering_angle, speed = self.controller(vehicle_state)

        return Action(steer=steering_angle, speed=speed)
