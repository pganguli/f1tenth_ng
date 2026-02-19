"""
Prototype of vehicle dynamics functions and classes for simulating 2D Single
Track dynamic model
Following the implementation of commanroad's Single Track Dynamics model
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
Author: Hongrui Zheng
"""

import numpy as np
from numba import njit

from .vehicle_params import VehicleParams

# Constants
GRAVITY = 9.81


@njit(cache=True)
def accl_constraints(
    vel: float,
    accl: float,
    params: VehicleParams,
) -> float:
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            params (VehicleParams): vehicle parameters

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    if vel > params.v_switch:
        pos_limit = params.a_max * params.v_switch / vel
    else:
        pos_limit = params.a_max

    # accl limit reached?
    if (vel <= params.v_min and accl <= 0) or (vel >= params.v_max and accl >= 0):
        accl = 0.0
    elif accl <= -params.a_max:
        accl = -params.a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl


@njit(cache=True)
def steering_constraint(
    steering_angle: float,
    steering_velocity: float,
    params: VehicleParams,
) -> float:
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            params (VehicleParams): vehicle parameters

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    if (steering_angle <= params.s_min and steering_velocity <= 0) or (
        steering_angle >= params.s_max and steering_velocity >= 0
    ):
        steering_velocity = 0.0
    elif steering_velocity <= params.sv_min:
        steering_velocity = params.sv_min
    elif steering_velocity >= params.sv_max:
        steering_velocity = params.sv_max

    return steering_velocity


@njit(cache=True)
def vehicle_dynamics_ks(
    x: np.ndarray,
    u_init: np.ndarray,
    params: VehicleParams,
) -> np.ndarray:
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = params.lf + params.lr

    # constraints
    u = np.array(
        [
            steering_constraint(x[2], u_init[0], params),
            accl_constraints(x[3], u_init[1], params),
        ]
    )

    # system dynamics
    f = np.array(
        [
            x[3] * np.cos(x[4]),
            x[3] * np.sin(x[4]),
            u[0],
            u[1],
            x[3] / lwb * np.tan(x[2]),
        ]
    )
    return f


@njit(cache=True)
def vehicle_dynamics_st(
    x: np.ndarray,
    u_init: np.ndarray,
    params: VehicleParams,
) -> np.ndarray:
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = GRAVITY

    # constraints
    u = np.array(
        [
            steering_constraint(x[2], u_init[0], params),
            accl_constraints(x[3], u_init[1], params),
        ]
    )

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.5:
        # wheelbase
        lwb = params.lf + params.lr

        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(
            x_ks,
            u,
            params,
        )
        f = np.hstack(
            (
                f_ks,
                np.array(
                    [
                        u[1] / lwb * np.tan(x[2])
                        + x[3] / (lwb * np.cos(x[2]) ** 2) * u[0],
                        0,
                    ]
                ),
            )
        )

    else:
        # system dynamics
        f = np.array(
            [
                x[3] * np.cos(x[6] + x[4]),
                x[3] * np.sin(x[6] + x[4]),
                u[0],
                u[1],
                x[5],
                -params.mu
                * params.m
                / (x[3] * params.MoI * (params.lr + params.lf))
                * (
                    params.lf**2 * params.C_Sf * (g * params.lr - u[1] * params.h)
                    + params.lr**2 * params.C_Sr * (g * params.lf + u[1] * params.h)
                )
                * x[5]
                + params.mu
                * params.m
                / (params.MoI * (params.lr + params.lf))
                * (
                    params.lr * params.C_Sr * (g * params.lf + u[1] * params.h)
                    - params.lf * params.C_Sf * (g * params.lr - u[1] * params.h)
                )
                * x[6]
                + params.mu
                * params.m
                / (params.MoI * (params.lr + params.lf))
                * params.lf
                * params.C_Sf
                * (g * params.lr - u[1] * params.h)
                * x[2],
                (
                    params.mu
                    / (x[3] ** 2 * (params.lr + params.lf))
                    * (
                        params.C_Sr * (g * params.lf + u[1] * params.h) * params.lr
                        - params.C_Sf * (g * params.lr - u[1] * params.h) * params.lf
                    )
                    - 1
                )
                * x[5]
                - params.mu
                / (x[3] * (params.lr + params.lf))
                * (
                    params.C_Sr * (g * params.lf + u[1] * params.h)
                    + params.C_Sf * (g * params.lr - u[1] * params.h)
                )
                * x[6]
                + params.mu
                / (x[3] * (params.lr + params.lf))
                * (params.C_Sf * (g * params.lr - u[1] * params.h))
                * x[2],
            ]
        )

    return f


@njit(cache=True)
def pid(
    speed: float,
    steer: float,
    current_speed: float,
    current_steer: float,
    params: VehicleParams,
) -> tuple[float, float]:
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
    """
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * params.sv_max
    else:
        sv = 0.0

    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.0:
        if vel_diff > 0:
            # accelerate
            kp = 10.0 * params.a_max / params.v_max
            accl = kp * vel_diff
        else:
            # braking
            kp = 10.0 * params.a_max / (-params.v_min)
            accl = kp * vel_diff
    # currently backwards
    else:
        if vel_diff > 0:
            # braking
            kp = 2.0 * params.a_max / params.v_max
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * params.a_max / (-params.v_min)
            accl = kp * vel_diff

    return accl, sv
