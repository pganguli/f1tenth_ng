"""
LQR utilities
"""

import numpy as np
from numba import njit


# pylint: disable=too-many-arguments, too-many-positional-arguments, invalid-name
@njit(cache=True)
def solve_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    tolerance: float,
    max_num_iteration: int,
) -> np.ndarray:
    """
    Solves the Discrete-time Algebraic Riccati Equation (DARE) iteratively.

    This finds the steady-state feedback matrix K that minimizes the quadratic
    cost function of the system.

    Args:
        A: Discrete-time state transition matrix.
        B: Discrete-time input matrix.
        Q: State cost matrix.
        R: Input cost matrix.
        tolerance: Convergence threshold for the P matrix update.
        max_num_iteration: Maximum number of iterations to perform.

    Returns:
        The optimal feedback gain matrix K.
    """

    M = np.zeros((Q.shape[0], R.shape[1]))

    AT = A.T
    BT = B.T
    MT = M.T

    P = Q
    num_iteration = 0
    diff = np.inf

    while num_iteration < max_num_iteration and diff > tolerance:
        num_iteration += 1
        P_next = (
            AT @ P @ A
            - (AT @ P @ B + M) @ np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT)
            + Q
        )

        diff = np.abs(np.max(P_next - P))
        P = P_next

    K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)

    return K


@njit(cache=True)
def update_matrix(
    vehicle_state: np.ndarray, state_size: int, timestep: float, wheelbase: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearizes a kinematic bicycle model into discrete state-space form.

    Args:
        vehicle_state: Current state [x, y, heading, velocity].
        state_size: Dimensions of the state vector.
        timestep: Sampling time in seconds.
        wheelbase: Physical distance between axles.

    Returns:
        A tuple of (Ad, Bd) matrices.
    """
    v = vehicle_state[3]

    matrix_ad_ = np.zeros((state_size, state_size))
    matrix_ad_[0][0] = 1.0
    matrix_ad_[0][1] = timestep
    matrix_ad_[1][2] = v
    matrix_ad_[2][2] = 1.0
    matrix_ad_[2][3] = timestep

    matrix_bd_ = np.zeros((state_size, 1))
    matrix_bd_[3][0] = v / wheelbase

    return matrix_ad_, matrix_bd_
