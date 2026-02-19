"""
Geometry utilities
"""

import numpy as np
from numba import njit


@njit(cache=True)
def pi_2_pi(angle: float) -> float:
    """
    Normalizes an angle to the range [-pi, pi].

    Args:
        angle: The input angle in radians.

    Returns:
        The normalized angle in radians.
    """
    if angle > np.pi:
        return angle - 2.0 * np.pi
    if angle < -np.pi:
        return angle + 2.0 * np.pi

    return angle
