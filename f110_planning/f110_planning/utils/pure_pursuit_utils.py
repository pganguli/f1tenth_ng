"""
Pure Pursuit utilities
"""

from typing import Optional

import numpy as np
from numba import njit


@njit(cache=True)
def nearest_point(
    point: np.ndarray, trajectory: np.ndarray
) -> tuple[np.ndarray, float, float, int]:
    """
    Return the nearest point along the given piecewise linear trajectory.

    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique,
            a divide by 0 error will destroy the world

    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector
            formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        float(dists[min_dist_segment]),
        float(t[min_dist_segment]),
        int(min_dist_segment),
    )


@njit(cache=True)
def _get_intersections(
    point: np.ndarray, radius: float, p1: np.ndarray, v_vec: np.ndarray
) -> tuple[float, float]:
    """
    Calculates the normalized intersection times of a circle and a line segment.

    Args:
        point: Center of the circle (vehicle position).
        radius: Radius of the circle (lookahead distance).
        p1: Start point of the line segment.
        v_vec: Direction vector of the line segment (p2 - p1).

    Returns:
        A tuple of (t1, t2) representing the intersection parameters along v_vec.
        Returns (NaN, NaN) if no intersection occurs.
    """
    a = np.dot(v_vec, v_vec)
    b = 2.0 * np.dot(v_vec, p1 - point)
    c = np.dot(p1, p1) + np.dot(point, point) - 2.0 * np.dot(p1, point) - radius**2
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.nan, np.nan
    disc = np.sqrt(disc)
    return (-b - disc) / (2.0 * a), (-b + disc) / (2.0 * a)


@njit(cache=True)
def intersect_point(
    point: np.ndarray,
    radius: float,
    trajectory: np.ndarray,
    t: float = 0.0,
    wrap: bool = False,
) -> tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    """
    Finds the first point on a trajectory a specific distance away from a given point.

    This function searches forward from the provided progress parameter 't'.
    It is used to find the lookahead point for Pure Pursuit.

    Args:
        point: Current [x, y] position of the vehicle.
        radius: Lookahead distance.
        trajectory: Array of [x, y, v] waypoints.
        t: Starting progress on the trajectory (index + fractional part).
        wrap: Whether the trajectory should wrap around (closed loop).

    Returns:
        tuple: (intersection_point, segment_index, segment_fraction)
            Returns (None, None, None) if no intersection is found within the search range.
    """
    start_idx = int(t)
    trajectory = np.ascontiguousarray(trajectory)

    num_points = trajectory.shape[0]
    total_offset = num_points if wrap else num_points - 1 - start_idx

    for offset in range(total_offset):
        idx = (start_idx + offset) % num_points
        p1 = trajectory[idx]
        v_vec = trajectory[(idx + 1) % num_points] + 1e-6 - p1

        t1, t2 = _get_intersections(point[0:2], radius, p1[0:2], v_vec[0:2])
        if not np.isnan(t1):
            lower = t % 1.0 if offset == 0 else 0.0
            if lower <= t1 <= 1.0:
                return p1 + t1 * v_vec, idx, t1
            if lower <= t2 <= 1.0:
                return p1 + t2 * v_vec, idx, t2

    return None, None, None


@njit(cache=True)
def get_actuation(
    pose_theta: float,
    lookahead_point: np.ndarray,
    position: np.ndarray,
    lookahead_distance: float,
    wheelbase: float,
) -> tuple[float, float]:
    """
    Computes steering and velocity based on a lookahead point using Pure Pursuit math.

    Args:
        pose_theta: Current orientation of the vehicle in radians.
        lookahead_point: Target point in global [x, y, v] format.
        position: Current [x, y] coordinates of the vehicle.
        lookahead_distance: Euclidean distance to the lookahead point.
        wheelbase: Physical distance between axles.

    Returns:
        tuple: (speed, steering_angle) in m/s and radians respectively.
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    final_speed = max(speed, 0.0) * (1.0 - np.abs(steering_angle) * 2.0 / np.pi)
    return final_speed, steering_angle
