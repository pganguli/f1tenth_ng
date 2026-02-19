"""
Pure Pursuit waypoint tracker
"""

import warnings
from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner
from ..utils import get_actuation, get_vehicle_state, intersect_point, nearest_point


class PurePursuitPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    Geometrically-inspired waypoint tracking controller.

    Pure Pursuit calculates the curvature required to move the vehicle from its
    current position to a point on the reference path that is one 'lookahead'
    distance away.

    Reference: Coulter, R. Craig. "Implementation of the Pure Pursuit Path Tracking Algorithm."
    Carnegie Mellon University, 1992.
    """

    def __init__(
        self,
        wheelbase: float = 0.33,
        lookahead_distance: float = 0.8,
        max_speed: float = 5.0,
        waypoints: Optional[np.ndarray] = None,
    ):
        """
        Initializes the Pure Pursuit planner with vehicle and path parameters.

        Args:
            wheelbase: Distance between front and rear axles in meters.
            lookahead_distance: The constant radius of the search circle.
            max_speed: The target longitudinal velocity for the tracking.
            waypoints: Loaded path coordinates, ideally [N, 2] or [N, 3+].
        """
        self.max_reacquire = 20.0
        self.wheelbase = wheelbase
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.waypoints = waypoints if waypoints is not None else np.array([])

    def _get_current_waypoint(
        self, lookahead_distance: float, position: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Calculates the intersection of the lookahead circle and the path.

        Args:
            lookahead_distance: Radius of the search circle.
            position: Current [x, y] coordinates of the vehicle.

        Returns:
            A [3,] array [x, y, v] for the target waypoint, or None if unreachable.
        """
        if self.waypoints is None or len(self.waypoints) == 0:
            raise ValueError("Waypoints must be provided to the planner.")

        _, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            # Search forward from the current track progress
            _, i2, _ = intersect_point(
                position, lookahead_distance, self.waypoints[:, 0:2], i + t, wrap=True
            )
            if i2 is None:
                return None
            return np.array(
                [self.waypoints[i2, 0], self.waypoints[i2, 1], self.max_speed]
            )

        # If too far to intersect, fallback to the nearest point if within reacquire range
        if nearest_dist < self.max_reacquire:
            return np.array(
                [self.waypoints[i, 0], self.waypoints[i, 1], self.max_speed]
            )

        return None

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Determines the steering and speed required to follow the path.
        """
        vehicle_state = get_vehicle_state(obs, ego_idx)
        position = vehicle_state[:2]
        lookahead_point = self._get_current_waypoint(self.lookahead_distance, position)

        if lookahead_point is None:
            warnings.warn("Lookahead point not found; stop signal sent.")
            return Action(steer=0.0, speed=0.0)

        speed, steering_angle = get_actuation(
            vehicle_state[2],
            lookahead_point,
            position,
            self.lookahead_distance,
            self.wheelbase,
        )

        return Action(steer=steering_angle, speed=speed)
