"""
Cross-track (lateral deviation) error metric.
"""

from typing import Any

import numpy as np

from f110_planning.base import Action
from f110_planning.utils import nearest_point

from .base import BaseMetric


class CrossTrackErrorMetric(BaseMetric):
    """
    Measures how far the vehicle deviates from a reference path.

    Uses :func:`nearest_point` to compute the perpendicular distance from the
    car position to the closest segment of the waypoint trajectory at every
    simulation step.

    Requires ``waypoints`` to be passed via :meth:`on_reset`.
    """

    def __init__(self) -> None:
        self._waypoints: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._errors: list[float] = []

    @property
    def name(self) -> str:
        return "Cross-Track Error"

    def on_reset(
        self,
        obs: dict[str, Any],
        waypoints: np.ndarray | None = None,
    ) -> None:
        if waypoints is None:
            raise ValueError(
                "CrossTrackErrorMetric requires waypoints (pass them via on_reset)."
            )
        self._waypoints = waypoints[:, :2].astype(np.float64)
        self._errors = []

    def on_step(
        self,
        obs: dict[str, Any],
        action: Action,
        reward: float,
        ego_idx: int = 0,
    ) -> None:
        pos = np.array(
            [obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]], dtype=np.float64
        )
        _, dist, _, _ = nearest_point(pos, self._waypoints)
        self._errors.append(dist)

    def report(self) -> dict[str, float]:
        errors = np.array(self._errors)
        if errors.size == 0:
            return {
                "crosstrack_rmse_m": 0.0,
                "crosstrack_mean_m": 0.0,
                "crosstrack_max_m": 0.0,
                "crosstrack_std_m": 0.0,
            }
        return {
            "crosstrack_rmse_m": round(float(np.sqrt(np.mean(errors**2))), 4),
            "crosstrack_mean_m": round(float(np.mean(errors)), 4),
            "crosstrack_max_m": round(float(np.max(errors)), 4),
            "crosstrack_std_m": round(float(np.std(errors)), 4),
        }
