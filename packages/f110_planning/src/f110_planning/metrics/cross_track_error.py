"""
Cross-track (lateral deviation) error metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from f110_planning.base import Action
from ..utils.pure_pursuit_utils import nearest_point

from .base import BaseMetric


def crosstrack_error(pos: np.ndarray, waypoints: np.ndarray) -> float:
    """Return the perpendicular distance from *pos* to the nearest segment of
    *waypoints*.

    Args:
        pos: ``(2,)`` array ``[x, y]`` of the vehicle position.
        waypoints: ``(N, 2+)`` array of reference waypoints.  Only the first
            two columns (x, y) are used.

    Returns:
        Non-negative lateral distance in metres.
    """
    wpts = np.ascontiguousarray(waypoints[:, :2], dtype=np.float64)
    p = np.asarray(pos, dtype=np.float64)
    _, dist, _, _ = nearest_point(p, wpts)
    return float(dist)


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
        dist = crosstrack_error(pos, self._waypoints)
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
