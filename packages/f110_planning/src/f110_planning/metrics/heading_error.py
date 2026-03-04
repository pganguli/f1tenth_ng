"""
Heading (orientation) error metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from f110_planning.base import Action
from ..utils.lidar_utils import get_heading_error

from .base import BaseMetric


def heading_error_rad(
    pos: np.ndarray,
    theta: float,
    waypoints: np.ndarray,
) -> float:
    """Return the signed heading error (vehicle vs. local path tangent) in radians.

    Args:
        pos: ``(2,)`` array ``[x, y]`` of the vehicle position.
        theta: Vehicle heading in radians (from simulator ``poses_theta``).
        waypoints: ``(N, 2+)`` reference waypoints.

    Returns:
        Signed heading error in radians.  Use ``abs()`` for magnitude.
    """
    wpts = np.asarray(waypoints, dtype=np.float64)
    p = np.asarray(pos, dtype=np.float64)
    return float(get_heading_error(wpts, p, float(theta)))


class HeadingErrorMetric(BaseMetric):
    """
    Measures the angular deviation between the car's heading and the local
    path tangent.

    Uses :func:`get_heading_error` from ``f110_planning.utils`` which returns
    a signed heading error in radians.  Summary statistics are reported in
    **degrees** for readability.

    Requires ``waypoints`` passed via :meth:`on_reset`.
    """

    def __init__(self) -> None:
        self._waypoints: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._errors: list[float] = []

    @property
    def name(self) -> str:
        return "Heading Error"

    def on_reset(
        self,
        obs: dict[str, Any],
        waypoints: np.ndarray | None = None,
    ) -> None:
        if waypoints is None:
            raise ValueError(
                "HeadingErrorMetric requires waypoints (pass them via on_reset)."
            )
        self._waypoints = waypoints.astype(np.float64)
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
        theta = float(obs["poses_theta"][ego_idx])
        heading_err = heading_error_rad(pos, theta, self._waypoints)
        self._errors.append(heading_err)

    def report(self) -> dict[str, float]:
        errors = np.array(self._errors)
        if errors.size == 0:
            return {
                "heading_error_mean_deg": 0.0,
                "heading_error_rmse_deg": 0.0,
                "heading_error_std_deg": 0.0,
                "heading_error_max_deg": 0.0,
            }
        abs_errors = np.abs(errors)
        return {
            "heading_error_mean_deg": round(float(np.degrees(np.mean(abs_errors))), 4),
            "heading_error_rmse_deg": round(
                float(np.degrees(np.sqrt(np.mean(errors**2)))), 4
            ),
            "heading_error_std_deg": round(float(np.degrees(np.std(errors))), 4),
            "heading_error_max_deg": round(float(np.degrees(np.max(abs_errors))), 4),
        }
