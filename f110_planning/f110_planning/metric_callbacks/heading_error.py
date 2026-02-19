"""
Heading (orientation) error metric.
"""

from typing import Any

import numpy as np

from f110_planning.base import Action
from f110_planning.utils import get_heading_error

from .base import BaseMetric


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
        heading_err = get_heading_error(self._waypoints, pos, theta)
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
