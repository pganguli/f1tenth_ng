"""
Per-DNN cloud call count metric for selective edge-cloud planners.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from f110_planning.base import Action

from .base import BaseMetric

#: DNN slot names — must match ``SelectiveCloudSchedulerEnv.DNN_NAMES``
#: and ``SelectiveEdgeCloudPlanner`` slot ordering.
DNN_NAMES: list[str] = ["left_wall", "track_width", "heading"]


@runtime_checkable
class _HasCallMask(Protocol):
    """Structural type for planners that expose per-DNN call decisions."""

    @property
    def last_call_mask(self) -> list[bool]:
        ...


class CloudCallCountMetric(BaseMetric):
    """Counts per-slot cloud DNN calls made by a selective planner.

    Reads ``planner.last_call_mask`` after each planning step — updated by
    both :class:`~f110_planning.reactive.SelectiveEdgeCloudPlanner` and the
    ``SelectivePolicyPlanner`` wrapper in the simulation scripts — and
    accumulates totals per DNN slot.

    Results appear in the JSON summary printed by
    :class:`~f110_planning.metrics.MetricAggregator` alongside all other
    simulation metrics::

        "Cloud DNN Calls": {
            "total_steps": 720.0,
            "left_wall_calls": 142.0,
            "left_wall_call_rate": 0.1972,
            "track_width_calls": 287.0,
            "track_width_call_rate": 0.3986,
            "heading_calls": 291.0,
            "heading_call_rate": 0.4042
        }

    Parameters
    ----------
    planner :
        Any object exposing a ``last_call_mask`` property that returns a
        ``list[bool]`` of length ``m`` (one entry per DNN slot).
    dnn_names : list[str], optional
        Override the default slot labels.  Defaults to
        :data:`DNN_NAMES` = ``["left_wall", "track_width", "heading"]``.
    """

    def __init__(
        self,
        planner: _HasCallMask,
        dnn_names: list[str] | None = None,
    ) -> None:
        if not isinstance(planner, _HasCallMask):
            raise TypeError(
                f"{type(planner).__name__} does not expose 'last_call_mask'; "
                "cannot attach CloudCallCountMetric."
            )
        self._planner = planner
        self._dnn_names: list[str] = dnn_names if dnn_names is not None else list(DNN_NAMES)
        self._counts: list[int] = [0] * len(self._dnn_names)
        self._steps: int = 0

    @property
    def name(self) -> str:
        return "Cloud DNN Calls"

    def on_reset(
        self,
        obs: dict[str, Any],  # noqa: ARG002
        waypoints: np.ndarray | None = None,  # noqa: ARG002
    ) -> None:
        self._counts = [0] * len(self._dnn_names)
        self._steps = 0

    def on_step(
        self,
        obs: dict[str, Any],  # noqa: ARG002
        action: Action,  # noqa: ARG002
        reward: float,  # noqa: ARG002
        ego_idx: int = 0,  # noqa: ARG002
    ) -> None:
        mask = self._planner.last_call_mask
        for i, called in enumerate(mask):
            if called:
                self._counts[i] += 1
        self._steps += 1

    def report(self) -> dict[str, float]:
        stats: dict[str, float] = {"total_steps": float(self._steps)}
        for dnn_name, count in zip(self._dnn_names, self._counts):
            stats[f"{dnn_name}_calls"] = float(count)
            rate = (count / self._steps) if self._steps > 0 else 0.0
            stats[f"{dnn_name}_call_rate"] = round(rate, 4)
        return stats
