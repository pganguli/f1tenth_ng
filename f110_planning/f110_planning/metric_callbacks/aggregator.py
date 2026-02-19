"""
Aggregator that orchestrates multiple metric callbacks and prints a summary.
"""

import json
from typing import Any

import numpy as np

from f110_planning.base import Action

from .base import BaseMetric
from .cross_track_error import CrossTrackErrorMetric
from .heading_error import HeadingErrorMetric
from .lap_time import LapTimeMetric
from .smoothness import SmoothnessMetric
from .speed import SpeedMetric
from .wall_proximity import WallProximityMetric


class MetricAggregator:
    """
    Manages a collection of :class:`BaseMetric` instances, delegating
    lifecycle calls and producing a unified summary report.

    Example usage::

        metrics = MetricAggregator.create_default(waypoints=wpts)
        metrics.on_reset(obs, waypoints=wpts)

        while not done:
            action = planner.plan(obs)
            obs, reward, term, trunc, _ = env.step(...)
            metrics.on_step(obs, action, reward)

        metrics.report()   # prints table and returns merged dict
    """

    def __init__(self, callbacks: list[BaseMetric]) -> None:
        self._callbacks = callbacks

    @property
    def metric_count(self) -> int:
        """Return the number of registered metric callbacks."""
        return len(self._callbacks)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def create_default(
        cls,
        waypoints: np.ndarray | None = None,
    ) -> "MetricAggregator":
        """
        Build an aggregator with all available metrics.

        Waypoint-dependent metrics (cross-track error, heading error) are
        included only when *waypoints* is provided and non-empty.

        Args:
            waypoints: Reference path ``(N, 2+)``, or ``None``.

        Returns:
            A fully-configured :class:`MetricAggregator`.
        """
        callbacks: list[BaseMetric] = [
            LapTimeMetric(),
            WallProximityMetric(),
            SmoothnessMetric(),
            SpeedMetric(),
        ]

        has_waypoints = waypoints is not None and waypoints.size > 0
        if has_waypoints:
            callbacks.insert(1, CrossTrackErrorMetric())
            callbacks.insert(2, HeadingErrorMetric())

        return cls(callbacks)

    # ------------------------------------------------------------------
    # Lifecycle delegation
    # ------------------------------------------------------------------
    def on_reset(
        self,
        obs: dict[str, Any],
        waypoints: np.ndarray | None = None,
    ) -> None:
        """Forward ``on_reset`` to every registered metric."""
        for cb in self._callbacks:
            cb.on_reset(obs, waypoints=waypoints)

    def on_step(
        self,
        obs: dict[str, Any],
        action: Action,
        reward: float,
        ego_idx: int = 0,
    ) -> None:
        """Forward ``on_step`` to every registered metric."""
        for cb in self._callbacks:
            cb.on_step(obs, action, reward, ego_idx=ego_idx)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def report(self) -> dict[str, float]:
        """
        Collect statistics from all metrics, print a formatted summary
        table, and return the merged dictionary.
        """
        merged: dict[str, float] = {}
        sections: list[tuple[str, dict[str, float]]] = []

        for cb in self._callbacks:
            stats = cb.report()
            merged.update(stats)
            sections.append((cb.name, stats))

        self._print_summary(sections)
        return merged

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(sections: list[tuple[str, dict[str, float]]]) -> None:
        """Print metrics as indented, parsable JSON to stdout."""
        print(json.dumps(dict(sections), indent=2))
