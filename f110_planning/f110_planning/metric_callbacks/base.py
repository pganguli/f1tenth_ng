"""
Base class for evaluation metric callbacks.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from f110_planning.base import Action


class BaseMetric(ABC):
    """
    Abstract base class for simulation evaluation metrics.

    Metric callbacks follow a lifecycle:
        1. ``on_reset()``  — called once at the start of each episode.
        2. ``on_step()``   — called after every ``env.step()``.
        3. ``report()``    — called at episode end to compute summary statistics.

    Subclasses must implement all three methods plus the ``name`` property.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for the metric (e.g. 'Lap Time')."""

    @abstractmethod
    def on_reset(
        self,
        obs: dict[str, Any],
        waypoints: np.ndarray | None = None,
    ) -> None:
        """
        Initialise / clear internal state at the start of an episode.

        Args:
            obs: Initial observation dictionary from ``env.reset()``.
            waypoints: Reference path ``(N, 2+)`` if available, else ``None``.
        """

    @abstractmethod
    def on_step(
        self,
        obs: dict[str, Any],
        action: Action,
        reward: float,
        ego_idx: int = 0,
    ) -> None:
        """
        Accumulate data from a single simulation step.

        Args:
            obs: Observation dictionary returned by ``env.step()``.
            action: The ``Action`` applied during this step.
            reward: Scalar reward (equal to ``env.timestep``, typically 0.01 s).
            ego_idx: Index of the ego vehicle in multi-agent observations.
        """

    @abstractmethod
    def report(self) -> dict[str, float]:
        """
        Compute and return summary statistics for the episode.

        Returns:
            A dictionary mapping human-readable stat names to float values.
        """
