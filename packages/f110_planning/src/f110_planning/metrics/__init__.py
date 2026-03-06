"""
Evaluation metrics for the F1TENTH simulation.

This package exposes both the stateful :class:`BaseMetric` callback classes
(which accumulate per-step statistics) and the pure, stateless computation
functions they delegate to.
"""

from .aggregator import MetricAggregator
from .base import BaseMetric
from .cloud_call_count import CloudCallCountMetric
from .cross_track_error import CrossTrackErrorMetric, crosstrack_error
from .heading_error import HeadingErrorMetric, heading_error_rad
from .lap_time import LapTimeMetric, has_collision
from .smoothness import SmoothnessMetric, steering_rate
from .speed import SpeedMetric, current_speed
from .wall_proximity import WallProximityMetric, min_wall_dist

__all__ = [
    # Callback classes
    "BaseMetric",
    "MetricAggregator",
    "LapTimeMetric",
    "CrossTrackErrorMetric",
    "HeadingErrorMetric",
    "WallProximityMetric",
    "SmoothnessMetric",
    "SpeedMetric",
    "CloudCallCountMetric",
    # Pure functions
    "crosstrack_error",
    "heading_error_rad",
    "has_collision",
    "steering_rate",
    "current_speed",
    "min_wall_dist",
]
