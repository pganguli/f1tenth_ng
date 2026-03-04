"""
F1TENTH Planning library.
"""

from .base import Action, BasePlanner, CloudScheduler
from .schedulers import FixedIntervalScheduler

# Import submodules AFTER defining Action and BasePlanner to avoid circular imports
# pylint: disable=duplicate-code
from .misc import (
    DummyPlanner,
    FlippyPlanner,
    HybridPlanner,
    ManualPlanner,
    RandomPlanner,
)
from .reactive import (
    BubblePlanner,
    DisparityExtenderPlanner,
    DynamicWaypointPlanner,
    EdgeCloudPlanner,
    GapFollowerPlanner,
    LidarDNNPlanner,
)
from .tracking import (
    LQRPlanner,
    PurePursuitPlanner,
    StanleyPlanner,
)

# pylint: enable=duplicate-code

from . import metrics

__all__ = [
    "metrics",
    # Action, BasePlanner, CloudScheduler are intentionally omitted — autodoc
    # documents them canonically from f110_planning.base; including them here
    # would cause Sphinx to register the NamedTuple fields twice.
    "FixedIntervalScheduler",
    "DummyPlanner",
    "FlippyPlanner",
    "HybridPlanner",
    "ManualPlanner",
    "RandomPlanner",
    "LQRPlanner",
    "PurePursuitPlanner",
    "StanleyPlanner",
    "BubblePlanner",
    "DisparityExtenderPlanner",
    "DynamicWaypointPlanner",
    "EdgeCloudPlanner",
    "GapFollowerPlanner",
    "LidarDNNPlanner",
]
