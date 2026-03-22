"""
Signal extraction helpers for safety-driven edge-cloud scheduling.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchedulingSignals:
    """
    Runtime signals consumed by scheduling policies.

    Attributes:
        deviation: Scalar risk/deviation signal used by safety triggers.
        uncertainty: Optional edge-model uncertainty estimate.
        speed: Optional ego longitudinal speed.
    """

    deviation: float
    uncertainty: float | None = None
    speed: float | None = None


def build_signals(
    obs: dict[str, Any],
    context: dict[str, Any] | None = None,
    ego_idx: int = 0,
) -> SchedulingSignals:
    """
    Build policy features from planner observation and runtime context.

    Priority for `deviation`:
    1) context["deviation"]
    2) context["edge_uncertainty"]
    3) obs["crosstrack_dist"][ego_idx]
    4) 0.0
    """

    ctx = context or {}
    deviation_value = ctx.get("deviation")
    uncertainty_value = ctx.get("edge_uncertainty")

    if deviation_value is None:
        deviation_value = uncertainty_value

    if deviation_value is None:
        crosstrack = obs.get("crosstrack_dist")
        if crosstrack is not None:
            try:
                deviation_value = float(crosstrack[ego_idx])
            except (TypeError, ValueError, IndexError):
                deviation_value = None

    if deviation_value is None:
        deviation_value = 0.0

    speed_value = ctx.get("ego_speed")
    if speed_value is None:
        linear_vels = obs.get("linear_vels_x")
        if linear_vels is not None:
            try:
                speed_value = float(linear_vels[ego_idx])
            except (TypeError, ValueError, IndexError):
                speed_value = None

    return SchedulingSignals(
        deviation=float(deviation_value),
        uncertainty=(
            float(uncertainty_value)
            if isinstance(uncertainty_value, (int, float))
            else None
        ),
        speed=float(speed_value) if isinstance(speed_value, (int, float)) else None,
    )
