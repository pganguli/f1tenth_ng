"""
Global physical constants and default configuration for the F1TENTH vehicle.

This file is the single auditable source of truth for:
- Physical constants used in dynamic-model calculations (e.g. gravity).
- Default vehicle parameter values for the F1TENTH 1:10 scale race car.
- Vehicle geometry used across the gym (dynamics, collision, rendering).

References for the F1TENTH car parameters:
  https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
  https://f1tenth.org/build.html
"""

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

GRAVITY: float = 9.81
"""Gravitational acceleration [m/s²]."""

# ---------------------------------------------------------------------------
# F1TENTH vehicle geometry
# ---------------------------------------------------------------------------

CAR_LENGTH: float = 0.58
"""Overall vehicle length [m]."""

CAR_WIDTH: float = 0.31
"""Overall vehicle width [m]."""

# ---------------------------------------------------------------------------
# Default F1TENTH vehicle dynamics parameters
# ---------------------------------------------------------------------------

DEFAULT_VEHICLE_PARAMS: dict = {
    # Tyre / road friction
    "mu": 1.0489,       # Surface friction coefficient [-]
    # Cornering stiffnesses (linearised tyre model)
    "C_Sf": 4.718,      # Front cornering stiffness [N/rad]
    "C_Sr": 5.4562,     # Rear  cornering stiffness [N/rad]
    # Geometry
    "lf": 0.15875,      # Distance from CG to front axle [m]
    "lr": 0.17145,      # Distance from CG to rear  axle [m]
    "h": 0.074,         # Height of centre of gravity [m]
    # Inertia
    "m": 3.74,          # Total vehicle mass [kg]
    "I": 0.04712,       # Yaw moment of inertia [kg·m²]
    # Steering constraints
    "s_min": -0.4189,   # Minimum steering angle [rad]  (≈ -24 °)
    "s_max":  0.4189,   # Maximum steering angle [rad]  (≈ +24 °)
    "sv_min": -3.2,     # Minimum steering-angle rate [rad/s]
    "sv_max":  3.2,     # Maximum steering-angle rate [rad/s]
    # Longitudinal dynamics
    "v_switch": 7.319,  # Velocity at which traction limit switches [m/s]
    "a_max": 9.51,      # Maximum longitudinal acceleration [m/s²]
    "v_min": -5.0,      # Minimum longitudinal velocity (reverse) [m/s]
    "v_max": 20.0,      # Maximum longitudinal velocity [m/s]
    # Physical dimensions (also used by collision / rendering)
    "width": CAR_WIDTH,
    "length": CAR_LENGTH,
}
"""Default dynamics parameter set for the F1TENTH 1:10 scale car."""
