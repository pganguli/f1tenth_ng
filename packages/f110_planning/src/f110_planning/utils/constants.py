"""
Global constants for the F1TENTH vehicle and its on-board LiDAR sensor.

This file is the single auditable source of truth for the physical and
sensor constants used throughout the planning stack.

All values correspond to the standard F1TENTH 1:10 scale platform with a
Hokuyo UST-10LX LiDAR (as configured in the f110_gym simulator).
"""

# ---------------------------------------------------------------------------
# F1TENTH vehicle geometry / actuation limits
# ---------------------------------------------------------------------------

F110_WIDTH: float = 0.31
"""Vehicle width [m]."""

F110_LENGTH: float = 0.58
"""Overall vehicle length [m]."""

F110_WHEELBASE: float = 0.33
"""Distance between front and rear axle centres [m]."""

F110_MAX_STEER: float = 0.4189
"""Maximum steering angle magnitude [rad]  (≈ 24 °)."""

# ---------------------------------------------------------------------------
# LiDAR sensor constants  (standard f110_gym / Hokuyo UST-10LX config)
# ---------------------------------------------------------------------------

LIDAR_NUM_BEAMS: int = 1080
"""Number of beams in one LiDAR scan."""

LIDAR_FOV: float = 4.7
"""Total LiDAR field of view [rad]  (≈ 270 °)."""

LIDAR_MIN_ANGLE: float = -LIDAR_FOV / 2
"""Angle of the first (leftmost) LiDAR beam [rad]."""

LIDAR_MAX_ANGLE: float = LIDAR_FOV / 2
"""Angle of the last (rightmost) LiDAR beam [rad]."""
