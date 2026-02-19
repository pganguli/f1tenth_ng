"""
Manual planner module for keyboard control.
"""

from typing import Any

import numpy as np
from pyglet import display as pyg_display
from pyglet import window as pyg_window

from ..base import Action, BasePlanner


class ManualPlanner(BasePlanner):  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """
    Keyboard-based control interface with simplified vehicle dynamics.

    This planner allows a human operator to control the vehicle using 'WASD' or
    Arrow keys. It simulates realistic steering and acceleration rates
    by integrating inputs over the simulation timestep.
    """

    @staticmethod
    def _kbd_init() -> Any:
        """
        Initializes the Pyglet keyboard handler by attaching to the active window.
        """
        display = pyg_display.get_display()
        keys = pyg_window.key.KeyStateHandler()
        windows = display.get_windows()
        if not windows:
            msg = "No active visualization window found for manual control."
            raise RuntimeError(msg)
        windows[0].push_handlers(keys)
        return keys

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        s_min: float = -0.4189,
        s_max: float = 0.4189,
        v_min: float = -5.0,
        v_max: float = 20.0,
        accel: float = 8.0,
        decel: float = 15.0,
        steer_rate: float = 1.5,
        dt: float = 0.01,
    ):
        """
        Sets the physical response limits for manual control.

        Args:
            s_min: Hard limit for left steering [rad].
            s_max: Hard limit for right steering [rad].
            v_min: Minimum velocity (including reverse) [m/s].
            v_max: Maximum velocity [m/s].
            accel: Rate of velocity increase when 'W' is held [m/s^2].
            decel: Braking rate when 'S' is held [m/s^2].
            steer_rate: Speed of the steering rack [rad/s].
            dt: Effective integration period (should match env timestep).
        """
        self.keys = ManualPlanner._kbd_init()
        self.s_min = s_min
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max
        self.accel = accel
        self.decel = decel
        self.steer_rate = steer_rate
        self.dt = dt

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Polls the keyboard state and integrates the vehicle control command.
        """
        current_speed = obs["linear_vels_x"][ego_idx]
        current_steer = obs["steering_angles"][ego_idx]

        # Speed control (Gas/Brake)
        if self.keys[pyg_window.key.W]:
            # Gas: increase target speed
            speed = min(self.v_max, current_speed + self.accel * self.dt)
        elif self.keys[pyg_window.key.S]:
            # Brake/Reverse: decrease target speed
            speed = max(self.v_min, current_speed - self.decel * self.dt)
        else:
            # Coasting: slowly return to zero speed if no input
            if abs(current_speed) < 0.2:
                speed = 0.0
            else:
                # Slight friction/coasting deceleration
                speed = (
                    current_speed
                    - np.sign(current_speed) * (self.accel / 4.0) * self.dt
                )

        # Steering control (Wheel)
        if self.keys[pyg_window.key.A]:
            # Turn left
            steer = min(self.s_max, current_steer + self.steer_rate * self.dt)
        elif self.keys[pyg_window.key.D]:
            # Turn right
            steer = max(self.s_min, current_steer - self.steer_rate * self.dt)
        else:
            # Return-to-center logic
            if abs(current_steer) < self.steer_rate * self.dt:
                steer = 0.0
            else:
                steer = (
                    current_steer - np.sign(current_steer) * self.steer_rate * self.dt
                )

        return Action(steer=steer, speed=speed)
