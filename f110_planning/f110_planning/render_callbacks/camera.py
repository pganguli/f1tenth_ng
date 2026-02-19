"""
Camera tracking render callback for following the ego vehicle.
"""

from typing import Callable

import numpy as np
from f110_gym.envs.rendering import EnvRenderer


def create_camera_tracking(rotate: bool = True) -> Callable[[EnvRenderer], None]:
    """
    Factory for an ego-centric camera tracking callback.

    The camera will center on the ego vehicle (index 0). If rotation is enabled,
    the world will rotate around the car such that the car's heading is
    always directed "Up" (North) on the screen.

    Scaling: Uses the standard 50.0 pixels-per-meter scale.
    """

    def camera_tracking(env_renderer: EnvRenderer) -> None:
        """
        Update camera to follow car (ego car at index 0).
        """
        if not env_renderer.cars or env_renderer.sim_state.poses is None:
            return

        # Get ego car pose [x, y, theta]
        ego_pose = env_renderer.sim_state.poses[0]
        ego_x, ego_y, ego_theta = ego_pose

        # Update camera boundaries with padding
        # The world in this renderer is scaled by 50.0
        scale = 50.0
        padding = 15.0 * scale  # 10 meters visible in each direction
        env_renderer.left = -padding
        env_renderer.right = padding
        env_renderer.top = padding
        env_renderer.bottom = -padding

        # Center camera x,y in world coordinates (scaled by 50.0)
        env_renderer.camera.x = ego_x * scale
        env_renderer.camera.y = ego_y * scale

        # Rotate the view.
        # To make car face Up (+Y), we need to rotate the world by -(theta - pi/2)
        if rotate:
            env_renderer.camera.rotation = -(ego_theta - np.pi / 2)
        else:
            env_renderer.camera.rotation = 0.0

    return camera_tracking
