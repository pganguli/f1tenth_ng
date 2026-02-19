"""
Render callback for visualizing waypoints in the simulation.
"""

import numpy as np
import pyglet
from f110_gym.envs.rendering import EnvRenderer


def create_waypoint_renderer(
    waypoints: np.ndarray,
    color: tuple[int, int, int] | tuple[int, int, int, int] = (183, 193, 222),
    name: str = "waypoint_shapes",
):
    """
    Factory to create a waypoint rendering callback.

    Args:
        waypoints: np.ndarray of shape (N, 2+) where waypoints[:, 0:2] are x, y.
        color: tuple of (r, g, b) or (r, g, b, a)
        name: unique name to store the shapes on the renderer object
    """
    # Pre-scale waypoints for visualization
    scaled_wpts = 50.0 * waypoints[:, 0:2]

    def render_waypoints(env_renderer: EnvRenderer) -> None:
        e = env_renderer

        if not hasattr(e, name):
            setattr(e, name, [])
            shapes = getattr(e, name)
            for i in range(scaled_wpts.shape[0]):
                circle = pyglet.shapes.Circle(
                    x=scaled_wpts[i, 0],
                    y=scaled_wpts[i, 1],
                    radius=2.0,
                    color=color,
                    batch=e.batch,
                )
                shapes.append(circle)

    return render_waypoints
