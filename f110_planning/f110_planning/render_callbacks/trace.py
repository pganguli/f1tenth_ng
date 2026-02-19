"""
Render callback for visualizing the trajectory trace of an agent.
"""

from collections.abc import Callable

import numpy as np
import pyglet
from f110_gym.envs.rendering import EnvRenderer


def create_trace_renderer(
    agent_idx: int = 0,
    max_points: int = 5000,
    color: tuple[int, int, int] = (255, 255, 0),
) -> Callable[[EnvRenderer], None]:
    """
    Factory to create a trajectory trace rendering callback.

    Args:
        agent_idx: Index of the agent to trace.
        max_points: Maximum number of points to keep in the trace.
        color: RGB tuple for the trace points.
    """
    last_pos = None
    min_dist = 0.1  # Minimum distance between trace points in meters
    shapes_attr = f"trace_shapes_{agent_idx}"

    def render_trace(env_renderer: EnvRenderer) -> None:
        nonlocal last_pos
        e = env_renderer

        if not hasattr(e, shapes_attr):
            setattr(e, shapes_attr, [])

        trace_shapes = getattr(e, shapes_attr)

        # Get current position for the specified agent
        if e.poses is not None and e.poses.shape[0] > agent_idx:
            ego_pos = e.poses[agent_idx, 0:2]

            # Add point if moved enough or first point
            if last_pos is None or np.linalg.norm(ego_pos - last_pos) > min_dist:
                scaled_pos = 50.0 * ego_pos

                point = pyglet.shapes.Circle(
                    x=scaled_pos[0],
                    y=scaled_pos[1],
                    radius=1.5,
                    color=color,
                    batch=e.batch,
                )
                trace_shapes.append(point)
                last_pos = ego_pos.copy()

                # Maintain max points
                if len(trace_shapes) > max_points:
                    old_point = trace_shapes.pop(0)
                    old_point.delete()

    return render_trace
