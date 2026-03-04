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


def create_cloud_call_renderer(
    planner,
    agent_idx: int = 0,
    max_points: int = 5000,
    color: tuple[int, int, int] = (255, 140, 0),
    radius: float = 3.0,
) -> Callable[[EnvRenderer], None]:
    """Render callback that highlights steps where a cloud call was issued.

    The renderer examines ``planner.last_cloud_call`` on each frame.  When
    ``True`` it appends an orange circle at the agent's current position.  This
    paints a deep-orange dot (larger than the normal trace) for each step the
    scheduler decided to contact the cloud.  The tracer history is capped at
    ``max_points`` to avoid unbounded memory growth.
    """
    shapes_attr = f"cloud_shapes_{agent_idx}"

    def render_cloud(env_renderer: EnvRenderer) -> None:
        # ensure list exists
        if not hasattr(env_renderer, shapes_attr):
            setattr(env_renderer, shapes_attr, [])
        shapes = getattr(env_renderer, shapes_attr)

        # only draw if the planner recorded a cloud call on the last plan
        if planner.last_cloud_call:
            if env_renderer.poses is not None and env_renderer.poses.shape[0] > agent_idx:
                ego_pos = env_renderer.poses[agent_idx, 0:2]
                scaled_pos = 50.0 * ego_pos
                point = pyglet.shapes.Circle(
                    x=scaled_pos[0],
                    y=scaled_pos[1],
                    radius=radius,
                    color=color,
                    batch=env_renderer.batch,
                )
                shapes.append(point)
                if len(shapes) > max_points:
                    old = shapes.pop(0)
                    old.delete()

    return render_cloud


# Per-DNN colors for create_selective_cloud_call_renderer.
# Index 0 = left_wall, 1 = track_width, 2 = heading — matches SelectiveEdgeCloudPlanner.
_DNN_COLORS: list[tuple[int, int, int]] = [
    (0, 210, 255),    # cyan      — left-wall distance
    (100, 230, 40),   # lime      — track-width
    (230, 60, 230),   # magenta   — heading error
]
_MULTI_CALL_COLOR: tuple[int, int, int] = (255, 255, 255)  # white — two or more DNNs called


def create_selective_cloud_call_renderer(
    planner,
    agent_idx: int = 0,
    max_points: int = 5000,
    radius: float = 3.0,
) -> Callable[[EnvRenderer], None]:
    """Render callback for :class:`~f110_planning.reactive.SelectiveEdgeCloudPlanner`.

    Draws a colour-coded dot at the agent's position whenever one or more cloud
    DNNs were called on the last planning step.  The dot colour encodes *which*
    DNN(s) were called:

    * **cyan** ``(0, 210, 255)``   — left-wall distance DNN
    * **lime** ``(100, 230, 40)``  — track-width DNN
    * **magenta** ``(230, 60, 230)`` — heading-error DNN
    * **white** ``(255, 255, 255)`` — two or more DNNs called simultaneously

    When no DNN was called the regular yellow trace dot (from
    :func:`create_trace_renderer`) is all that appears — no dot is added by
    this callback.

    Parameters
    ----------
    planner : SelectiveEdgeCloudPlanner
        The planner whose ``last_call_mask`` attribute is read each frame.
    agent_idx : int
        Index of the agent to track.
    max_points : int
        Maximum retained dots before oldest are evicted.
    radius : float
        Circle radius in screen pixels.
    """
    shapes_attr = f"sel_cloud_shapes_{agent_idx}"

    def render_selective_cloud(env_renderer: EnvRenderer) -> None:
        if not hasattr(env_renderer, shapes_attr):
            setattr(env_renderer, shapes_attr, [])
        shapes = getattr(env_renderer, shapes_attr)

        call_mask: list[bool] = getattr(planner, "last_call_mask", [])
        n_calls = sum(call_mask)
        if n_calls == 0:
            return  # nothing to draw; regular trace handles baseline

        if env_renderer.poses is None or env_renderer.poses.shape[0] <= agent_idx:
            return

        ego_pos = env_renderer.poses[agent_idx, 0:2]
        scaled_pos = 50.0 * ego_pos

        if n_calls > 1:
            dot_color = _MULTI_CALL_COLOR
        else:
            called_idx = call_mask.index(True)
            dot_color = _DNN_COLORS[called_idx] if called_idx < len(_DNN_COLORS) else _MULTI_CALL_COLOR

        point = pyglet.shapes.Circle(
            x=scaled_pos[0],
            y=scaled_pos[1],
            radius=radius,
            color=dot_color,
            batch=env_renderer.batch,
        )
        shapes.append(point)
        if len(shapes) > max_points:
            old = shapes.pop(0)
            old.delete()

    return render_selective_cloud
