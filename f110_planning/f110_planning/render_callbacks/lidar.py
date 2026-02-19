"""
LiDAR visualization and analysis utilities for the F1TENTH simulation.
"""

import numpy as np
import pyglet
from f110_gym.envs.rendering import EnvRenderer

from f110_planning.utils import get_heading_error, get_side_distances


def render_lidar(env_renderer: EnvRenderer) -> None:
    """
    Render LIDAR point cloud rays.

    Draws lines from the car center to detected obstacles. Closer hits are
    rendered in red, while farther ones fade to blue.

    Note: The world-to-pixel scale is 50.0.
    """
    # Get latest scans from EnvRenderer
    if env_renderer.scans is None or env_renderer.poses is None:
        return

    # Car center (approx) from vertices
    if not env_renderer.cars:
        return

    x_car = np.mean(env_renderer.cars[0].vertices[::2])
    y_car = np.mean(env_renderer.cars[0].vertices[1::2])
    yaw = env_renderer.poses[0][2]

    # Clear previous lines
    if not hasattr(env_renderer, "lidar_lines"):
        env_renderer.lidar_lines = []
    else:
        for line in env_renderer.lidar_lines:
            line.delete()
        env_renderer.lidar_lines = []

    # Draw every 30th ray for performance
    # Scale for visualization is 50.0
    scan = env_renderer.scans[0]
    angles = np.linspace(-4.7 / 2, 4.7 / 2, len(scan))

    for r, theta in zip(scan[::30], angles[::30]):
        if r <= 0.0:
            continue

        # Simple color coding: closer = red, farther = blue
        c = int(min(r / 10.0, 1.0) * 255)
        env_renderer.lidar_lines.append(
            pyglet.shapes.Line(
                x_car,
                y_car,
                x_car + r * np.cos(theta + yaw) * 50.0,
                y_car + r * np.sin(theta + yaw) * 50.0,
                thickness=1,
                color=(255, 255 - c, 255 - c),
                batch=env_renderer.batch,
            )
        )


def render_side_distances(env_renderer: EnvRenderer) -> None:
    """
    Render the left and right side distances on the screen.
    """
    e = env_renderer
    if e.scans is None:
        return

    left_dist, right_dist = get_side_distances(e.scans[0])

    if not hasattr(e, "side_dist_labels"):
        e.side_dist_labels = {
            "left": pyglet.text.Label(
                "",
                font_size=16,
                x=e.width - 20,
                y=20,
                anchor_x="right",
                color=(255, 255, 255, 255),
                batch=e.ui_batch,
            ),
            "right": pyglet.text.Label(
                "",
                font_size=16,
                x=e.width - 20,
                y=50,
                anchor_x="right",
                color=(255, 255, 255, 255),
                batch=e.ui_batch,
            ),
        }

    e.side_dist_labels["left"].text = f"Left Distance: {left_dist:.2f} m"
    e.side_dist_labels["right"].text = f"Right Distance: {right_dist:.2f} m"

    # Update positions in case window was resized
    e.side_dist_labels["left"].x = e.width - 20
    e.side_dist_labels["left"].y = 20
    e.side_dist_labels["right"].x = e.width - 20
    e.side_dist_labels["right"].y = 50


def create_heading_error_renderer(waypoints: np.ndarray, agent_idx: int = 0):
    """
    Factory to create a heading error rendering callback.

    Args:
        waypoints: Array of waypoints [N, 2] or [N, 3+] with x, y coordinates
        agent_idx: Index of the agent to display heading error for
    """

    def render_heading_error(env_renderer: EnvRenderer) -> None:
        e = env_renderer
        if e.poses is None or e.poses.shape[0] <= agent_idx:
            return

        # Get current car position and orientation
        car_position = np.array([e.poses[agent_idx, 0], e.poses[agent_idx, 1]])
        car_theta = e.poses[agent_idx, 2]

        # Calculate heading error
        heading_error = get_heading_error(waypoints, car_position, car_theta)

        # Create label if it doesn't exist
        if not hasattr(e, "heading_error_label"):
            e.heading_error_label = pyglet.text.Label(
                "",
                font_size=16,
                x=e.width - 20,
                y=80,  # Position above the wall distance labels
                anchor_x="right",
                color=(
                    255,
                    255,
                    0,
                    255,
                ),  # Yellow color to distinguish from wall distances
                batch=e.ui_batch,
            )

        # Update label text and position
        deg = np.degrees(heading_error)
        e.heading_error_label.text = (
            f"Heading Error: {heading_error: .3f} rad ({deg: .1f}°)"
        )
        e.heading_error_label.x = e.width - 20
        e.heading_error_label.y = 80

    return render_heading_error
