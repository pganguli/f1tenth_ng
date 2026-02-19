"""
Render callback for visualizing the dynamic waypoint
"""

import pyglet


def create_dynamic_waypoint_renderer(planner, agent_idx=0):
    """
    Factory to visualize the planner's internal target waypoint.

    This renderer extracts 'last_target_point' from the planner instance,
    ensuring that the visual magenta point matches the exactly computed
    adaptive lookahead point used in the steering logic.
    """

    # Graphics objects (created once, updated each frame)
    waypoint_circle = None
    connection_line = None

    def render_callback(env_renderer):
        nonlocal waypoint_circle, connection_line

        # Ensure we have a point to draw
        if planner.last_target_point is None:
            return

        # Get ego position to draw the connection line
        world_position = env_renderer.poses[agent_idx, :2]
        car_center_px = 50.0 * world_position

        # Convert the world-frame point (saved by the planner) to pixels
        # Scale of 50.0 used throughout the rendering system
        target_px = 50.0 * planner.last_target_point[:2]

        # Create or update circle
        if waypoint_circle is None:
            waypoint_circle = pyglet.shapes.Circle(
                x=target_px[0],
                y=target_px[1],
                radius=8,
                color=(255, 100, 255),  # Magenta
                batch=env_renderer.batch,
            )
        else:
            waypoint_circle.x = target_px[0]
            waypoint_circle.y = target_px[1]

        # Create or update line from car to waypoint
        if connection_line is None:
            connection_line = pyglet.shapes.Line(
                x=car_center_px[0],
                y=car_center_px[1],
                x2=target_px[0],
                y2=target_px[1],
                thickness=2,
                color=(255, 100, 255, 150),  # Semi-transparent magenta
                batch=env_renderer.batch,
            )
        else:
            connection_line.x = car_center_px[0]
            connection_line.y = car_center_px[1]
            connection_line.x2 = target_px[0]
            connection_line.y2 = target_px[1]

    return render_callback
