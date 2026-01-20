import numpy as np
from pyglet.gl import GL_POINTS
from simple_pid import PID

from .utils import (
    first_point_on_trajectory_intersecting_circle,
    get_actuation,
    index_to_angle,
    nearest_point_on_trajectory,
    polar_to_cart,
)


class BasePlanner:
    def plan(self, obs):
        raise NotImplementedError


class PurePursuitPlanner(BasePlanner):
    """
    Example Planner
    """

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.0

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(
            conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip
        )

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        # points = self.waypoints

        points = np.vstack(
            (
                self.waypoints[:, self.conf.wpt_xind],
                self.waypoints[:, self.conf.wpt_yind],
            )
        ).T

        scaled_points = 50.0 * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ("v3f/stream", [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                    ("c3B/stream", [183, 193, 222]),
                )
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [
                    scaled_points[i, 0],
                    scaled_points[i, 1],
                    0.0,
                ]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack(
            (
                self.waypoints[:, self.conf.wpt_xind],
                self.waypoints[:, self.conf.wpt_yind],
            )
        ).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i + t, wrap=True
            )
            if i2 is None:
                return None
            current_waypoint = np.empty((3,))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(
            self.waypoints, lookahead_distance, position, pose_theta
        )

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(
            pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase
        )
        speed = vgain * speed

        return speed, steering_angle


class FlippyPlanner(BasePlanner):
    """
    Planner designed to exploit integration methods and dynamics.
    For testing only. To observe this error, use single track dynamics for all velocities >0.1
    """

    def __init__(self, speed=1, flip_every=1, steer=2):
        self.speed = speed
        self.flip_every = flip_every
        self.counter = 0
        self.steer = steer

    def render_waypoints(self, *args, **kwargs):
        pass

    def plan(self, *args, **kwargs):
        if self.counter % self.flip_every == 0:
            self.counter = 0
            self.steer *= -1
        return self.speed, self.steer


class WallFollowingPlanner(BasePlanner):
    DISTANCE_TARGET = 1.3
    WALL_OFFSET = 10
    MAX_SPEED = 2.0
    MIN_SPEED = 1.0

    def __init__(self):
        self.pid = PID(1.0, 0.0, 0.3, setpoint=self.DISTANCE_TARGET)

    # TODO: Add render_waypoints(self, e)

    def plan(self, obs):
        scan = obs["scans"][0]

        octant = len(scan) // 6
        base = octant - self.WALL_OFFSET

        p0 = polar_to_cart(scan[base], index_to_angle(base))
        p1 = polar_to_cart(
            scan[base + self.WALL_OFFSET], index_to_angle(base + self.WALL_OFFSET)
        )

        wall_dir = p1 - p0
        wall_dist = abs(np.cross(wall_dir, -p0)) / np.linalg.norm(wall_dir)

        steer = float(self.pid(wall_dist))
        speed_interp = min(abs(steer) / np.pi, 1.0)
        speed = self.MAX_SPEED * (1 - speed_interp) + self.MIN_SPEED * speed_interp

        return speed, steer


class LongestPathPlanner(BasePlanner):
    MAX_SPEED = 4.0
    MIN_SPEED = 2.0

    # TODO: Add render_waypoints(self, e)

    def plan(self, obs):
        scan = obs["scans"][0]

        front = scan[len(scan) // 6 : 5 * len(scan) // 6]
        best_i = np.argmax(front) + len(scan) // 6

        angle = index_to_angle(best_i) - np.pi / 2
        speed_interp = min(abs(angle) / (np.pi / 2), 1.0)
        speed = self.MAX_SPEED * (1 - speed_interp) + self.MIN_SPEED * speed_interp

        return speed, angle


class GapFollowerPlanner(BasePlanner):
    BUBBLE_RADIUS = 160
    MAX_SPEED = 6.0
    MIN_SPEED = 3.0

    # TODO: Add render_waypoints(self, e)

    def plan(self, obs):
        scan = np.array(obs["scans"][0])

        closest = np.argmin(scan)
        scan[max(0, closest - self.BUBBLE_RADIUS) : closest + self.BUBBLE_RADIUS] = 0

        gaps = np.ma.masked_where(scan == 0, scan)
        slices = np.ma.notmasked_contiguous(gaps)
        if not slices:
            return self.MIN_SPEED, 0.0

        largest = max(slices, key=lambda s: s.stop - s.start)
        best = largest.start + np.argmax(scan[largest])

        angle = index_to_angle(best) - np.pi / 2
        speed = self.MAX_SPEED if abs(angle) < 0.2 else self.MIN_SPEED

        return speed, angle
