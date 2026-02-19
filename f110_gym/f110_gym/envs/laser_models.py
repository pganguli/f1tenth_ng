"""
Prototype of Utility functions and classes for simulating 2D LIDAR scans
Author: Hongrui Zheng
"""

import os
from typing import Optional

import numpy as np
import yaml
from numba import njit
from PIL import Image
from scipy.ndimage import distance_transform_edt as edt


def get_dt(bitmap: np.ndarray, resolution: float) -> np.ndarray:
    """
    Distance transformation, returns the distance matrix from the input bitmap.
    Uses scipy.ndimage, cannot be JITted.

        Args:
            bitmap (numpy.ndarray, (n, m)): input binary bitmap of the environment,
                where 0 is obstacles, and 255 (or anything > 0) is freespace
            resolution (float): resolution of the input bitmap (m/cell)

        Returns:
            dt (numpy.ndarray, (n, m)): output distance matrix, where each cell has the
                                        corresponding distance (in meters) to the closest obstacle
    """
    dt = resolution * edt(bitmap)
    return dt


@njit(cache=True)
def xy_2_rc(
    x: float,
    y: float,
    map_params: tuple,
) -> tuple[int, int]:
    """
    Translate (x, y) coordinate into (r, c) in the matrix

        Args:
            x (float): coordinate in x (m)
            y (float): coordinate in y (m)
            map_params (tuple): (orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)

        Returns:
            r (int): row number in the transform matrix of the given point
            c (int): column number in the transform matrix of the given point
    """
    orig_x, orig_y, orig_c, orig_s, height, width, resolution = map_params[:7]

    # rotation (inlined translation)
    x_rot = (x - orig_x) * orig_c + (y - orig_y) * orig_s
    y_rot = -(x - orig_x) * orig_s + (y - orig_y) * orig_c

    # clip the state to be a cell
    if (
        x_rot < 0
        or x_rot >= width * resolution
        or y_rot < 0
        or y_rot >= height * resolution
    ):
        c = -1
        r = -1
    else:
        c = int(x_rot / resolution)
        r = int(y_rot / resolution)

    return r, c


@njit(cache=True)
def distance_transform(
    x: float,
    y: float,
    map_params: tuple,
) -> float:
    """
    Look up corresponding distance in the distance matrix

        Args:
            x (float): x coordinate of the lookup point
            y (float): y coordinate of the lookup point
            map_params (tuple): (orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)

        Returns:
            distance (float): corresponding shortest distance to obstacle in meters
    """
    dt = map_params[7]
    r, c = xy_2_rc(x, y, map_params)
    distance = dt[r, c]
    return float(distance)


@njit(cache=True)
def trace_ray(
    x: float,
    y: float,
    theta_index: float,
    scan_params: tuple,
    map_params: tuple,
) -> float:
    """
    Find the length of a specific ray at a specific scan angle theta
    Purely math calculation and loops, should be JITted.

        Args:
            x (float): current x coordinate of the ego (scan) frame
            y (float): current y coordinate of the ego (scan) frame
            theta_index(int): current index of the scan beam in the scan range
            sines (numpy.ndarray (n, )): pre-calculated sines of the angle array
            cosines (numpy.ndarray (n, )): pre-calculated cosines ...

        Returns:
            total_distance (float): the distance to first obstacle on the current scan beam
    """

    # int casting, and index precal trigs
    theta_index_ = int(theta_index)
    sines = scan_params[4]
    cosines = scan_params[5]
    eps = scan_params[6]
    max_range = scan_params[7]

    s = sines[theta_index_]
    c = cosines[theta_index_]

    # distance to nearest initialization
    dist_to_nearest = distance_transform(x, y, map_params)
    total_dist = dist_to_nearest

    # ray tracing iterations
    while dist_to_nearest > eps and total_dist <= max_range:
        # move in the direction of the ray by dist_to_nearest
        x += dist_to_nearest * c
        y += dist_to_nearest * s

        # update dist_to_nearest for current point on ray
        # also keeps track of total ray length
        dist_to_nearest = distance_transform(x, y, map_params)
        total_dist += dist_to_nearest

    total_dist = min(total_dist, max_range)

    return total_dist


@njit(cache=True)
def get_scan(
    pose: np.ndarray,
    scan_params: tuple,
    map_params: tuple,
) -> np.ndarray:
    """
    Perform the scan for each discretized angle of each beam of the laser,
    loop heavy, should be JITted

        Args:
            pose (numpy.ndarray(3, )): current pose of the scan frame in the map
            scan_params (tuple): (theta_dis, fov, num_beams, theta_index_increment,
                                 sines, cosines, eps, max_range)
            map_params (tuple): (orig_x, orig_y, orig_c, orig_s, height, width,
                                resolution, dt)

        Returns:
            scan (numpy.ndarray(n, )): resulting laser scan at the pose, n=num_beams
    """
    theta_dis, fov, num_beams, theta_index_increment, _, _, _, _ = scan_params

    # empty scan array init
    scan = np.empty((num_beams,))

    # make theta discrete by mapping the range [-pi, pi] onto [0, theta_dis]
    theta_index = theta_dis * (pose[2] - fov / 2.0) / (2.0 * np.pi)

    # make sure it's wrapped properly
    theta_index = np.fmod(theta_index, theta_dis)
    while theta_index < 0:
        theta_index += theta_dis

    # sweep through each beam
    for i in range(0, num_beams):
        # trace the current beam
        scan[i] = trace_ray(
            pose[0],
            pose[1],
            theta_index,
            scan_params,
            map_params,
        )

        # increment the beam index
        theta_index += theta_index_increment

        # make sure it stays in the range [0, theta_dis)
        while theta_index >= theta_dis:
            theta_index -= theta_dis

    return scan


@njit(cache=True, error_model="numpy")
def check_ttc_jit(
    scan: np.ndarray,
    vel: float,
    cosines: np.ndarray,
    side_distances: np.ndarray,
    ttc_thresh: float,
) -> bool:
    """
    Checks the iTTC of each beam in a scan for collision with environment

    Args:
        scan (np.ndarray(num_beams, )): current scan to check
        vel (float): current velocity
        cosines (np.ndarray(num_beams, )): precomped cosines of the scan angles
        side_distances (np.ndarray(num_beams, )): precomped distances at each beam
            from the laser to the sides of the car
        ttc_thresh (float): threshold for iTTC for collision

    Returns:
        in_collision (bool): whether vehicle is in collision with environment
    """
    in_collision = False
    if vel != 0.0:
        num_beams = scan.shape[0]
        for i in range(num_beams):
            proj_vel = vel * cosines[i]
            ttc = (scan[i] - side_distances[i]) / proj_vel
            if 0.0 <= ttc < ttc_thresh:
                in_collision = True
                break
    else:
        in_collision = False

    return in_collision


@njit(cache=True)
def cross(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cross product of two 2-vectors

    Args:
        v1, v2 (np.ndarray(2, )): input vectors

    Returns:
        crossproduct (float): cross product
    """
    return float(v1[0] * v2[1] - v1[1] * v2[0])


@njit(cache=True)
def are_collinear(pt_a: np.ndarray, pt_b: np.ndarray, pt_c: np.ndarray) -> bool:
    """
    Checks if three points are collinear in 2D

    Args:
        pt_a, pt_b, pt_c (np.ndarray(2, )): points to check in 2D

    Returns:
        col (bool): whether three points are collinear
    """
    tol = 1e-8
    ba = pt_b - pt_a
    ca = pt_a - pt_c
    col = np.fabs(cross(ba, ca)) < tol
    return col


@njit(cache=True)
def get_range(
    pose, beam_theta: float, va, vb
) :
    """
    Get the distance at a beam angle to the vector formed by two of the four vertices of a vehicle

    Args:
        pose (np.ndarray(3, )): pose of the scanning vehicle
        beam_theta (float): angle of the current beam (world frame)
        va, vb (np.ndarray(2, )): the two vertices forming an edge

    Returns:
        distance (float): smallest distance at beam theta from scanning pose to edge
    """
    o = pose[0:2]
    vec_oa = o - va
    vec_ab = vb - va
    vec_normal = np.array(
        [np.cos(beam_theta + np.pi / 2.0), np.sin(beam_theta + np.pi / 2.0)]
    )

    denom = vec_ab.dot(vec_normal)
    distance = np.inf

    if np.fabs(denom) > 0.0:
        d1 = cross(vec_ab, vec_oa) / denom
        d2 = vec_oa.dot(vec_normal) / denom
        if d1 >= 0.0 and 0.0 <= d2 <= 1.0:
            distance = d1
    elif are_collinear(o, va, vb):
        da = np.linalg.norm(va - o)
        db = np.linalg.norm(vb - o)
        distance = min(da, db)

    return float(distance)


@njit(cache=True)
def get_blocked_view_indices(
    pose, vertices, scan_angles
) -> tuple[int, int]:
    """
    Get the indices of the start and end of blocked fov in scans by another vehicle

    Args:
        pose (np.ndarray(3, )): pose of the scanning vehicle
        vertices (np.ndarray(4, 2)): four vertices of a vehicle pose
        scan_angles (np.ndarray(num_beams, )): corresponding beam angles

    Returns:
        min_ind (int): index of the start of the blocked view
        max_ind (int): index of the end of the blocked view
    """
    # find four vectors formed by pose and 4 vertices:
    vecs = vertices - pose[:2]
    # norms = np.sqrt(np.sum(np.square(vecs), axis=1))
    norms = np.empty((4,))
    for i in range(4):
        norms[i] = np.sqrt(vecs[i, 0] ** 2 + vecs[i, 1] ** 2)
    unit_vecs = np.empty((4, 2))
    for i in range(4):
        unit_vecs[i, :] = vecs[i, :] / norms[i]

    # find angles between all four and pose vector
    pose_theta = pose[2]
    angles_with_x = np.empty((4,))
    for i in range(4):
        angle = pose_theta - np.arctan2(unit_vecs[i, 1], unit_vecs[i, 0])
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        angles_with_x[i] = -angle

    inds = [int(np.argmin(np.abs(scan_angles - angles_with_x[i]))) for i in range(4)]
    return min(inds), max(inds)


@njit(cache=True)
def ray_cast(
    pose: np.ndarray, scan: np.ndarray, scan_angles: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """
    Modify a scan by ray casting onto another agent's four vertices

    Args:
        pose (np.ndarray(3, )): pose of the vehicle performing scan
        scan (np.ndarray(num_beams, )): original scan to modify
        scan_angles (np.ndarray(num_beams, )): corresponding beam angles
        vertices (np.ndarray(4, 2)): four vertices of a vehicle pose

    Returns:
        new_scan (np.ndarray(num_beams, )): modified scan
    """
    # pad vertices so loops around
    looped_vertices = np.empty((5, 2))
    looped_vertices[0:4, :] = vertices
    looped_vertices[4, :] = vertices[0, :]

    min_ind, max_ind = get_blocked_view_indices(pose, vertices, scan_angles)
    # looping over beams
    for i in range(min_ind, max_ind + 1):
        # looping over vertices
        for j in range(4):
            # check if original scan is longer than ray casted distance
            scan_range = get_range(
                pose,
                pose[2] + scan_angles[i],
                looped_vertices[j, :],
                looped_vertices[j + 1, :],
            )
            if scan_range < scan[i]:
                scan[i] = scan_range
    return scan


@njit(cache=True)
def ray_cast_multiple(
    pose: np.ndarray,
    scan: np.ndarray,
    scan_angles: np.ndarray,
    opp_vertices: np.ndarray,
) -> np.ndarray:
    """
    Modify a scan by ray casting onto multiple other agents

    Args:
        pose (np.ndarray(3, )): pose of the vehicle performing scan
        scan (np.ndarray(num_beams, )): original scan to modify
        scan_angles (np.ndarray(num_beams, )): corresponding beam angles
        opp_vertices (np.ndarray(num_opps, 4, 2)): vertices of all other agents

    Returns:
        new_scan (np.ndarray(num_beams, )): modified scan
    """
    new_scan = scan
    for i in range(opp_vertices.shape[0]):
        new_scan = ray_cast(pose, new_scan, scan_angles, opp_vertices[i, :, :])
    return new_scan


# pylint: disable=too-many-instance-attributes
class ScanSimulator2D:
    """
    2D LIDAR scan simulator class

    Init params:
        num_beams (int): number of beams in the scan
        fov (float): field of view of the laser scan
        eps (float, default=0.0001): ray tracing iteration termination condition
        theta_dis (int, default=2000): number of steps to discretize the angles
            between 0 and 2pi for look up
        max_range (float, default=30.0): maximum range of the laser
    """

    def __init__(
        self,
        num_beams: int,
        fov: float,
        **kwargs,
    ):
        # initialization
        self.num_beams = num_beams
        self.fov = fov
        self.eps = kwargs.get("eps", 0.0001)
        self.theta_dis = kwargs.get("theta_dis", 2000)
        self.max_range = kwargs.get("max_range", 30.0)
        self.angle_increment = self.fov / (self.num_beams - 1)
        self.theta_index_increment = (
            self.theta_dis * self.angle_increment / (2.0 * np.pi)
        )
        self.orig_c = None
        self.orig_s = None
        self.orig_x = None
        self.orig_y = None
        self.map_height = None
        self.map_width = None
        self.map_resolution = None
        self.origin = None
        self.map_img = None
        self.dt = None

        # precomputing corresponding cosines and sines of the angle array
        theta_arr = np.linspace(0.0, 2 * np.pi, num=self.theta_dis)
        self.sines = np.sin(theta_arr)
        self.cosines = np.cos(theta_arr)

    # pylint: disable=too-many-return-statements
    def set_map(self, map_path: str, map_ext: str) -> None:
        """
        Set the bitmap of the scan simulator by path

            Args:
                map_path (str): path to the map yaml file
                map_ext (str): extension (image type) of the map image

            Returns:
                flag (bool): if image reading and loading is successful
        """
        # check if map yaml exists
        if not os.path.exists(map_path):
            print(f"Map yaml file not found at {map_path}")
            return False

        # load map image
        map_img_path = os.path.splitext(map_path)[0] + map_ext
        if not os.path.exists(map_img_path):
            print(f"Map image file not found at {map_img_path}")
            return False

        try:
            img_obj = Image.open(map_img_path).transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )
            # convert to grayscale if needed
            if img_obj.mode != "L":
                img_obj = img_obj.convert("L")
            self.map_img = np.array(img_obj)
        except (OSError, Image.UnidentifiedImageError) as ex:
            print(f"Error opening/processing map image: {ex}")
            return False

        self.map_img = self.map_img.astype(np.float64)

        # grayscale -> binary
        self.map_img[self.map_img <= 128.0] = 0.0
        self.map_img[self.map_img > 128.0] = 255.0

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        # load map yaml
        with open(map_path, "r", encoding="utf-8") as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata.get("resolution")
                self.origin = map_metadata.get("origin")
            except yaml.YAMLError as ex:
                print(f"Error loading map yaml: {ex}")
                return False

        if self.map_resolution is None or self.origin is None:
            print(f"Map metadata at {map_path} is missing 'resolution' or 'origin'.")
            return False

        # calculate map parameters
        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])

        # get the distance transform
        if self.map_resolution is None:
            return False

        self.dt = get_dt(self.map_img, self.map_resolution)

        return True

    def scan(
        self,
        pose,
        rng: Optional[np.random.Generator],
        std_dev: float = 0.01,
    ) :
        """
        Perform simulated 2D scan by pose on the given map

            Args:
                pose (numpy.ndarray (3, )): pose of the scan frame (x, y, theta)
                rng (numpy.random.Generator): random number generator to use for
                    whitenoise in scan, or None
                std_dev (float, default=0.01): standard deviation of the generated
                    whitenoise in the scan

            Returns:
                scan (numpy.ndarray (n, )): data array of the laserscan, n=num_beams

            Raises:
                ValueError: when scan is called before a map is set
        """

        if self.map_height is None:
            raise ValueError("Map is not set for scan simulator.")

        map_params = (
            self.orig_x,
            self.orig_y,
            self.orig_c,
            self.orig_s,
            self.map_height,
            self.map_width,
            self.map_resolution,
            self.dt,
        )
        scan_params = (
            self.theta_dis,
            self.fov,
            self.num_beams,
            self.theta_index_increment,
            self.sines,
            self.cosines,
            self.eps,
            self.max_range,
        )

        scan = get_scan(
            pose,
            scan_params,
            map_params,
        )

        if rng is not None:
            noise = rng.normal(0.0, std_dev, size=self.num_beams)
            scan += noise

        return scan

    def get_increment(self) -> float:
        """
        Get the increment of the scan angles

        Args:
            None

        Returns:
            increment (float): angle increment
        """
        return self.angle_increment
