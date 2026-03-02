"""
RaceCar class implementation for the F1TENTH gym environment.
Handles vehicle dynamics, LIDAR simulation, and collision checking.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .collision_models import get_all_vertices
from .dynamic_models import pid, vehicle_dynamics_st
from .integrator import Integrator
from .laser_models import ScanSimulator2D, check_ttc_jit, ray_cast_multiple
from .vehicle_params import VehicleParams


@dataclass
class LidarConfig:
    """LIDAR sensor configuration parameters."""

    num_beams: int
    fov: float
    lidar_dist: float
    ttc_thresh: float


@dataclass
class RaceCarConfig:
    """Configuration parameters for the race car simulation."""

    seed: int
    is_ego: bool
    time_step: float
    integrator: Integrator
    lidar: LidarConfig


@dataclass
class RaceCarVehicleParams:
    """Physical parameters of the vehicle."""

    params: dict[str, Any]
    params_tuple: VehicleParams


@dataclass
class RaceCarControlState:
    """Control inputs and buffers for steering and acceleration."""

    accel: float
    steer_angle_vel: float
    steer_buffer: np.ndarray
    steer_buffer_size: int


class RaceCar:
    """
    Base level race car class, handles the physics and laser scan of a single vehicle

    Data Members:
        v_params (RaceCarVehicleParams): physical vehicle parameters
        config (RaceCarConfig): simulation configuration
        state (np.ndarray (7, )): state vector
            [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        opp_poses (np.ndarray | None): current poses of other agents
        control (RaceCarControlState): control inputs and steering buffer
        in_collision (bool): collision indicator
        scan_rng (np.random.Generator): random number generator for scan noise

    """

    # static objects that don't need to be stored in class instances
    scan_simulator = None
    cosines = None
    scan_angles = None
    side_distances = None

    def __init__(
        self,
        params: dict[str, Any],
        seed: int,
        **kwargs: Any,
    ) -> None:
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf',
                'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min',
                'sv_max', 'v_switch', 'a_max': 9.51, 'v_min', 'v_max',
                'length', 'width'}
            is_ego (bool, default=False): ego identifier
            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser
            lidar_dist (float, default=0): vertical distance between LiDAR and backshaft

        Returns:
            None
        """

        num_beams = kwargs.get("num_beams", 1080)
        fov = kwargs.get("fov", 4.7)

        # initialization
        self.v_params = RaceCarVehicleParams(
            params=params,
            params_tuple=VehicleParams(
                mu=params["mu"],
                C_Sf=params["C_Sf"],
                C_Sr=params["C_Sr"],
                lf=params["lf"],
                lr=params["lr"],
                h=params["h"],
                m=params["m"],
                MoI=params["I"],
                s_min=params["s_min"],
                s_max=params["s_max"],
                sv_min=params["sv_min"],
                sv_max=params["sv_max"],
                v_switch=params["v_switch"],
                a_max=params["a_max"],
                v_min=params["v_min"],
                v_max=params["v_max"],
            ),
        )
        self.config = RaceCarConfig(
            seed=seed,
            is_ego=kwargs.get("is_ego", False),
            time_step=kwargs.get("time_step", 0.01),
            integrator=kwargs.get("integrator", Integrator.EULER),
            lidar=LidarConfig(
                num_beams=num_beams,
                fov=fov,
                lidar_dist=kwargs.get("lidar_dist", 0.0),
                ttc_thresh=0.005,
            ),
        )

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.state = np.zeros((7,))

        # pose of opponents in the world
        self.opp_poses = None

        # control state
        self.control = RaceCarControlState(
            accel=0.0,
            steer_angle_vel=0.0,
            steer_buffer=np.empty((0,)),
            steer_buffer_size=2,
        )

        # collision identifier
        self.in_collision = False

        self.scan_rng = np.random.default_rng(seed=self.config.seed)

        # initialize scan sim
        if RaceCar.scan_simulator is None:
            RaceCar._init_scan_simulator(params, num_beams, fov)

    @staticmethod
    def _init_scan_simulator(params: dict[str, Any], num_beams: int, fov: float) -> None:
        """Helper to initialize shared class-level scan simulator and arrays."""
        RaceCar.scan_simulator = ScanSimulator2D(num_beams, fov)
        scan_ang_incr = RaceCar.scan_simulator.get_increment()

        cosines = np.zeros((num_beams,))
        scan_angles = np.zeros((num_beams,))
        side_distances = np.zeros((num_beams,))

        dist_sides = params["width"] / 2.0
        dist_fr = (params["lf"] + params["lr"]) / 2.0

        for i in range(num_beams):
            angle = -fov / 2.0 + i * scan_ang_incr
            scan_angles[i] = angle
            cosines[i] = np.cos(angle)

            if angle > 0:
                if angle < np.pi / 2:
                    to_side = dist_sides / np.sin(angle)
                    to_fr = dist_fr / np.cos(angle)
                    side_distances[i] = min(to_side, to_fr)
                else:
                    to_side = dist_sides / np.cos(angle - np.pi / 2.0)
                    to_fr = dist_fr / np.sin(angle - np.pi / 2.0)
                    side_distances[i] = min(to_side, to_fr)
            else:
                if angle > -np.pi / 2:
                    to_side = dist_sides / np.sin(-angle)
                    to_fr = dist_fr / np.cos(-angle)
                    side_distances[i] = min(to_side, to_fr)
                else:
                    to_side = dist_sides / np.cos(-angle - np.pi / 2)
                    to_fr = dist_fr / np.sin(-angle - np.pi / 2)
                    side_distances[i] = min(to_side, to_fr)

        RaceCar.cosines = cosines
        RaceCar.scan_angles = scan_angles
        RaceCar.side_distances = side_distances

    def update_params(self, params: dict[str, Any]) -> None:
        """
        Updates the physical parameters of the vehicle
        Note that does not need to be called at initialization of class anymore

        Args:
            params (dict): new parameters for the vehicle

        Returns:
            None
        """
        self.v_params.params = params
        self.v_params.params_tuple = VehicleParams(
            mu=params["mu"],
            C_Sf=params["C_Sf"],
            C_Sr=params["C_Sr"],
            lf=params["lf"],
            lr=params["lr"],
            h=params["h"],
            m=params["m"],
            MoI=params["I"],
            s_min=params["s_min"],
            s_max=params["s_max"],
            sv_min=params["sv_min"],
            sv_max=params["sv_max"],
            v_switch=params["v_switch"],
            a_max=params["a_max"],
            v_min=params["v_min"],
            v_max=params["v_max"],
        )

    def set_map(self, map_path: str, map_ext: str) -> None:
        """
        Sets the map for scan simulator

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file
        """
        if RaceCar.scan_simulator is not None:
            RaceCar.scan_simulator.set_map(map_path, map_ext)

    def reset(self, pose: np.ndarray) -> None:
        """
        Resets the vehicle to a pose

        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to

        Returns:
            None
        """
        # clear control inputs
        self.control.accel = 0.0
        self.control.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        # clear state
        self.state = np.zeros((7,))
        self.state[0:2] = pose[0:2]
        self.state[4] = pose[2]
        self.control.steer_buffer = np.empty((0,))
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.config.seed)

    def ray_cast_agents(self, scan: np.ndarray) -> np.ndarray:
        """
        Ray cast onto other agents in the env, modify original scan

        Args:
            scan (np.ndarray, (n, )): original scan range array

        Returns:
            new_scan (np.ndarray, (n, )): modified scan
        """

        # starting from original scan
        new_scan = scan

        # loop over all opponent vehicle poses
        if self.opp_poses is not None and self.opp_poses.shape[0] > 0:
            # get vertices of all opponents
            opp_vertices = get_all_vertices(
                self.opp_poses,
                self.v_params.params["length"],
                self.v_params.params["width"],
            )

            if RaceCar.scan_angles is not None:
                new_scan = ray_cast_multiple(
                    np.append(self.state[0:2], self.state[4]),
                    new_scan,
                    RaceCar.scan_angles,
                    opp_vertices,
                )

        return new_scan

    def check_ttc(self, current_scan: np.ndarray) -> bool:
        """
        Check iTTC against the environment, sets vehicle states accordingly if collision occurs.
        Note that this does NOT check collision with other agents.

        state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        Args:
            current_scan (np.ndarray): current laser scan

        Returns:
            None
        """

        in_collision = False
        if (
            RaceCar.cosines is not None
            and RaceCar.side_distances is not None
            and check_ttc_jit is not None
        ):
            in_collision = check_ttc_jit(
                current_scan,
                self.state[3],
                RaceCar.cosines,
                RaceCar.side_distances,
                self.config.lidar.ttc_thresh,
            )

        # if in collision stop vehicle
        if in_collision:
            self.state[3:] = 0.0
            self.control.accel = 0.0
            self.control.steer_angle_vel = 0.0

        # update state
        self.in_collision = in_collision

        return in_collision

    def update_pose(self, raw_steer: float, vel: float) -> np.ndarray:
        """
        Steps the vehicle's physical simulation

        Args:
            steer (float): desired steering angle
            vel (float): desired longitudinal velocity

        Returns:
            current_scan
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        # apply steering delay
        steer = self._apply_steering_delay(raw_steer)

        # steering angle velocity input to steering velocity acceleration input
        accl, sv = pid(
            vel,
            steer,
            self.state[3],
            self.state[2],
            self.v_params.params_tuple,
        )

        # integrate dynamics
        self.state = self._integrate_dynamics(accl, sv)

        # bound yaw angle
        if self.state[4] > 2 * np.pi:
            self.state[4] = self.state[4] - 2 * np.pi
        elif self.state[4] < 0:
            self.state[4] += 2 * np.pi

        # update scan
        scan_pose = np.array(
            [
                self.state[0] + self.config.lidar.lidar_dist * np.cos(self.state[4]),
                self.state[1] + self.config.lidar.lidar_dist * np.sin(self.state[4]),
                self.state[4],
            ]
        )
        if RaceCar.scan_simulator is not None:
            current_scan = RaceCar.scan_simulator.scan(scan_pose, self.scan_rng)
        else:
            current_scan = np.zeros((self.config.lidar.num_beams,))

        return current_scan

    def _apply_steering_delay(self, raw_steer: float) -> float:
        """Helper to apply steering delay buffer."""
        if self.control.steer_buffer.shape[0] < self.control.steer_buffer_size:
            steer = 0.0
            self.control.steer_buffer = np.append(raw_steer, self.control.steer_buffer)
        else:
            steer = float(self.control.steer_buffer[-1])
            self.control.steer_buffer = self.control.steer_buffer[:-1]
            self.control.steer_buffer = np.append(raw_steer, self.control.steer_buffer)
        return steer

    def _integrate_dynamics(self, accl: float, sv: float) -> np.ndarray:
        """Helper to perform dynamics integration."""
        u = np.array([sv, accl])
        if self.config.integrator is Integrator.RK4:
            k1 = vehicle_dynamics_st(self.state, u, self.v_params.params_tuple)
            k2 = vehicle_dynamics_st(
                self.state + self.config.time_step * (k1 / 2),
                u,
                self.v_params.params_tuple,
            )
            k3 = vehicle_dynamics_st(
                self.state + self.config.time_step * (k2 / 2),
                u,
                self.v_params.params_tuple,
            )
            k4 = vehicle_dynamics_st(
                self.state + self.config.time_step * k3,
                u,
                self.v_params.params_tuple,
            )
            return self.state + self.config.time_step * (1 / 6) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )
        if self.config.integrator is Integrator.EULER:
            f = vehicle_dynamics_st(self.state, u, self.v_params.params_tuple)
            return self.state + self.config.time_step * f
        raise SyntaxError(
            f"Invalid Integrator Specified. Provided {self.config.integrator.name}."
            " Please choose RK4 or Euler"
        )

    def update_opp_poses(self, opp_poses: np.ndarray) -> None:
        """
        Updates the vehicle's information on other vehicles

        Args:
            opp_poses (np.ndarray(num_other_agents, 3)): updated poses of other agents

        Returns:
            None
        """
        self.opp_poses = opp_poses

    def update_scan(self, agent_scans: list[np.ndarray], agent_index: int) -> None:
        """
        Steps the vehicle's laser scan simulation
        Separated from update_pose because needs to update scan based on NEW
        poses of agents in the environment

        Args:
            agent scans list (modified in-place),
            agent index (int)

        Returns:
            None
        """

        current_scan = agent_scans[agent_index]

        # check ttc
        self.check_ttc(current_scan)

        # ray cast other agents to modify scan
        new_scan = self.ray_cast_agents(current_scan)

        agent_scans[agent_index] = new_scan
