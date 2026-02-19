"""
Simulator class for the F1TENTH Gym environment.
"""

from typing import Any

import numpy as np

from .collision_models import collision_multiple, get_all_vertices
from .race_car import RaceCar
from .simulator_params import SimulatorParams


class Simulator:
    """
    Simulation engine for F1TENTH.

    Handles the temporal update of all vehicles, collision detection, 
    and sensor (LIDAR) data generation.

    Attributes:
        num_agents (int): Number of simulated cars.
        time_step (float): Physics integration interval (dt).
        agent_poses (np.ndarray): Array of [x, y, theta] for all agents.
        steering_angles (np.ndarray): Current steering angle for all agents.
        agents (list[RaceCar]): List of individual RaceCar instances.
        collisions (np.ndarray): Boolean mask indicating if an agent has crashed.
    """

    def __init__(
        self,
        params: SimulatorParams,
    ) -> None:
        """
        Init function

        Args:
            params (SimulatorParams): simulation parameters

        Returns:
            None
        """
        self.params = params
        self.agent_poses = np.empty((self.params.num_agents, 3))
        self.agents = []
        self.collisions = np.zeros((self.params.num_agents,))
        self.collision_idx = -1 * np.ones((self.params.num_agents,))

        # initializing agents
        for i in range(self.params.num_agents):
            is_ego = i == self.params.ego_idx
            agent = RaceCar(
                params=self.params.vehicle_params,
                seed=self.params.seed,
                is_ego=is_ego,
                time_step=self.params.time_step,
                integrator=self.params.integrator,
                lidar_dist=self.params.lidar_dist,
            )
            self.agents.append(agent)

    def set_map(self, map_path: str, map_ext: str) -> None:
        """
        Sets the map of the environment and sets the map for scan simulator of each agent

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file

        Returns:
            None
        """
        for agent in self.agents:
            agent.set_map(map_path, map_ext)

    def update_params(self, params: dict[str, Any], agent_idx: int = -1) -> None:
        """
        Updates the params of agents, if an index of an agent is given, update
        only that agent's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agent that needs param update,
                                         if negative, update all agents

        Returns:
            None
        """
        if agent_idx < 0:
            # update params for all
            self.params.vehicle_params = params
            for agent in self.agents:
                agent.update_params(params)
        elif 0 <= agent_idx < self.params.num_agents:
            # only update one agent's params
            if agent_idx == self.params.ego_idx:
                self.params.vehicle_params = params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError("Index given is out of bounds for list of agents.")

    def check_collision(self) -> None:
        """
        Checks for collision between agents using GJK and agents' body vertices

        Args:
            None

        Returns:
            None
        """
        # get vertices of all agents
        all_vertices = get_all_vertices(
            self.agent_poses,
            self.params.vehicle_params["length"],
            self.params.vehicle_params["width"],
        )
        self.collisions, self.collision_idx = collision_multiple(all_vertices)

    def step(self, control_inputs: np.ndarray) -> dict[str, Any]:
        """
        Steps the simulation environment

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all
                agents, first column is desired steering angle, second column is
                desired velocity

        Returns:
            observations (dict): dictionary for observations: poses of agents,
                current laser scan of each agent, collision indicators, etc.
        """

        agent_scans = []

        # looping over agents
        for i, agent in enumerate(self.agents):
            # update each agent's pose
            current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])
            agent_scans.append(current_scan)

            # update sim's information of agent poses
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])

        # check collisions between all agents
        self.check_collision()

        for i, agent in enumerate(self.agents):
            # update agent's information on other agents
            opp_poses = np.concatenate(
                (self.agent_poses[0:i, :], self.agent_poses[i + 1 :, :]), axis=0
            )
            agent.update_opp_poses(opp_poses)

            # update each agent's current scan based on other agents
            agent.update_scan(agent_scans, i)

            # update agent collision with environment
            if agent.in_collision:
                self.collisions[i] = 1.0

        # fill in observations
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # collision_angles is removed from observations
        observations = {
            "ego_idx": self.params.ego_idx,
            "scans": np.array(agent_scans, dtype=np.float64),
            "poses_x": np.array(
                [agent.state[0] for agent in self.agents], dtype=np.float64
            ),
            "poses_y": np.array(
                [agent.state[1] for agent in self.agents], dtype=np.float64
            ),
            "poses_theta": np.array(
                [agent.state[4] for agent in self.agents], dtype=np.float64
            ),
            "linear_vels_x": np.array(
                [agent.state[3] for agent in self.agents], dtype=np.float64
            ),
            "steering_angles": np.array(
                [agent.state[2] for agent in self.agents], dtype=np.float64
            ),
            "linear_vels_y": np.zeros(self.params.num_agents, dtype=np.float64),
            "ang_vels_z": np.array(
                [agent.state[5] for agent in self.agents], dtype=np.float64
            ),
            "collisions": self.collisions.astype(np.float64),
        }

        return observations

    def reset(self, poses: np.ndarray) -> None:
        """
        Resets the simulation environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            None
        """

        if poses.shape[0] != self.params.num_agents:
            raise ValueError(
                "Number of poses for reset does not match number of agents."
            )

        # loop over poses to reset
        for i in range(self.params.num_agents):
            self.agents[i].reset(poses[i, :])
