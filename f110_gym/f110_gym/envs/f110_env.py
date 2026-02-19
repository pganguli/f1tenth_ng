"""
Author: Hongrui Zheng
"""

# gym imports
import time
from typing import Any, Optional

import gymnasium as gym

# others
import numpy as np
from numba import njit

# gl
import os
import pyglet
if os.environ.get('DISPLAY') is None:
    pyglet.options['headless'] = True
from gymnasium import spaces

# base classes
from .base_classes import Integrator, Simulator
from .rendering import EnvRenderer
from .simulator_params import SimulatorParams

pyglet.options["debug_gl"] = False

# rendering constants
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
@njit(cache=True)
def update_lap_counts(
    poses_x: np.ndarray,
    poses_y: np.ndarray,
    start_xs: np.ndarray,
    start_ys: np.ndarray,
    start_rot: np.ndarray,
    num_agents: int,
    current_time: float,
    near_starts: np.ndarray,
    toggle_list: np.ndarray,
    lap_counts: np.ndarray,
    lap_times: np.ndarray,
) -> None:
    """
    Update lap counts and times based on vehicle positions relative to start line.
    """
    left_t = 2.0
    right_t = 2.0

    for i in range(num_agents):
        dx = poses_x[i] - start_xs[i]
        dy = poses_y[i] - start_ys[i]

        tx = start_rot[0, 0] * dx + start_rot[0, 1] * dy
        ty = start_rot[1, 0] * dx + start_rot[1, 1] * dy

        if ty > left_t:
            ty -= left_t
        elif ty < -right_t:
            ty = -right_t - ty
        else:
            ty = 0.0

        dist2 = tx**2 + ty**2
        closes = dist2 <= 0.1

        if closes and not near_starts[i]:
            near_starts[i] = True
            toggle_list[i] += 1
        elif not closes and near_starts[i]:
            near_starts[i] = False
            toggle_list[i] += 1

        lap_counts[i] = toggle_list[i] // 2
        if toggle_list[i] < 4:
            lap_times[i] = current_time


class F110Env(gym.Env):
    """
    F1TENTH Gym Environment.

    This environment simulates the dynamics of high-speed 1/10th scale racing cars.
    It provides a Gymnasium-compatible interface for training and evaluating 
    navigation and control algorithms.

    State includes:
    - LIDAR scans (2D point clouds)
    - Ego poses (x, y, theta)
    - Linear and angular velocities
    - Steering angles (at current step)
    - Lap counters and times
    - Collision status

    Initialization (kwargs):
        seed (int): Random seed.
        map (str): Path to map yaml.
        params (dict): Vehicle physics parameters (mu, masses, etc.).
        num_agents (int): Number of racing agents.
        timestep (float): Simulation physics interval.
        ego_idx (int): Global index of the ego agent.
    """

    # pylint: disable=too-many-instance-attributes

    metadata = {"render_modes": ["human", "human_fast"], "render_fps": 200}

    # rendering
    renderer: Optional["EnvRenderer"] = None
    current_obs: Optional[dict[str, Any]] = None
    render_callbacks: list[Any] = []

    def __init__(self, **kwargs: Any) -> None:
        # kwargs extraction
        self.seed = kwargs.get("seed", 12345)
        self.render_mode = kwargs.get("render_mode", "human")
        self.render_fps = kwargs.get("render_fps", 100)
        self.last_render_time = None

        map_name = kwargs.get("map")
        if map_name is None:
            raise ValueError(
                "Map must be specified. Please provide 'map' parameter to gym.make()."
            )
        self.map_name = str(map_name)
        self.map_path = self.map_name + ".yaml"
        self.map_ext = str(kwargs.get("map_ext", ".png"))

        default_params = {
            "mu": 1.0489,
            "C_Sf": 4.718,
            "C_Sr": 5.4562,
            "lf": 0.15875,
            "lr": 0.17145,
            "h": 0.074,
            "m": 3.74,
            "I": 0.04712,
            "s_min": -0.4189,
            "s_max": 0.4189,
            "sv_min": -3.2,
            "sv_max": 3.2,
            "v_switch": 7.319,
            "a_max": 9.51,
            "v_min": -5.0,
            "v_max": 20.0,
            "width": 0.31,
            "length": 0.58,
        }
        self.params = kwargs.get("params", default_params)

        # simulation parameters
        self.num_agents = kwargs.get("num_agents", 2)
        self.timestep = kwargs.get("timestep", 0.01)

        # lap completion parameters
        self.max_laps = kwargs.get(
            "max_laps", 2
        )  # None to disable lap-based termination

        # default ego index
        self.ego_idx = kwargs.get("ego_idx", 0)

        # default integrator
        self.integrator = kwargs.get("integrator", Integrator.RK4)

        # default LiDAR position
        self.lidar_dist = kwargs.get("lidar_dist", 0.0)

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,), dtype=np.float64)

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,), dtype=np.float64)
        self.lap_counts = np.zeros((self.num_agents,), dtype=np.float64)
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,), dtype=np.float64)
        self.start_xs = np.zeros((self.num_agents,), dtype=np.float64)
        self.start_ys = np.zeros((self.num_agents,), dtype=np.float64)
        self.start_thetas = np.zeros((self.num_agents,), dtype=np.float64)
        self.start_rot = np.eye(2)

        # initiate stuff
        sim_params = SimulatorParams(
            vehicle_params=self.params,
            num_agents=self.num_agents,
            seed=self.seed,
            time_step=self.timestep,
            ego_idx=self.ego_idx,
            integrator=self.integrator,
            lidar_dist=self.lidar_dist,
        )
        self.sim = Simulator(sim_params)
        self.sim.set_map(self.map_path, self.map_ext)

        # stateful observations for rendering
        self.render_obs = None

        # Gymnasium requires action_space and observation_space
        # Action space: (num_agents, 2) - [steering, velocity] per agent
        self.action_space = spaces.Box(
            low=np.array(
                [[self.params["s_min"], self.params["v_min"]]] * self.num_agents,
                dtype=np.float64,
            ),
            high=np.array(
                [[self.params["s_max"], self.params["v_max"]]] * self.num_agents,
                dtype=np.float64,
            ),
            dtype=np.float64,
        )

        # Observation space: Dict space with various observations
        # scans shape depends on lidar config, using 1080 beams as default
        num_beams = 1080
        self.observation_space = spaces.Dict(
            {
                "ego_idx": spaces.Discrete(self.num_agents),
                "scans": spaces.Box(
                    low=0.0,
                    high=100.0,
                    shape=(self.num_agents, num_beams),
                    dtype=np.float64,
                ),
                "poses_x": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "poses_y": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "poses_theta": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "linear_vels_x": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "steering_angles": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_agents,),
                    dtype=np.float64,
                ),
                "linear_vels_y": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "ang_vels_z": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "collisions": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_agents,), dtype=np.float64
                ),
                "lap_times": spaces.Box(
                    low=0.0, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "lap_counts": spaces.Box(
                    low=0.0, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
            }
        )

    def _check_done(self) -> tuple[bool, bool]:
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (np.ndarray): boolean mask of agents that have finished the lap
        """

        # pylint: disable=too-many-locals

        update_lap_counts(
            np.array(self.poses_x),
            np.array(self.poses_y),
            self.start_xs,
            self.start_ys,
            self.start_rot,
            self.num_agents,
            self.current_time,
            self.near_starts,
            self.toggle_list,
            self.lap_counts,
            self.lap_times,
        )

        # Check for collision-based termination
        collision_done = bool(self.collisions[self.ego_idx])

        # Check for lap-based termination only if max_laps is set
        if self.max_laps is None:
            lap_done = False
        else:
            required_toggles = self.max_laps * 2
            lap_done = bool(np.all(self.toggle_list >= required_toggles))

        done = collision_done or lap_done

        return bool(done), lap_done

    def _update_state(self, obs_dict: dict[str, Any]) -> None:
        """
        Update the env's states according to observations

        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict["poses_x"]
        self.poses_y = obs_dict["poses_y"]
        self.poses_theta = obs_dict["poses_theta"]
        self.collisions = obs_dict["collisions"]

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            terminated (bool): if the simulation is terminated
            truncated (bool): if the simulation is truncated
            info (dict): auxillary information dictionary
        """

        # call simulation step
        obs = self.sim.step(action)
        obs["lap_times"] = self.lap_times
        obs["lap_counts"] = self.lap_counts

        F110Env.current_obs = obs

        self.render_obs = {
            "ego_idx": obs["ego_idx"],
            "poses_x": obs["poses_x"],
            "poses_y": obs["poses_y"],
            "poses_theta": obs["poses_theta"],
            "steering_angles": obs["steering_angles"],
            "lap_times": obs["lap_times"],
            "lap_counts": obs["lap_counts"],
            "scans": obs["scans"],
        }

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state(obs)

        # check done
        done, toggle_list = self._check_done()
        info = {"checkpoint_done": toggle_list}

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset the gym environment by given poses

        Args:
            seed: random seed for the environment
            options: dictionary of options, including 'poses'

        Returns:
            obs (dict): observation of the current step
            info (dict): auxillary information dictionary
        """
        super().reset(seed=seed)

        if options is not None:
            poses = options.get("poses")
        else:
            poses = None

        if poses is None:
            # Default poses if not provided (example fallback)
            poses = np.zeros((self.num_agents, 3))

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,), dtype=np.float64)
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,), dtype=np.float64)

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action)

        self.render_obs = {
            "ego_idx": obs["ego_idx"],
            "poses_x": obs["poses_x"],
            "poses_y": obs["poses_y"],
            "poses_theta": obs["poses_theta"],
            "steering_angles": obs["steering_angles"],
            "lap_times": obs["lap_times"],
            "lap_counts": obs["lap_counts"],
            "scans": obs["scans"],
        }

        return obs, info

    def update_map(self, map_path: str, map_ext: str) -> None:
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params: dict[str, Any], index: int = -1) -> None:
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func: Any) -> None:
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function
                                                            to call during render()
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self) -> None:
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out,
        use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner),
        and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time
                         elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """

        if F110Env.renderer is None:
            # first call, initialize everything
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self.map_name, self.map_ext)

        if self.render_obs is None:
            raise RuntimeError("Please call reset() before render().")

        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)

        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()

        if self.render_mode == "human":
            current_time = time.time()
            if self.last_render_time is not None:
                # Calculate how much we need to sleep to maintain target FPS
                elapsed = current_time - self.last_render_time
                sleep_time = (1.0 / self.render_fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.last_render_time = time.time()
        elif self.render_mode == "human_fast":
            pass
