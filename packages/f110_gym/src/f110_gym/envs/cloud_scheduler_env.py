"""Gymnasium environment for learning cloud scheduling policies.

This wrapper creates an underlying ``f110_gym:f110-v0`` environment along
with an :class:`~f110_planning.reactive.EdgeCloudPlanner` whose
:class:`~f110_planning.schedulers.RLScheduler` is controlled by the RL agent.
The action space is simply ``Discrete(2)`` (0=no cloud, 1=call cloud) and the
observation dictionary mirrors the underlying F110 observation with two extra
entries describing the planner state.

A customizable reward function may be passed; by default the environment
returns the negative squared cross-track error for the ego vehicle (as that is
closely related to the RMSE objective).  Users can override the reward to any
other signal that depends on the observation and chosen scheduling action.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# defer heavy planning imports until runtime to avoid circular
# dependencies when the ``f110_planning`` package is imported (e.g. during
# testing of schedulers).
from f110_planning.schedulers import RLScheduler
from f110_planning.metrics import crosstrack_error


class CloudSchedulerEnv(gym.Env):  # pylint: disable=too-many-instance-attributes
    """Gym environment that exposes cloud scheduling as the action.

    The environment forwards low-level control decisions to an internal
    ``EdgeCloudPlanner`` and returns the usual simulator observations.  The
    agent is responsible for toggling the scheduler on each step.
    """

    metadata = {"render_modes": ["human", "human_fast"], "render_fps": 200}

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals, redefined-builtin
        self,
        *,
        map: str,
        waypoints: np.ndarray,
        cloud_latency: int = 10,
        alpha_left: float = 0.5,
        alpha_track: float = 0.5,
        alpha_heading: float = 0.5,
        edge_left_wall_model_path: Optional[str] = None,
        edge_track_width_model_path: Optional[str] = None,
        edge_heading_model_path: Optional[str] = None,
        cloud_left_wall_model_path: Optional[str] = None,
        cloud_track_width_model_path: Optional[str] = None,
        cloud_heading_model_path: Optional[str] = None,
        reward_fn: Optional[Callable[[dict[str, Any], int], float]] = None,
        cloud_cost: float = 0.1,
        cloud_cost_window: int = 100,
        **env_kwargs: Any,
    ) -> None:
        """Initialise the scheduling environment.

        Parameters mirror those of ``f110_gym:f110-v0`` plus scheduler and
        planner configuration.  ``reward_fn`` receives the observation returned
        by the low-level environment (after the planner has stepped) and the
        binary scheduling action (0 or 1) and must return a scalar reward.
        Model paths should point to self-sufficient ``.pt`` TorchScript files
        produced by :func:`~f110_planning.utils.nn_models.save_as_torchscript`.
        Architecture and hyperparameter information is read directly from each
        file — no separate ``arch_id`` argument is required.

        ``cloud_cost`` scales the rolling call-rate penalty added to the
        default reward.  ``cloud_cost_window`` sets the number of recent steps
        used to compute that rate (a smaller window makes the penalty more
        reactive; a larger window smooths it out).
        """
        # underlying simulator
        self._env = gym.make("f110_gym:f110-v0", map=map, **env_kwargs)
        self._waypoints = waypoints.copy()

        # scheduler and planner
        self._rl_scheduler = RLScheduler()
        # deferred import to break the circular dependency between f110_gym and
        # f110_planning (f110_planning.utils imports f110_gym internals).
        from f110_planning.reactive import EdgeCloudPlanner  # pylint: disable=import-outside-toplevel
        self._planner = EdgeCloudPlanner(
            cloud_latency=cloud_latency,
            alpha_left=alpha_left,
            alpha_track=alpha_track,
            alpha_heading=alpha_heading,
            scheduler=self._rl_scheduler,
            edge_left_wall_model_path=edge_left_wall_model_path,
            edge_track_width_model_path=edge_track_width_model_path,
            edge_heading_model_path=edge_heading_model_path,
            cloud_left_wall_model_path=cloud_left_wall_model_path,
            cloud_track_width_model_path=cloud_track_width_model_path,
            cloud_heading_model_path=cloud_heading_model_path,
        )

        # cloud-cost penalty configuration
        self._cloud_cost = cloud_cost
        self._cloud_cost_window = cloud_cost_window
        # rolling history of actions (0/1) for the call-rate penalty
        self._call_history: deque[int] = deque(maxlen=cloud_cost_window)

        # reward function
        self._reward_fn = reward_fn if reward_fn is not None else self._default_reward

        # spaces
        self.action_space = spaces.Discrete(2)
        base_obs_space = self._env.observation_space
        # add hints about cloud state
        extra = {
            "latest_cloud_action": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
            ),
            "cloud_request_pending": spaces.Discrete(2),
            # ensure shape is at least 1d so that SB3 feature extractors can
            # flatten without errors
            "crosstrack_dist": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64),
        }
        assert isinstance(base_obs_space, spaces.Dict)
        self.observation_space = spaces.Dict({**base_obs_space.spaces, **extra})

        # keep track of step count for info
        self._step = 0
        self._last_obs: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._planner.reset()
        self._rl_scheduler.reset()
        self._step = 0
        self._call_history.clear()
        self._last_obs = obs
        return self._augment_obs(obs), info or {}

    def step(self, action: int):
        """Apply a scheduling action and advance the environment by one step.

        Args:
            action (int): scheduling decision; ``1`` triggers a cloud call on
                this step, ``0`` uses the cached cloud result.

        Returns:
            tuple: ``(obs, reward, terminated, truncated, info)`` following the
                Gymnasium convention.  ``obs`` is the augmented observation dict
                (including ``latest_cloud_action``, ``cloud_request_pending``,
                and ``crosstrack_dist``).  ``info`` contains the keys forwarded
                from the underlying ``f110-v0`` environment plus
                ``latest_cloud_action``, ``cloud_request_pending``, and
                ``step``.
        """
        # scheduler action is applied before planner invocation
        self._rl_scheduler.set_action(bool(action))

        # planner requires the latest observation dictionary; fall back to
        # ``_last_obs`` which is set during reset and at the end of every
        # step.
        if self._last_obs is None:
            raise RuntimeError("Environment not reset before stepping")
        control = self._planner.plan(self._last_obs)

        # step the underlying gym env with the control output
        obs, _base_reward, terminated, truncated, info = self._env.step(
            np.array([[control.steer, control.speed]])
        )

        # update stored observation for next call
        self._last_obs = obs

        # record action before calling reward so _default_reward sees it
        self._call_history.append(int(action))

        reward = self._reward_fn(obs, action)
        self._step += 1

        # pylint: disable=protected-access
        new_info = {
            **info,
            "latest_cloud_action": np.array(
                [self._planner._latest_cloud_action.steer,
                 self._planner._latest_cloud_action.speed]
            )
            if self._planner._latest_cloud_action is not None
            else np.array([0.0, 0.0]),
            "cloud_request_pending": 1 if self._rl_scheduler._call_next else 0,
            "step": self._step,
        }
        return self._augment_obs(obs), reward, terminated, truncated, new_info

    def render(self) -> None:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _augment_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        # expense: copy to avoid surprising the caller if they mutate
        # pylint: disable=protected-access
        out = {**obs}
        out["latest_cloud_action"] = (
            np.array([self._planner._latest_cloud_action.steer,
                      self._planner._latest_cloud_action.speed])
            if self._planner._latest_cloud_action is not None
            else np.array([0.0, 0.0])
        )
        out["cloud_request_pending"] = 1 if self._rl_scheduler._call_next else 0
        # compute a simple cross-track distance for the ego agent so the agent
        # can access the error directly if desired.
        pos = np.array([out["poses_x"][0], out["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, self._waypoints)
        out["crosstrack_dist"] = np.array([float(dist)])
        return out

    def _default_reward(self, obs: dict[str, Any], _action: int) -> float:
        """Reward: negative squared CTE minus a rolling cloud call-rate penalty.

        The penalty term is ``-cloud_cost * call_rate`` where ``call_rate`` is
        the fraction of the last ``cloud_cost_window`` steps in which the agent
        chose to call the cloud (action=1).  This discourages the agent from
        always calling the cloud while still rewarding low cross-track error.
        """
        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, self._waypoints)
        cte_term = -float(dist ** 2)
        if self._cloud_cost > 0.0 and len(self._call_history) > 0:
            call_rate = sum(self._call_history) / len(self._call_history)
            cost_term = -self._cloud_cost * call_rate
        else:
            cost_term = 0.0
        return cte_term + cost_term
