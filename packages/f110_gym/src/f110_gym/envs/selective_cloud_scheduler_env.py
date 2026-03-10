"""Gymnasium environment for learning selective per-DNN cloud scheduling policies.

The RL agent controls *which* of the m=3 cloud DNNs (left-wall distance,
track-width, heading-error) to call on each step.  The action is a continuous
vector of logits ``Box(m,)``; the environment applies softmax and selects the
top-k indices as the ``call_mask`` passed to
:class:`~f110_planning.reactive.SelectiveEdgeCloudPlanner`.

This design lets the same environment work with any SB3 algorithm that supports
continuous action spaces (PPO, SAC, TD3, …).

The observation extends the base F110 observation with:

* ``edge_left_dist``      — current edge left-wall estimate (m)
* ``edge_track_width``    — current edge track-width estimate (m)
* ``edge_heading_error``  — current edge heading-error estimate (rad)
* ``cloud_left_dist``     — held/resolved cloud left-wall value (m)
* ``cloud_track_width``   — held/resolved cloud track-width value (m)
* ``cloud_heading_error`` — held/resolved cloud heading-error value (rad)
* ``last_steer``          — steering command applied on the previous step (rad)
* ``last_speed``          — speed command applied on the previous step (m/s)
* ``cloud_calls_mask``    — MultiBinary(m) indicating which DNNs were called

The following base-env keys are intentionally **excluded** from the agent
observation because they are either map-absolute (``poses_x``, ``poses_y``,
``poses_theta``) or very high-dimensional (``scans``):

    ``scans``, ``poses_x``, ``poses_y``, ``poses_theta``
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from f110_planning.metrics import crosstrack_error
from f110_planning.utils import F110_MAX_STEER


class SelectiveCloudSchedulerEnv(gym.Env):  # pylint: disable=too-many-instance-attributes
    """Gym env that exposes selective per-DNN cloud scheduling as the action.

    Parameters
    ----------
    map : str
        Path to map YAML without extension (forwarded to ``f110-v0``).
    waypoints : np.ndarray
        Reference waypoints used for CTE computation in the default reward.
    cloud_latency : int
        Round-trip latency in simulation steps for any cloud DNN call.
    alpha_steer, alpha_speed : float
        Edge-cloud blending weights for the underlying planner.
    top_k : int
        Number of DNNs to call per step (the env enforces this by picking the
        top-k softmax probabilities from the agent's logit vector).
    edge_*_model_path, cloud_*_model_path : str | None
        TorchScript ``.pt`` model paths forwarded to the planner.
    reward_fn : Callable[[dict, list[bool]], float] | None
        Custom reward function.  Receives the simulator observation and the
        resolved call mask.  Defaults to :meth:`_default_reward` (–CTE²).
    **env_kwargs
        Extra keyword arguments forwarded verbatim to ``f110-v0``.
    """

    metadata = {"render_modes": ["human", "human_fast"], "render_fps": 200}

    #: Names of the m=3 DNN slots — order matches SelectiveEdgeCloudPlanner
    DNN_NAMES: list[str] = ["left_wall", "track_width", "heading"]

    #: Logit action bounds.  Wide enough to express any preference ordering;
    #: softmax is shift-invariant so expressiveness is not reduced.
    ACTION_LOW: float = -10.0
    ACTION_HIGH: float = 10.0

    #: Base-env keys excluded from the agent observation.
    #: Kept: linear_vels_x, linear_vels_y, steering_angles
    #: - scans         : raw 1080-beam LiDAR (high-dim; agent uses DNN outputs instead)
    #: - poses_x/y     : absolute map-frame position (not transferable across maps)
    #: - poses_theta   : absolute map-frame heading (same reason)
    #: - ang_vels_z    : yaw rate (redundant given steering_angles + vels)
    #: - ego_idx       : always 0 in single-agent training
    #: - collisions    : episode termination signal, not a decision input
    #: - lap_times     : wall-clock; not relevant to DNN scheduling decisions
    #: - lap_counts    : same
    _OBS_EXCLUDED_KEYS: frozenset = frozenset({
        "scans", "poses_x", "poses_y", "poses_theta",
        "ang_vels_z", "ego_idx", "collisions", "lap_times", "lap_counts",
    })

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals, redefined-builtin
        self,
        *,
        map: str,
        waypoints: np.ndarray,
        cloud_latency: int = 10,
        alpha_steer: float = 0.7,
        alpha_speed: float = 0.7,
        top_k: int = 1,
        edge_left_wall_model_path: Optional[str] = None,
        edge_track_width_model_path: Optional[str] = None,
        edge_heading_model_path: Optional[str] = None,
        cloud_left_wall_model_path: Optional[str] = None,
        cloud_track_width_model_path: Optional[str] = None,
        cloud_heading_model_path: Optional[str] = None,
        reward_fn: Optional[Callable[[dict[str, Any], list[bool]], float]] = None,
        **env_kwargs: Any,
    ) -> None:
        m = len(self.DNN_NAMES)
        self._m = m
        self._top_k = top_k
        self._waypoints = waypoints.copy()

        # Underlying F110 simulator.
        # Always enforce num_agents=1: this env drives a single ego agent and
        # passes exactly one (steer, speed) pair per step.  Any num_agents
        # supplied by the caller is silently overridden to avoid an IndexError
        # in the simulator's control loop.
        env_kwargs.pop("num_agents", None)
        self._env = gym.make("f110_gym:f110-v0", map=map, num_agents=1, **env_kwargs)

        # Selective planner — deferred import to avoid circular deps
        from f110_planning.reactive import SelectiveEdgeCloudPlanner  # pylint: disable=import-outside-toplevel

        self._planner = SelectiveEdgeCloudPlanner(
            cloud_latency=cloud_latency,
            alpha_steer=alpha_steer,
            alpha_speed=alpha_speed,
            top_k=top_k,
            edge_left_wall_model_path=edge_left_wall_model_path,
            edge_track_width_model_path=edge_track_width_model_path,
            edge_heading_model_path=edge_heading_model_path,
            cloud_left_wall_model_path=cloud_left_wall_model_path,
            cloud_track_width_model_path=cloud_track_width_model_path,
            cloud_heading_model_path=cloud_heading_model_path,
        )

        self._reward_fn = reward_fn if reward_fn is not None else self._default_reward

        # -----------------------------------------------------------------
        # Spaces
        # -----------------------------------------------------------------
        # Action: continuous logits over m DNNs; env converts to top-k mask.
        # Bounded to [-10, 10] so on-policy algorithms (PPO, A2C) satisfy
        # SB3's finite-bounds assertion; softmax is invariant to shifts so
        # the practical expressiveness is unchanged.
        self.action_space = spaces.Box(
            low=self.ACTION_LOW, high=self.ACTION_HIGH, shape=(m,), dtype=np.float32
        )

        base_obs_space = self._env.observation_space
        assert isinstance(base_obs_space, spaces.Dict)

        pi = float(np.pi)
        extra: dict[str, spaces.Space] = {

            "edge_left_dist": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "edge_track_width": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "edge_heading_error": spaces.Box(-pi, pi, shape=(1,), dtype=np.float32),
            "cloud_left_dist": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "cloud_track_width": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "cloud_heading_error": spaces.Box(-pi, pi, shape=(1,), dtype=np.float32),
            "last_steer": spaces.Box(
                -float(F110_MAX_STEER), float(F110_MAX_STEER), shape=(1,), dtype=np.float32
            ),
            "last_speed": spaces.Box(0.0, 20.0, shape=(1,), dtype=np.float32),
            "cloud_calls_mask": spaces.MultiBinary(m),
        }
        filtered_base = {
            k: v for k, v in base_obs_space.spaces.items()
            if k not in self._OBS_EXCLUDED_KEYS
        }
        self.observation_space = spaces.Dict({**filtered_base, **extra})

        self._step: int = 0
        self._last_obs: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._planner.reset()
        self._step = 0
        self._last_obs = obs
        return self._augment_obs(obs), info or {}

    def step(self, action: np.ndarray):
        """Step the environment.

        Parameters
        ----------
        action : np.ndarray of shape (m,)
            Logits over the m DNN slots.  The env applies softmax and picks
            the top-k indices to form the call mask.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        call_mask = self._resolve_topk(action)

        if self._last_obs is None:
            raise RuntimeError("Environment not reset before stepping")

        control = self._planner.plan(self._last_obs, call_mask=call_mask)

        obs, _base_reward, terminated, truncated, info = self._env.step(
            np.array([[control.steer, control.speed]])
        )
        self._last_obs = obs

        reward = self._reward_fn(obs, call_mask)
        self._step += 1

        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        cte = float(crosstrack_error(pos, self._waypoints))
        new_info = {**info, "call_mask": call_mask, "step": self._step, "cte": cte}
        return self._augment_obs(obs), reward, terminated, truncated, new_info

    def render(self) -> None:
        return self._env.render()  # type: ignore[return-value]

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_topk(self, logits: np.ndarray) -> list[bool]:
        """Numerically-stable softmax → top-k bool mask."""
        logits = np.asarray(logits, dtype=np.float64)
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        top_k_idx = np.argsort(probs)[-self._top_k :]
        mask = [False] * self._m
        for i in top_k_idx:
            mask[i] = True
        return mask

    def _augment_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in obs.items() if k not in self._OBS_EXCLUDED_KEYS}
        p = self._planner
        out["edge_left_dist"] = np.array([p.last_edge_left], dtype=np.float32)
        out["edge_track_width"] = np.array([p.last_edge_track], dtype=np.float32)
        out["edge_heading_error"] = np.array([p.last_edge_heading], dtype=np.float32)
        out["cloud_left_dist"] = np.array([p.last_cloud_left], dtype=np.float32)
        out["cloud_track_width"] = np.array([p.last_cloud_track], dtype=np.float32)
        out["cloud_heading_error"] = np.array([p.last_cloud_heading], dtype=np.float32)
        last = p.last_action
        out["last_steer"] = np.array([last.steer if last is not None else 0.0], dtype=np.float32)
        out["last_speed"] = np.array([last.speed if last is not None else 0.0], dtype=np.float32)
        out["cloud_calls_mask"] = np.array(p.last_call_mask, dtype=np.int8)
        return out

    def _default_reward(self, obs: dict[str, Any], _call_mask: list[bool]) -> float:
        """Default reward: negative squared cross-track error only.

        No cloud-cost penalty — the agent must discover the cost/quality
        trade-off purely from the CTE signal.
        """
        pos = np.array([obs["poses_x"][0], obs["poses_y"][0]], dtype=np.float64)
        dist = crosstrack_error(pos, self._waypoints)
        return -float(dist ** 2)
