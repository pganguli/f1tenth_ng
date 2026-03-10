"""Gymnasium wrapper that randomises the map at each episode reset.

On every call to :meth:`reset` one of the pre-built inner environments is
chosen uniformly at random and becomes the active delegate for the subsequent
episode.  All other Gymnasium methods (``step``, ``render``, ``close``,
``observation_space``, ``action_space``) are forwarded to the currently-active
env.

This gives each parallel SubprocVecEnv worker full exposure to every training
map, preventing the agent from learning map-specific shortcuts.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym


class MultiMapEnv(gym.Env):
    """Delegate-style wrapper that switches map randomly on each reset.

    Parameters
    ----------
    envs : list[gym.Env]
        Pre-built environments, one per map.  All must share the same
        ``observation_space`` and ``action_space`` (guaranteed when they are
        all instances of ``SelectiveCloudSchedulerEnv`` with the same config).
    seed : int | None
        Seed for the map-selection RNG.  Each worker should receive a
        different seed so they don't all pick the same map simultaneously.
    """

    metadata: dict = {}

    def __init__(self, envs: list[gym.Env], seed: Optional[int] = None) -> None:
        if not envs:
            raise ValueError("MultiMapEnv requires at least one inner env.")
        self._envs = envs
        self._active: gym.Env = envs[0]
        self._rng = np.random.default_rng(seed)

        # All inner envs share the same spaces — take from the first.
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.metadata = envs[0].metadata

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        idx = int(self._rng.integers(len(self._envs)))
        self._active = self._envs[idx]
        return self._active.reset(seed=seed, options=options)

    def step(self, action: Any):
        return self._active.step(action)

    def render(self):
        return self._active.render()

    def close(self) -> None:
        for env in self._envs:
            env.close()

    # ------------------------------------------------------------------
    # Pass-through properties used by SB3 / wrappers
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> gym.Env:
        return self._active.unwrapped

    @property
    def np_random(self):
        return self._active.np_random
