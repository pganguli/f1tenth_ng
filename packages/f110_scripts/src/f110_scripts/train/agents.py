"""RL agent factory for cloud-scheduler training.

Add an entry to :data:`REGISTRY` to make a new algorithm selectable via
``--agent`` in ``train_rl.py``.  The value must be an SB3-compatible
on-policy or off-policy algorithm class whose constructor accepts::

    AgentClass(policy, env, verbose=..., tensorboard_log=..., **kwargs)

and whose instances expose ``.learn(total_timesteps, callback=...)`` and
``.save(path)``.

Supported out of the box
------------------------
* ``"ppo"``  — :class:`stable_baselines3.PPO`   (on-policy, discrete + continuous)
* ``"sac"``  — :class:`stable_baselines3.SAC`   (off-policy, continuous only)
* ``"td3"``  — :class:`stable_baselines3.TD3`   (off-policy, continuous only)
* ``"a2c"``  — :class:`stable_baselines3.A2C`   (on-policy)

Adding a new agent (e.g. TQC from sb3-contrib)::

    from sb3_contrib import TQC
    from f110_scripts.train.agents import REGISTRY
    REGISTRY["tqc"] = TQC
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _lazy_registry() -> dict[str, type]:
    """Build registry lazily so missing optional deps don't crash at import."""
    from stable_baselines3 import A2C, PPO, SAC, TD3  # pylint: disable=import-outside-toplevel

    return {
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
        "a2c": A2C,
    }


#: Maps algorithm name (lower-case) → SB3 algorithm class.
#: Populated on first call to :func:`make_agent`.
REGISTRY: dict[str, type] = {}


def make_agent(
    name: str,
    env: gym.Env,
    *,
    policy: str = "MultiInputPolicy",
    tensorboard_log: str = "data/models/sb3_logs/rl_scheduler",
    verbose: int = 1,
    **kwargs: Any,
) -> Any:
    """Instantiate an SB3 agent by name.

    Parameters
    ----------
    name : str
        Key in :data:`REGISTRY` (case-insensitive).
    env : gym.Env
        The Gymnasium environment to train on.
    policy : str
        SB3 policy class name.  ``"MultiInputPolicy"`` works for all built-in
        algorithms when the observation space is a ``Dict``.
    tensorboard_log : str
        Directory for TensorBoard logs.
    verbose : int
        SB3 verbosity level.
    **kwargs
        Additional keyword arguments forwarded to the algorithm constructor
        (e.g. ``learning_rate``, ``n_steps``, ``batch_size``).

    Returns
    -------
    SB3 algorithm instance with ``.learn()`` and ``.save()`` methods.

    Raises
    ------
    KeyError
        If ``name`` is not in :data:`REGISTRY`.
    """
    global REGISTRY  # pylint: disable=global-statement
    if not REGISTRY:
        REGISTRY.update(_lazy_registry())

    key = name.lower()
    if key not in REGISTRY:
        available = sorted(REGISTRY.keys())
        raise KeyError(f"Unknown agent '{name}'. Available: {available}")

    cls = REGISTRY[key]
    return cls(
        policy,
        env,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        **kwargs,
    )
