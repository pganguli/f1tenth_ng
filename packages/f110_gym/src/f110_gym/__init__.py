"""
This module registers the F1TENTH Gym environment.
"""

from gymnasium.envs.registration import register

register(
    id="f110-v0",
    entry_point="f110_gym.envs:F110Env",
)

# environment used for training binary (call/no-call) RL cloud schedulers
register(
    id="f110-cloud-scheduler-v0",
    entry_point="f110_gym.envs:CloudSchedulerEnv",
)

# environment used for training selective per-DNN RL cloud schedulers
register(
    id="f110-selective-cloud-scheduler-v0",
    entry_point="f110_gym.envs:SelectiveCloudSchedulerEnv",
)

__all__ = ["register"]
