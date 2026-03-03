#!/usr/bin/env python3
"""Simple script to train an RL policy for cloud scheduling.

This script uses Stable Baselines3 to learn a binary scheduler policy using the
``f110_gym:f110-cloud-scheduler-v0`` environment.  The CLI exposes a few key
hyperparameters; the training loop is intentionally minimal.

Usage example::

    python scripts/train/train_rl.py \
        --map data/maps/F1/Oschersleben/Oschersleben_map \
        --waypoints data/maps/F1/Oschersleben/Oschersleben_centerline.tsv \
        --cloud-latency 10 \
        --timesteps 1000000 \
        --save-path data/models/cloud_scheduler.zip

The trained policy is saved in the provided ``--save-path``.
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym

# stable-baselines3 is optional, so import lazily to keep the script lightweight
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from f110_planning.utils import load_waypoints


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RL training."""
    parser = argparse.ArgumentParser(
        description="Train RL cloud scheduler policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--map", type=str, required=True, help="Map YAML (no extension)")
    parser.add_argument(
        "--waypoints", type=str, required=True, help="Waypoint file for ego vehicle"
    )
    parser.add_argument(
        "--num-agents", type=int, default=1, help="Number of agents (should be 1 for RL scheduling)"
    )
    parser.add_argument(
        "--cloud-latency", type=int, default=10, help="Cloud round-trip latency (steps)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="data/models/cloud_scheduler.zip",
        help="Where to store the learned policy",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity level for SB3")
    parser.add_argument(
        "--cloud-cost",
        type=float,
        default=0.1,
        help="Scale factor \u03bb for the rolling cloud call-rate penalty in the reward",
    )
    parser.add_argument(
        "--cloud-cost-window",
        type=int,
        default=100,
        help="Number of recent steps used to compute the rolling cloud call rate",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, build environment, train, and save the policy."""
    args = parse_args()

    waypoints = load_waypoints(args.waypoints)

    env = gym.make(
        "f110_gym:f110-cloud-scheduler-v0",
        map=args.map,
        waypoints=waypoints,
        cloud_latency=args.cloud_latency,
        num_agents=args.num_agents,
        cloud_cost=args.cloud_cost,
        cloud_cost_window=args.cloud_cost_window,
        render_mode=None,
    )
    # Wrap with vectorized environment for SB3
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=args.verbose,
        tensorboard_log="data/models/sb3_logs/rl_scheduler",
    )

    # callback to save periodic checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=str(Path(args.save_path).parent),
        name_prefix="rl_cloud_scheduler",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    # ensure directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path)
    print(f"Saved policy to {args.save_path}")


if __name__ == "__main__":
    main()
