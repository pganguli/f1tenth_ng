#!/usr/bin/env python3
"""Train an RL policy for selective per-DNN cloud scheduling.

Uses ``f110_gym:f110-selective-cloud-scheduler-v0`` where the agent outputs
logits over m=3 cloud DNNs and the environment picks the top-k to call each
step.  The RL algorithm and reward function are both selectable via CLI flags
for easy experimentation.

Usage example (single map, single CPU)::

    python train_rl.py \\
        --map data/maps/F1/Oschersleben/Oschersleben_map \\
        --waypoints data/maps/F1/Oschersleben/Oschersleben_centerline.tsv \\
        --agent ppo --reward cte_only --top-k 1 \\
        --alpha-steer 0.5 --alpha-speed 0.1 \\
        --timesteps 5000000

Multi-map + multi-CPU (e.g. Slurm node with 32 cores and a GPU)::

    python train_rl.py \\
        --map  data/maps/F1/Oschersleben/Oschersleben_map \\
               data/maps/F1/Monza/Monza_map \\
        --waypoints data/maps/F1/Oschersleben/Oschersleben_centerline.tsv \\
                   data/maps/F1/Monza/Monza_centerline.tsv \\
        --n-envs 16 --device auto \\
        --timesteps 10000000 --n-steps 4096 --batch-size 512

Resume from checkpoint::

    python train_rl.py ... --resume data/models/rl_cloud_scheduler_2000000_steps.zip
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

import f110_gym  # noqa: F401 — registers Gym environments
import gymnasium as gym
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from f110_planning.utils import load_waypoints
from f110_scripts.train.agents import make_agent
from f110_scripts.train.rewards import make_reward


# ---------------------------------------------------------------------------
# Picklable environment factory (required for SubprocVecEnv)
# ---------------------------------------------------------------------------

def _make_single_env(
    map_path: str,
    waypoints: np.ndarray,
    reward_name: str,
    cloud_latency: int,
    alpha_steer: float,
    alpha_speed: float,
    top_k: int,
    max_episode_steps: int,
    edge_left_wall_model: str,
    edge_track_width_model: str,
    edge_heading_model: str,
    cloud_left_wall_model: str,
    cloud_track_width_model: str,
    cloud_heading_model: str,
) -> gym.Env:
    """Build and wrap a single training environment instance.

    Kept as a module-level function (not a lambda or closure) so it can be
    pickled by :class:`~stable_baselines3.common.vec_env.SubprocVecEnv`.
    """
    # Re-import inside the worker to ensure f110_gym is registered in subprocesses.
    import f110_gym as _  # noqa: F401
    from f110_scripts.train.rewards import make_reward as _make_reward

    reward_fn = _make_reward(reward_name, waypoints=waypoints)
    env = gym.make(
        "f110_gym:f110-selective-cloud-scheduler-v0",
        map=map_path,
        waypoints=waypoints,
        cloud_latency=cloud_latency,
        alpha_steer=alpha_steer,
        alpha_speed=alpha_speed,
        top_k=top_k,
        edge_left_wall_model_path=edge_left_wall_model,
        edge_track_width_model_path=edge_track_width_model,
        edge_heading_model_path=edge_heading_model,
        cloud_left_wall_model_path=cloud_left_wall_model,
        cloud_track_width_model_path=cloud_track_width_model,
        cloud_heading_model_path=cloud_heading_model,
        reward_fn=reward_fn,
        render_mode=None,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selective-DNN RL training."""
    parser = argparse.ArgumentParser(
        description="Train RL selective cloud-DNN scheduler policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----- environment -----
    parser.add_argument(
        "--map", type=str, nargs="+",
        metavar="MAP",
        default=["data/maps/F1/Oschersleben/Oschersleben_map"],
        help=(
            "Map YAML path(s) without extension.  Provide multiple values to "
            "train across several tracks (environments are distributed "
            "round-robin across the supplied maps)."
        ),
    )
    parser.add_argument(
        "--waypoints", type=str, nargs="+",
        metavar="WAYPOINTS",
        default=["data/maps/F1/Oschersleben/Oschersleben_centerline.tsv"],
        help="Waypoint TSV file(s) matching each --map entry.",
    )
    parser.add_argument(
        "--cloud-latency", type=int, default=10,
        help="Cloud round-trip latency (steps)",
    )
    parser.add_argument(
        "--top-k", type=int, default=1,
        help="Number of cloud DNNs to call per step (out of m=3)",
    )
    parser.add_argument(
        "--alpha-steer", type=float, default=0.5,
        help="Cloud blending weight for steering (0=edge only, 1=cloud only)",
    )
    parser.add_argument(
        "--alpha-speed", type=float, default=0.1,
        help="Cloud blending weight for speed (0=edge only, 1=cloud only)",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=2000,
        help="TimeLimit wrapper: max steps per episode.",
    )

    # ----- model paths -----
    models = parser.add_argument_group("models", "TorchScript .pt model paths")
    models.add_argument(
        "--edge-left-wall-model", type=str,
        default="data/models/left_wall_dist_arch1.pt",
    )
    models.add_argument(
        "--edge-track-width-model", type=str,
        default="data/models/track_width_arch1.pt",
    )
    models.add_argument(
        "--edge-heading-model", type=str,
        default="data/models/heading_error_arch1.pt",
    )
    models.add_argument(
        "--cloud-left-wall-model", type=str,
        default="data/models/left_wall_dist_arch6.pt",
    )
    models.add_argument(
        "--cloud-track-width-model", type=str,
        default="data/models/track_width_arch6.pt",
    )
    models.add_argument(
        "--cloud-heading-model", type=str,
        default="data/models/heading_error_arch6.pt",
    )

    # ----- agent & reward -----
    parser.add_argument(
        "--agent", type=str, default="ppo",
        help="RL algorithm: ppo | sac | td3 | a2c (or any key in agents.REGISTRY)",
    )
    parser.add_argument(
        "--reward", type=str, default="cte_only",
        help="Reward function name (see rewards.REGISTRY)",
    )

    # ----- hardware -----
    parser.add_argument(
        "--n-envs", type=int, default=1,
        help=(
            "Number of parallel training environments.  >1 spawns a "
            "SubprocVecEnv (one OS process per env) and is ideal for "
            "multi-core CPU nodes.  On-policy algorithms (PPO, A2C) benefit "
            "most; off-policy ones (SAC, TD3) typically use n-envs=1."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "'auto' lets SB3 pick CUDA when available (recommended for Slurm "
            "GPU nodes), 'cpu' forces CPU-only."
        ),
    )

    # ----- training -----
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--save-path", type=str, default="data/models/cloud_scheduler.zip",
        help="Destination for the final trained policy",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=100_000,
        help=(
            "Save a checkpoint every N timesteps (per env; multiply by "
            "--n-envs for wall-clock frequency)."
        ),
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        metavar="ZIP",
        help=(
            "Path to a previously saved .zip checkpoint to resume training "
            "from.  The model weights and optimizer state are restored; "
            "training continues for --timesteps additional steps (with "
            "reset_num_timesteps=False so TensorBoard step counts stay "
            "continuous).  The environment configuration (maps, alphas, …) "
            "is still taken from the current CLI flags."
        ),
    )
    parser.add_argument(
        "--eval-freq", type=int, default=0,
        help=(
            "Evaluate the current policy every N timesteps on a held-out env "
            "and log the mean episode reward.  0 = disabled."
        ),
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5,
        help="Number of episodes per evaluation (used when --eval-freq > 0).",
    )
    parser.add_argument(
        "--progress-bar", action="store_true", default=False,
        help="Show a tqdm progress bar during training.",
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level for SB3 (0=silent, 1=info, 2=debug)",
    )

    # ----- algorithm hyperparameters (forwarded to SB3) -----
    hp = parser.add_argument_group(
        "hyperparameters",
        "Algorithm hyperparameters forwarded to the SB3 constructor.  "
        "Unrecognised flags for the chosen algorithm are silently ignored.",
    )
    hp.add_argument(
        "--lr", "--learning-rate", dest="learning_rate", type=float, default=3e-4,
        help="Learning rate.",
    )
    hp.add_argument(
        "--n-steps", type=int, default=2048,
        help="On-policy rollout steps per env per update (PPO/A2C).",
    )
    hp.add_argument(
        "--batch-size", type=int, default=256,
        help=(
            "Minibatch size for gradient updates.  Larger values (256–512) "
            "improve GPU utilisation; must evenly divide n_steps × n_envs."
        ),
    )
    hp.add_argument(
        "--n-epochs", type=int, default=10,
        help="PPO: number of passes over the rollout buffer per update.",
    )
    hp.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor.",
    )
    hp.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="GAE λ for advantage estimation (PPO/A2C).",
    )
    hp.add_argument(
        "--ent-coef", type=float, default=0.01,
        help=(
            "Entropy coefficient.  A small positive value (0.01) encourages "
            "the agent to explore all three DNN slots rather than collapsing "
            "to always calling the same one."
        ),
    )
    hp.add_argument(
        "--clip-range", type=float, default=0.2,
        help="PPO clipping parameter ε.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_agent_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return only the hyperparameter kwargs that were explicitly set."""
    mapping = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "clip_range": args.clip_range,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _build_vec_env(args: argparse.Namespace) -> tuple[Any, list[np.ndarray]]:
    """Construct a (Sub)ProcVecEnv across all maps, returning (vec_env, all_waypoints)."""
    maps = args.map
    wpt_files = args.waypoints

    if len(maps) != len(wpt_files):
        raise ValueError(
            f"--map ({len(maps)} entries) and --waypoints ({len(wpt_files)} entries) "
            "must have the same number of values."
        )

    all_waypoints = [load_waypoints(f) for f in wpt_files]
    n_envs = args.n_envs

    # Distribute envs across maps round-robin
    def _factory(idx: int):
        map_idx = idx % len(maps)
        waypoints = all_waypoints[map_idx]
        map_path = maps[map_idx]
        # Capture everything by value (not args reference) for clean pickling
        return lambda: _make_single_env(
            map_path=map_path,
            waypoints=waypoints,
            reward_name=args.reward,
            cloud_latency=args.cloud_latency,
            alpha_steer=args.alpha_steer,
            alpha_speed=args.alpha_speed,
            top_k=args.top_k,
            max_episode_steps=args.max_episode_steps,
            edge_left_wall_model=args.edge_left_wall_model,
            edge_track_width_model=args.edge_track_width_model,
            edge_heading_model=args.edge_heading_model,
            cloud_left_wall_model=args.cloud_left_wall_model,
            cloud_track_width_model=args.cloud_track_width_model,
            cloud_heading_model=args.cloud_heading_model,
        )

    factories = [_factory(i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(factories) if n_envs > 1 else DummyVecEnv(factories)
    return vec_env, all_waypoints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse args, build environments, train, and save policy."""
    args = parse_args()

    vec_env, all_waypoints = _build_vec_env(args)

    agent_kwargs = _collect_agent_kwargs(args)

    if args.resume and Path(args.resume).exists():
        # Resume: load weights + optimiser state, swap in the new env.
        print(f"Resuming from checkpoint: {args.resume}")
        from f110_scripts.train.agents import REGISTRY  # pylint: disable=import-outside-toplevel
        if not REGISTRY:
            from stable_baselines3 import A2C, PPO, SAC, TD3  # pylint: disable=import-outside-toplevel
            REGISTRY.update({"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C})
        algo_cls = REGISTRY[args.agent.lower()]
        model = algo_cls.load(
            args.resume,
            env=vec_env,
            device=args.device,
            verbose=args.verbose,
            **agent_kwargs,
        )
        reset_num_timesteps = False
    else:
        model = make_agent(
            args.agent,
            vec_env,
            device=args.device,
            verbose=args.verbose,
            **agent_kwargs,
        )
        reset_num_timesteps = True

    # ----- callbacks -----
    checkpoint_dir = str(Path(args.save_path).parent)
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks: list[Any] = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=checkpoint_dir,
            name_prefix="rl_cloud_scheduler",
            verbose=1,
        )
    ]

    if args.eval_freq > 0:
        # Build a separate single eval env on the first map.
        # Use SubprocVecEnv to match the training vec env type and silence SB3's
        # "Training and eval env are not of the same type" warning.
        _eval_factory = [lambda: _make_single_env(  # type: ignore[misc]
            map_path=args.map[0],
            waypoints=all_waypoints[0],
            reward_name=args.reward,
            cloud_latency=args.cloud_latency,
            alpha_steer=args.alpha_steer,
            alpha_speed=args.alpha_speed,
            top_k=args.top_k,
            max_episode_steps=args.max_episode_steps,
            edge_left_wall_model=args.edge_left_wall_model,
            edge_track_width_model=args.edge_track_width_model,
            edge_heading_model=args.edge_heading_model,
            cloud_left_wall_model=args.cloud_left_wall_model,
            cloud_track_width_model=args.cloud_track_width_model,
            cloud_heading_model=args.cloud_heading_model,
        )]
        eval_env = SubprocVecEnv(_eval_factory) if args.n_envs > 1 else DummyVecEnv(_eval_factory)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=checkpoint_dir,
                log_path=checkpoint_dir,
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                verbose=1,
            )
        )

    # ----- train -----
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=args.progress_bar,
    )

    model.save(args.save_path)
    print(f"Saved policy to {args.save_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
