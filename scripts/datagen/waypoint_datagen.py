#!/usr/bin/env python3
"""
Data generation script for F1TENTH simulation.

Collects lidar scans, wall distance data, and heading errors while following waypoints.
Outputs data in NPZ format for efficient storage and loading.
Supports stochastic noise and drift injection to increase data variety for robust training.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from f110_planning.render_callbacks import (
    create_camera_tracking,
    create_trace_renderer,
    create_waypoint_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import (
    add_common_sim_args,
    get_heading_error,
    get_side_distances,
    load_waypoints,
    setup_env,
)

DEFAULT_OUTPUT_DIR = "data/datasets"
DEFAULT_STEERING_NOISE = 0.05
DEFAULT_DRIFT_PROB = 0.01
DEFAULT_DRIFT_MAGNITUDE = 0.3
DEFAULT_LOOKAHEAD_RANGE = [0.6, 1.2]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate F1TENTH simulation data with lidar scans and wall distances."
    )

    add_common_sim_args(parser)

    parser.add_argument(
        "--planner",
        type=str,
        choices=["pure_pursuit"],
        default="pure_pursuit",
        help="Planner to use for navigation. Default: pure_pursuit",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output NPZ file path. If not specified, generates filename from parameters.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum number of simulation steps. Default: 10000",
    )

    parser.add_argument(
        "--steering-noise",
        type=float,
        default=DEFAULT_STEERING_NOISE,
        help=(
            "Standard deviation of Gaussian noise added to steering (radians). "
            f"Default: {DEFAULT_STEERING_NOISE}"
        ),
    )

    parser.add_argument(
        "--drift-prob",
        type=float,
        default=DEFAULT_DRIFT_PROB,
        help=(
            "Probability per step of injecting a steering drift/perturbation. "
            f"Default: {DEFAULT_DRIFT_PROB}"
        ),
    )

    parser.add_argument(
        "--drift-magnitude",
        type=float,
        default=DEFAULT_DRIFT_MAGNITUDE,
        help=f"Magnitude of the steering drift bias. Default: {DEFAULT_DRIFT_MAGNITUDE}",
    )

    parser.add_argument(
        "--lookahead-range",
        type=float,
        nargs=2,
        default=DEFAULT_LOOKAHEAD_RANGE,
        help=(
            "Range [min, max] for randomized lookahead distance. "
            f"Default: {DEFAULT_LOOKAHEAD_RANGE}"
        ),
    )

    return parser.parse_args()


def create_planner(planner_type: str, waypoints: np.ndarray) -> PurePursuitPlanner:
    """
    Creates a planner instance based on the specified type.

    Args:
        planner_type: Type of planner to create ("pure_pursuit").
        waypoints: Reference path for the planner to follow.

    Returns:
        An initialized planner instance.
    """
    if planner_type == "pure_pursuit":
        return PurePursuitPlanner(waypoints=waypoints)

    raise ValueError(f"Unknown planner type: {planner_type}")


def setup_environment(
    args: argparse.Namespace,
) -> tuple[Any, PurePursuitPlanner, np.ndarray]:
    """
    Configures the simulation environment for data collection.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A tuple containing (environment, planner, waypoints).
    """
    waypoints = load_waypoints(args.waypoints)
    planner = create_planner(args.planner, waypoints)

    render_mode = args.render_mode if args.render_mode != "None" else None
    env = setup_env(args, render_mode)

    if args.render_mode in ["human", "human_fast"]:
        env.unwrapped.add_render_callback(create_camera_tracking(rotate=True))
        env.unwrapped.add_render_callback(render_lidar)
        env.unwrapped.add_render_callback(render_side_distances)
        env.unwrapped.add_render_callback(
            create_trace_renderer(color=(255, 255, 0), max_points=10000)
        )
        if waypoints.size > 0:
            env.unwrapped.add_render_callback(
                create_waypoint_renderer(waypoints, color=(255, 255, 255, 64))
            )

    return env, planner, waypoints


def _apply_steering_noise(
    steer: float, drift_active: int, drift_val: float, args: argparse.Namespace
) -> tuple[float, int, float]:
    """Applies stochastic noise and periodic drift to the steering command."""
    # 1. Stochastic Steering Noise
    if args.steering_noise > 0:
        steer += np.random.normal(0, args.steering_noise)

    # 2. Intermittent Drift Injection
    new_drift_active = drift_active
    new_drift_val = drift_val

    if new_drift_active > 0:
        steer += new_drift_val
        new_drift_active -= 1
    elif args.drift_prob > 0 and np.random.random() < args.drift_prob:
        new_drift_active = np.random.randint(10, 30)
        new_drift_val = np.random.choice([-1.0, 1.0]) * args.drift_magnitude
        steer += new_drift_val

    return steer, new_drift_active, new_drift_val


def _gather_step_data(
    obs: dict[str, np.ndarray], waypoints: np.ndarray, ego_idx: int = 0
) -> tuple[np.ndarray, float, float, float]:
    """Extracts LiDAR and error metrics from the current observation."""
    scan = obs["scans"][ego_idx]
    left_dist, right_dist = get_side_distances(scan)
    car_pos = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
    h_err = get_heading_error(waypoints, car_pos, obs["poses_theta"][ego_idx])
    return scan, left_dist, right_dist, h_err


def _get_noisy_action(
    planner: PurePursuitPlanner,
    obs: dict,
    drift_active: int,
    drift_val: float,
    args: argparse.Namespace,
) -> tuple[float, float, int, float]:
    """Generates a steering command with lookahead variation and noise."""
    l_min, l_max = args.lookahead_range
    planner.lookahead_distance = np.random.uniform(l_min, l_max)

    action = planner.plan(obs, ego_idx=0)
    steer, new_active, new_val = _apply_steering_noise(
        action.steer, drift_active, drift_val, args
    )
    return steer, action.speed, new_active, new_val


def _run_simulation(
    env: Any,
    planner: PurePursuitPlanner,
    waypoints: np.ndarray,
    args: argparse.Namespace,
    history: dict[str, list],
) -> int:
    """Runs the simulation loop and populates the history dictionary."""
    obs, _ = env.reset(
        options={"poses": np.array([[args.start_x, args.start_y, args.start_theta]])}
    )

    if args.render_mode in ["human", "human_fast"]:
        env.render()

    step, done, drift_active, drift_val = 0, False, 0, 0.0

    while step < args.max_steps and not done:
        steer, speed, drift_active, drift_val = _get_noisy_action(
            planner, obs, drift_active, drift_val, args
        )

        obs, _, term, trunc, _ = env.step(np.array([[steer, speed]]))

        if args.render_mode in ["human", "human_fast"]:
            env.render()

        # Gather and store data
        step_data = _gather_step_data(obs, waypoints)
        history["scans"].append(step_data[0])
        history["l_dists"].append(step_data[1])
        history["r_dists"].append(step_data[2])
        history["h_errors"].append(step_data[3])

        done, step = (term or trunc), step + 1
    return step


def collect_data(
    env: Any,
    planner: PurePursuitPlanner,
    waypoints: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Executes simulation steps to gather LiDAR and telemetry data.

    Args:
        env: Gymnasium environment.
        planner: The autonomous controller following the path.
        waypoints: The reference path array.
        args: Simulation configuration arguments.

    Returns:
        A tuple containing:
            - Dictionary of collected data channels (scans, distances, errors).
            - Total number of successful steps recorded.
    """
    history = {"scans": [], "l_dists": [], "r_dists": [], "h_errors": []}

    total_steps = _run_simulation(env, planner, waypoints, args, history)

    dataset = {
        "scans": np.array(history["scans"]),
        "left_wall_dist": np.array(history["l_dists"]),
        "right_wall_dist": np.array(history["r_dists"]),
        "heading_error": np.array(history["h_errors"]),
    }

    return dataset, total_steps


def save_data(data: dict[str, np.ndarray], output_path: str) -> None:
    """
    Serializes the collected simulation data to a compressed NPZ file.

    Args:
        data: Dictionary of data channels.
        output_path: Destination file path.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)


def generate_output_filename(args: argparse.Namespace, num_samples: int) -> str:
    """
    Constructs a descriptive filename based on simulation parameters.

    Args:
        args: Command-line arguments containing map and planner info.
        num_samples: Total count of data points collected.

    Returns:
        A structured string for the output path.
    """
    map_name = Path(args.map).name
    filename = f"lidar_tracking_{map_name}_{args.planner}_n{num_samples}.npz"
    return str(Path(DEFAULT_OUTPUT_DIR) / filename)


def main():
    """Main execution function."""
    args = parse_args()

    print(f"Setting up environment with map: {args.map}")
    print(f"Loading waypoints from: {args.waypoints}")
    print(f"Using planner: {args.planner}")

    env, planner, waypoints = setup_environment(args)

    print(f"Collecting data for up to {args.max_steps} steps...")
    if args.steering_noise > 0 or args.drift_prob > 0:
        print(
            f"Variance Boost Active: noise={args.steering_noise}, drift_prob={args.drift_prob}"
        )

    data, num_steps = collect_data(env, planner, waypoints, args)

    env.close()

    # Generate output filename if not specified
    output_path = (
        args.output if args.output else generate_output_filename(args, num_steps)
    )

    save_data(data, output_path)

    print(f"Saved {num_steps} samples to {output_path}")
    print("Data shapes:")
    for key, value in data.items():
        print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    main()
