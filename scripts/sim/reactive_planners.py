#!/usr/bin/env python3
"""
Simulation script to test various reactive planners.
Supports: bubble, gap_follower, disparity, and dynamic.
"""

import argparse
import time
from typing import Any

import numpy as np
from f110_planning.reactive import (
    BubblePlanner,
    DisparityExtenderPlanner,
    DynamicWaypointPlanner,
    GapFollowerPlanner,
    LidarDNNPlanner,
)
from f110_planning.render_callbacks import (
    create_camera_tracking,
    create_dynamic_waypoint_renderer,
    create_heading_error_renderer,
    create_trace_renderer,
    create_waypoint_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.utils import add_common_sim_args, load_waypoints, setup_env

# Default configuration
DEFAULT_PLANNER = "dnn"


def parse_args():
    """
    Registers command-line arguments for reactive simulation experiments.
    """
    parser = argparse.ArgumentParser(
        description="F1TENTH Reactive Planner Evaluation Suite"
    )

    add_common_sim_args(parser)

    parser.add_argument(
        "--planner",
        type=str,
        choices=["bubble", "gap", "disparity", "dynamic", "dnn"],
        default=DEFAULT_PLANNER,
        help="Algorithm for obstacle avoidance and navigation.",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Overrides the default velocity for the chosen planner (m/s).",
    )

    parser.add_argument(
        "--lookahead",
        type=float,
        default=1.5,
        help="Adaptive lookahead gain for 'dynamic' and 'dnn' planners.",
    )

    parser.add_argument(
        "--lateral-gain",
        type=float,
        default=1.0,
        help="Aggressiveness factor for centering between walls.",
    )

    parser.add_argument(
        "--safety-radius",
        type=float,
        default=1.3,
        help="Collision avoidance radius (meters) for the Bubble Planner.",
    )

    parser.add_argument(
        "--bubble-radius",
        type=int,
        default=160,
        help="Number of LiDAR beams to mask for Gap Follower 'virtual bubbles'.",
    )

    parser.add_argument(
        "--wall-model",
        type=str,
        default="data/models/left_wall_dist,right_wall_dist_arch5.pth",
        help="Path to the trained DualHeadWallModel state dictionary.",
    )

    parser.add_argument(
        "--left-model",
        type=str,
        default=None,
        help="Path to a standalone Left wall distance model.",
    )

    parser.add_argument(
        "--right-model",
        type=str,
        default=None,
        help="Path to a standalone Right wall distance model.",
    )

    parser.add_argument(
        "--heading-model",
        type=str,
        default="data/models/heading_error_arch7.pth",
        help="Path to the trained HeadingError model.",
    )

    parser.add_argument(
        "--arch",
        type=int,
        default=5,
        help="Architecture index (1-7) used during wall-distance training.",
    )

    parser.add_argument(
        "--heading-arch",
        type=int,
        default=7,
        help="Architecture index (1-7) used during heading-error training.",
    )

    return parser.parse_args()


def _create_planner(args: argparse.Namespace, waypoints: np.ndarray) -> Any:
    """Instantiates the requested reactive planner based on CLI arguments."""
    if args.planner == "bubble":
        kwargs = {"safety_radius": args.safety_radius}
        if args.speed is not None:
            kwargs["avoidance_speed"] = args.speed
        return BubblePlanner(**kwargs)

    if args.planner == "gap":
        kwargs = {"bubble_radius": args.bubble_radius}
        if args.speed is not None:
            kwargs["corners_speed"] = min(4.0, args.speed)
            kwargs["straights_speed"] = args.speed
        return GapFollowerPlanner(**kwargs)

    if args.planner == "disparity":
        planner = DisparityExtenderPlanner()
        if args.speed is not None:
            planner.absolute_max_speed = args.speed
        return planner

    if args.planner == "dynamic":
        kwargs = {
            "waypoints": waypoints,
            "lookahead_distance": args.lookahead,
            "lateral_gain": args.lateral_gain,
        }
        if args.speed is not None:
            kwargs["max_speed"] = args.speed
        return DynamicWaypointPlanner(**kwargs)

    if args.planner == "dnn":
        kwargs = {
            "wall_model_path": args.wall_model,
            "left_model_path": args.left_model,
            "right_model_path": args.right_model,
            "heading_model_path": args.heading_model,
            "arch_id": args.arch,
            "heading_arch_id": args.heading_arch,
            "lookahead_distance": args.lookahead,
            "lateral_gain": args.lateral_gain,
        }
        if args.speed is not None:
            kwargs["max_speed"] = args.speed
        return LidarDNNPlanner(**kwargs)

    raise ValueError(f"Unsupported planner logic: {args.planner}")


def _setup_rendering(
    env: Any, args: argparse.Namespace, waypoints: np.ndarray, planner: Any
) -> None:
    """Configures environment render callbacks."""
    env.unwrapped.add_render_callback(create_camera_tracking(rotate=True))
    env.unwrapped.add_render_callback(render_lidar)
    env.unwrapped.add_render_callback(render_side_distances)
    env.unwrapped.add_render_callback(create_trace_renderer(agent_idx=0))

    if waypoints.size > 0:
        env.unwrapped.add_render_callback(create_waypoint_renderer(waypoints))
        env.unwrapped.add_render_callback(create_heading_error_renderer(waypoints, 0))

    if args.planner in ["dynamic", "dnn"]:
        env.unwrapped.add_render_callback(
            create_dynamic_waypoint_renderer(planner, agent_idx=0)
        )


def main() -> None:
    """
    Entry point for running reactive planning simulations.
    """
    args = parse_args()

    # Determine render mode and initialize environment
    r_mode = None if args.render_mode == "None" else args.render_mode
    waypoints = load_waypoints(args.waypoints)
    planner = _create_planner(args, waypoints)
    env = setup_env(args, r_mode)

    if r_mode:
        _setup_rendering(env, args, waypoints, planner)

    # Initial reset
    pose = np.array([[args.start_x, args.start_y, args.start_theta]])
    obs, _ = env.reset(options={"poses": pose})
    if r_mode:
        env.render()

    print(f"Executing {args.planner} simulation loop...")
    laptime, start_time, done = 0.0, time.time(), False

    try:
        while not done:
            action = planner.plan(obs, ego_idx=0)
            obs, reward, terminated, truncated, _ = env.step(
                np.array([[action.steer, action.speed]])
            )
            done, laptime = (terminated or truncated), laptime + float(reward)

            if r_mode:
                env.render()
    except KeyboardInterrupt:
        print("\nSimulation aborted by user.")

    total_real_time = time.time() - start_time
    print("\n--- Simulation Summary ---")
    print(f"Planner:           {args.planner}")
    print(f"Simulated Runtime: {laptime:.3f}s")
    print(f"Real Wall Time:    {total_real_time:.3f}s")
    if total_real_time > 0:
        print(f"RT-Factor:         {laptime / total_real_time:.2f}x")

    env.close()


if __name__ == "__main__":
    main()
