#!/usr/bin/env python3
"""
Simulation script for waypoint tracking with multiple agents.
Supports single-agent hybrid control (default) and multi-agent waypoint following.
"""

import argparse
import time
from typing import Any

import numpy as np
from f110_planning.misc import HybridPlanner, ManualPlanner
from f110_planning.render_callbacks import (
    create_camera_tracking,
    create_heading_error_renderer,
    create_trace_renderer,
    create_waypoint_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import add_common_sim_args, load_waypoints, setup_env

# Predefined color palette for agent traces
COLOR_PALETTE = {
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "orange": (255, 165, 0),
}
TRACE_COLORS = ["yellow", "cyan", "magenta", "red", "green", "blue"]


def parse_args():
    """
    Registers command-line arguments for path-tracking simulation experiments.
    """
    parser = argparse.ArgumentParser(description="Multi-Agent Pure Pursuit Simulation")

    add_common_sim_args(parser, multi_waypoint=True)

    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=True,
        help="Enables manual override (WASD keys) for the primary agent.",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_false",
        dest="hybrid",
        help="Disables manual override; agent proceeds fully autonomously.",
    )

    return parser.parse_args()


def _init_planners(
    num_agents: int, agent_waypoints: list[np.ndarray], enable_hybrid: bool
) -> list[Any]:
    """Initializes a list of planners for all agents."""
    planners = []
    for i in range(num_agents):
        auto_planner = PurePursuitPlanner(waypoints=agent_waypoints[i])

        if i == 0 and enable_hybrid:
            manual_planner = ManualPlanner()
            planner = HybridPlanner(manual_planner, auto_planner)
        else:
            planner = auto_planner

        planners.append(planner)
    return planners


def _setup_rendering(
    env: Any, num_agents: int, agent_waypoints: list[np.ndarray]
) -> None:
    """Configures multi-agent rendering callbacks."""
    env.unwrapped.add_render_callback(create_camera_tracking(rotate=True))
    env.unwrapped.add_render_callback(render_lidar)
    env.unwrapped.add_render_callback(render_side_distances)

    for i in range(num_agents):
        color_name = TRACE_COLORS[i % len(TRACE_COLORS)]
        color_rgb = COLOR_PALETTE[color_name]

        # Draw the static reference path
        env.unwrapped.add_render_callback(create_waypoint_renderer(agent_waypoints[i]))
        # Trace the actual driven path
        env.unwrapped.add_render_callback(
            create_trace_renderer(agent_idx=i, color=color_rgb)
        )
        # Visualize the heading deviation
        env.unwrapped.add_render_callback(
            create_heading_error_renderer(agent_waypoints[i], i)
        )


def _run_tracking_sim(
    env: Any, obs: dict, planners: list, num_agents: int, r_mode: str | None
) -> np.ndarray:
    """Executes the multi-agent tracking loop."""
    laptimes = np.zeros(num_agents)
    done = False

    try:
        while not done:
            actions = []
            for i in range(num_agents):
                action = planners[i].plan(obs, ego_idx=i)
                actions.append([action.steer, action.speed])

            obs, rewards, term, trunc, _ = env.step(np.array(actions))

            if isinstance(term, np.ndarray):
                done = term.any() or trunc.any()
                laptimes += rewards
            else:
                done = term or trunc
                laptimes[0] += rewards

            if r_mode:
                env.render()
    except KeyboardInterrupt:
        print("\nSimulation halted.")

    return laptimes


def _print_results(num_agents: int, laptimes: np.ndarray, total_time: float) -> None:
    """Displays simulation performance and laptime results."""
    print("\n--- Tracking Results ---")
    print(f"Agents:            {num_agents}")
    for i in range(num_agents):
        print(f"  Agent {i} Laptime: {laptimes[i]:.2f}s")

    if total_time > 0:
        print(f"RT-Factor (Avg):    {laptimes[0] / total_time:.2f}x")


def main() -> None:
    """
    Entry point for running multi-agent waypoint tracking simulation.
    """
    args = parse_args()
    r_mode = None if args.render_mode == "None" else args.render_mode
    agent_waypoints = [load_waypoints(w) for w in args.waypoints]
    num_agents = len(agent_waypoints)

    env = setup_env(args, r_mode)

    # Initial reset and rendering setup
    poses = np.zeros((num_agents, 3))
    for i in range(num_agents):
        poses[i] = [args.start_x, args.start_y + (i * 0.4), args.start_theta]

    obs, _ = env.reset(options={"poses": poses})
    if r_mode:
        _setup_rendering(env, num_agents, agent_waypoints)
        env.render()

    planners = _init_planners(
        num_agents, agent_waypoints, args.hybrid and r_mode is not None
    )

    print(f"Starting {num_agents}-agent tracking simulation...")
    t_start = time.time()
    laptimes = _run_tracking_sim(env, obs, planners, num_agents, r_mode)

    _print_results(num_agents, laptimes, time.time() - t_start)
    env.close()


if __name__ == "__main__":
    main()
