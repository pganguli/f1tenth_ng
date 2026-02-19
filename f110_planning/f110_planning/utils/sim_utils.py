"""
Shared utilities and constants for F1TENTH simulation and data generation scripts.
"""

import argparse
from typing import Any, Optional

# Default map and waypoint configuration
DEFAULT_MAP = "data/maps/F1/Oschersleben/Oschersleben_map"
DEFAULT_MAP_EXT = ".png"
DEFAULT_WAYPOINTS = "data/maps/F1/Oschersleben/Oschersleben_centerline.tsv"

# Default vehicle starting pose
DEFAULT_START_X = 0.0
DEFAULT_START_Y = 0.0
DEFAULT_START_THETA = 2.85

# Default rendering settings
DEFAULT_RENDER_MODE = "human_fast"
DEFAULT_RENDER_FPS = 60


def add_common_sim_args(
    parser: argparse.ArgumentParser, multi_waypoint: bool = False
) -> None:
    """
    Registers standard simulation CLI arguments to an ArgumentParser.

    Args:
        parser: The ArgumentParser instance to populate.
        multi_waypoint: If True, allows multiple waypoint files to be passed,
            which setup_env will interpret as multiple agents.
    """
    parser.add_argument(
        "--map",
        type=str,
        default=DEFAULT_MAP,
        help="Path to the map YAML file (without extension).",
    )
    parser.add_argument(
        "--map-ext",
        type=str,
        default=DEFAULT_MAP_EXT,
        help="The image extension used by the map (e.g., .png, .pgm).",
    )

    if multi_waypoint:
        parser.add_argument(
            "--waypoints",
            type=str,
            nargs="+",
            default=[DEFAULT_WAYPOINTS],
            help="Whitespace-separated list of waypoint filenames. One agent is created per file.",
        )
    else:
        parser.add_argument(
            "--waypoints",
            type=str,
            default=DEFAULT_WAYPOINTS,
            help="Path to the .csv or .tsv waypoint file for the agent to follow.",
        )

    parser.add_argument(
        "--start-x",
        type=float,
        default=DEFAULT_START_X,
        help="Initial X-coordinate for the vehicle in the map frame.",
    )
    parser.add_argument(
        "--start-y",
        type=float,
        default=DEFAULT_START_Y,
        help="Initial Y-coordinate for the vehicle in the map frame.",
    )
    parser.add_argument(
        "--start-theta",
        type=float,
        default=DEFAULT_START_THETA,
        help="Initial orientation (yaw) of the vehicle in radians.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["human", "human_fast", "None"],
        default=DEFAULT_RENDER_MODE,
        help="Visualization mode. 'human_fast' omits some overlays for performance.",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=DEFAULT_RENDER_FPS,
        help="Target frames per second for rendering.",
    )


def setup_env(args: argparse.Namespace, render_mode: Optional[str] = None) -> Any:
    """
    Initializes a standard F1TENTH Gym environment based on CLI arguments.

    Args:
        args: Parsed arguments containing map, waypoints, and render settings.
        render_mode: Overrides the render mode in args if provided.

    Returns:
        The initialized F1TENTH Gym environment.
    """
    import gymnasium as gym  # pylint: disable=import-outside-toplevel
    from f110_gym.envs.base_classes import (
        Integrator,  # pylint: disable=import-outside-toplevel
    )

    num_agents = getattr(args, "num_agents", 1)
    if hasattr(args, "waypoints") and isinstance(args.waypoints, list):
        num_agents = len(args.waypoints)

    render_fps = getattr(args, "render_fps", 60)
    actual_render_mode = render_mode if render_mode is not None else args.render_mode
    if actual_render_mode == "None":
        actual_render_mode = None

    env = gym.make(
        "f110_gym:f110-v0",
        map=args.map,
        map_ext=args.map_ext,
        num_agents=num_agents,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode=actual_render_mode,
        render_fps=render_fps,
        max_laps=None,
    )
    return env
