"""
Shared utilities and constants for F1TENTH simulation and data generation scripts.
"""

import argparse
from typing import Any, Optional, Union

import gymnasium as gym
from f110_gym.envs.base_classes import Integrator

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
        default=None,
        help=(
            "Initial orientation (yaw) of the vehicle in radians. "
            "Defaults to the value stored in the map YAML (start_pose[2]), "
            f"or {DEFAULT_START_THETA} if the YAML has no such key."
        ),
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["human", "human_fast", "None"],
        default=DEFAULT_RENDER_MODE,
        help="Visualization mode. 'human_fast' omits some overlays for performance.",
    )

    parser.add_argument(
        "--max-laps",
        type=int,
        default=None,
        help=(
            "Maximum number of laps before the environment terminates. "
            "Pass 0 or omit for no lap limit (default)."
        ),
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
        max_laps=getattr(args, "max_laps", None),
    )
    return env


def load_start_pose_from_yaml(
    map_path: str,
) -> Optional[tuple[float, float, float]]:
    """
    Read ``start_pose`` from a map YAML file.

    The YAML key is expected to be a 3-element list ``[x, y, theta]``,
    matching the ``origin`` convention used by the map format::

        start_pose: [0.0, 0.0, 2.857332]

    Args:
        map_path: Path to the map without the .yaml extension
                  (matches the --map CLI argument).

    Returns:
        ``(start_x, start_y, start_theta)`` if the key is present and valid,
        otherwise ``None``.
    """
    import yaml  # local import to avoid a hard top-level dependency

    yaml_path = map_path + ".yaml"
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        pose = data.get("start_pose")
        if isinstance(pose, (list, tuple)) and len(pose) == 3:
            return float(pose[0]), float(pose[1]), float(pose[2])
    except (OSError, TypeError, AttributeError):
        pass
    return None


def resolve_start_pose(
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    """
    Return the starting pose for the vehicle.

    Priority (highest first):

    1. Explicit ``--start-theta`` CLI value (non-None).
    2. ``start_x`` / ``start_y`` / ``start_theta`` keys in the map YAML.
    3. Module-level defaults (``DEFAULT_START_X/Y/THETA``).

    ``--start-x`` and ``--start-y`` always fall back to their defaults when
    not supplied; theta is the only value sourced from the YAML because
    x/y are universally 0.0 across all current maps.

    Args:
        args: Parsed arguments from :func:`add_common_sim_args`.

    Returns:
        ``(start_x, start_y, start_theta)`` as floats.
    """
    sx = args.start_x if args.start_x is not None else DEFAULT_START_X
    sy = args.start_y if args.start_y is not None else DEFAULT_START_Y
    st = args.start_theta  # None when the user did not pass --start-theta

    if st is None:
        yaml_pose = load_start_pose_from_yaml(args.map)
        if yaml_pose is not None:
            sx, sy, st = yaml_pose
        else:
            st = DEFAULT_START_THETA

    return float(sx), float(sy), float(st)
