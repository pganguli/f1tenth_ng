#!/usr/bin/env python3
"""Coarse-to-fine alpha sweep for Edge-Cloud planner."""
# pylint: disable=wrong-import-position

import argparse
import copy
import sys
from pathlib import Path

# import gym to ensure environment registration side effects
import gymnasium  # noqa: F401 pylint: disable=unused-import
import numpy as np

# make sibling packages importable before pulling in local modules
sys.path.append(str(Path(__file__).resolve().parent.parent / "sim"))
sys.path.append(str(Path(__file__).resolve().parent))

# the following modules reside in sibling directories added above; pylint
# can't resolve them statically.
from reactive_planners import (  # pylint: disable=import-error
    _build_reactive_parser,
    _create_planner,
    _run_reactive_sim,
)
from tune_utils import coarse_to_fine_search

# import first-party modules after third-party/local ones to satisfy
# pylint's import-order checks
import f110_gym  # noqa: F401 pylint: disable=unused-import
from f110_planning.utils import load_waypoints, resolve_start_pose, setup_env


def _parse_args() -> argparse.Namespace:
    """
    Build and parse all arguments for the alpha-tuning sweep.

    Combines the full reactive-planner argument set with tuning-specific
    options in a single ``--help``-visible parser.  Rendering defaults to
    ``None`` (disabled) so the sweep can run headlessly; pass
    ``--render-mode human`` to override.
    """
    parent = _build_reactive_parser()

    parser = argparse.ArgumentParser(
        parents=[parent],
        description=(
            "Coarse-to-fine alpha sweep for the Edge-Cloud planner.\n\n"
            "Performs a two-phase grid search over alpha_steer / alpha_speed "
            "to minimise cross-track RMSE while avoiding collisions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Default to headless, single-lap for sweep
    parser.set_defaults(render_mode="None", max_laps=2, cloud_strategy="always")

    tune = parser.add_argument_group("tuning", "Coarse-to-fine search configuration")
    tune.add_argument(
        "--coarse-points",
        type=int,
        default=4,
        metavar="N",
        help="Grid dimension for the coarse search phase (N×N evaluations).",
    )
    tune.add_argument(
        "--fine-points",
        type=int,
        default=3,
        metavar="N",
        help="Grid dimension for the fine search phase (N×N evaluations).",
    )
    tune.add_argument(
        "--alpha-steer-min",
        type=float,
        default=0.1,
        help="Lower bound of the alpha_steer search range.",
    )
    tune.add_argument(
        "--alpha-steer-max",
        type=float,
        default=0.9,
        help="Upper bound of the alpha_steer search range.",
    )
    tune.add_argument(
        "--alpha-speed-min",
        type=float,
        default=0.1,
        help="Lower bound of the alpha_speed search range.",
    )
    tune.add_argument(
        "--alpha-speed-max",
        type=float,
        default=0.9,
        help="Upper bound of the alpha_speed search range.",
    )

    parallel_group = tune.add_mutually_exclusive_group()
    parallel_group.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        default=True,
        help=(
            "Enable parallel evaluation across alpha grid points "
            "(automatically disabled when rendering is active)."
        ),
    )
    parallel_group.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel evaluation; run grid points sequentially.",
    )

    return parser.parse_args()


def evaluate_runner(args, waypoints, start_pose, render_override=None):
    """Factory producing a single-(steer,speed) evaluator."""

    def _eval(alpha_steer: float, alpha_speed: float) -> tuple:
        eval_args = copy.deepcopy(args)
        # Map 2-D search axes to the 3 feature-level alpha parameters:
        # alpha_steer drives lateral features (left_wall, heading)
        # alpha_speed drives the track-width feature
        eval_args.alpha_left = alpha_steer
        eval_args.alpha_heading = alpha_steer
        eval_args.alpha_track = alpha_speed
        eval_args.sigma_proc_left = None
        eval_args.sigma_proc_track = None
        eval_args.sigma_proc_heading = None
        planner = _create_planner(eval_args, waypoints)

        env_local = setup_env(args, render_override)
        obs, _ = env_local.reset(options={"poses": start_pose})
        laptime, metrics_dict = _run_reactive_sim(
            env_local, obs, planner, r_mode=render_override, waypoints=waypoints
        )
        env_local.close()

        collisions = metrics_dict.get("collision", 0.0)
        if collisions > 0:
            return (1000.0 - laptime, False)
        return (metrics_dict.get("crosstrack_rmse_m", 999.0), True)

    return _eval


def main() -> None:
    """Execute the tuning sweep based on CLI arguments."""
    # pylint: disable=too-many-locals
    args = _parse_args()

    if args.planner != "edge_cloud":
        print("Overriding planner to edge_cloud for tuning.")
        args.planner = "edge_cloud"

    waypoints = load_waypoints(args.waypoints)
    start_pose = np.array([list(resolve_start_pose(args))])

    render_override = None if args.render_mode == "None" else args.render_mode
    print("rendering:", render_override or "off")

    eval_fn_raw = evaluate_runner(args, waypoints, start_pose, render_override)

    # Parallel evaluation is only safe when there is no active render window.
    actual_parallel = args.parallel and (render_override is None)
    print("parallel:", "on" if actual_parallel else "off")

    coarse_grid_size = args.coarse_points
    fine_grid_size = args.fine_points
    alpha_steer_min = args.alpha_steer_min
    alpha_steer_max = args.alpha_steer_max
    alpha_speed_min = args.alpha_speed_min
    alpha_speed_max = args.alpha_speed_max

    print(
        "\nCoarse-to-fine search",
        f"map={args.map} latency={args.cloud_latency}/{args.cloud_interval}",
        f"grid {coarse_grid_size}x{coarse_grid_size} -> {fine_grid_size}x{fine_grid_size}",
    )
    print("-" * 50)

    # Run coarse_to_fine_search, giving it the raw evaluator directly so that
    # the crash_free boolean returned by evaluate_runner is observed.  enable
    # parallel evaluation only when rendering is disabled (otherwise multiple
    # pyglet windows would compete).
    best_steer, best_speed, best_score, best_crash_free = coarse_to_fine_search(
        eval_fn_raw,
        coarse_grid_size=coarse_grid_size,
        fine_grid_size=fine_grid_size,
        verbose=True,
        parallel=actual_parallel,
        steer_min=alpha_steer_min,
        steer_max=alpha_steer_max,
        speed_min=alpha_speed_min,
        speed_max=alpha_speed_max,
    )

    if not best_crash_free:
        print("warning: no crash-free runs")
        sys.exit(1)

    # report optimal configuration
    print("\nOptimal configuration")
    print(f"  alpha_left/heading = {best_steer:.3f}")
    print(f"  alpha_track        = {best_speed:.3f}")
    print(f"  score       = {best_score:.3f}")
    print(f"  crash_free  = {best_crash_free}")


if __name__ == "__main__":
    main()
