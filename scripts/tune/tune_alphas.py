#!/usr/bin/env python3
"""Coarse-to-fine alpha sweep for Edge-Cloud planner."""
# pylint: disable=wrong-import-position

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
    _create_planner,
    _run_reactive_sim,
    parse_args,
)
from tune_utils import coarse_to_fine_search

# import first-party modules after third-party/local ones to satisfy
# pylint's import-order checks
import f110_gym  # noqa: F401 pylint: disable=unused-import
from f110_planning.utils import load_waypoints, setup_env


def evaluate_runner(args, waypoints, start_pose, render_override=None):
    """Factory producing a single-(steer,speed) evaluator."""

    def _eval(alpha_steer: float, alpha_speed: float) -> tuple:
        eval_args = copy.deepcopy(args)
        eval_args.alpha_steer = alpha_steer
        eval_args.alpha_speed = alpha_speed
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
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    coarse_grid_size = 3
    fine_grid_size = 3
    alpha_steer_min = 0.0
    alpha_steer_max = 1.0
    alpha_speed_min = 0.0
    alpha_speed_max = 1.0
    parallel_enabled = True

    # grab a few extra flags before letting parse_args see the rest
    render_given = any(arg.startswith("--render-mode") for arg in sys.argv[1:])
    argv_clean = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--coarse-points":
            coarse_grid_size = int(sys.argv[i + 1])
            i += 2
        elif arg == "--fine-points":
            fine_grid_size = int(sys.argv[i + 1])
            i += 2
        elif arg == "--alpha-steer-min":
            alpha_steer_min = float(sys.argv[i + 1])
            i += 2
        elif arg == "--alpha-steer-max":
            alpha_steer_max = float(sys.argv[i + 1])
            i += 2
        elif arg == "--alpha-speed-min":
            alpha_speed_min = float(sys.argv[i + 1])
            i += 2
        elif arg == "--alpha-speed-max":
            alpha_speed_max = float(sys.argv[i + 1])
            i += 2
        elif arg == "--no-parallel":
            parallel_enabled = False
            i += 1
        elif arg == "--parallel":
            parallel_enabled = True
            i += 1
        else:
            argv_clean.append(arg)
            i += 1

    sys.argv = argv_clean
    args = parse_args()
    if not render_given:
        args.render_mode = "None"
    if args.planner != "edge_cloud":
        print("Overriding planner to edge_cloud for tuning.")
        args.planner = "edge_cloud"

    waypoints = load_waypoints(args.waypoints)
    start_pose = np.array([[args.start_x, args.start_y, args.start_theta]])

    render_override = None if args.render_mode == "None" else args.render_mode
    print("rendering:", render_override or "off")

    eval_fn_raw = evaluate_runner(args, waypoints, start_pose, render_override)

    # show effective parallel setting
    actual_parallel = parallel_enabled and (render_override is None)
    print("parallel:", "on" if actual_parallel else "off")

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

    any_crash_free = False
    for alpha_steer in np.linspace(alpha_steer_min, alpha_steer_max, coarse_grid_size):
        for alpha_speed in np.linspace(
            alpha_speed_min, alpha_speed_max, coarse_grid_size
        ):
            _, crash_free = eval_fn_raw(alpha_steer, alpha_speed)
            if crash_free:
                any_crash_free = True
                break
        if any_crash_free:
            break

    if not any_crash_free:
        print("warning: no crash-free runs")
        sys.exit(1)

    # report optimal configuration
    print("\nOptimal configuration")
    print(f"  alpha_steer = {best_steer:.3f}")
    print(f"  alpha_speed = {best_speed:.3f}")
    print(f"  score       = {best_score:.3f}")
    print(f"  crash_free  = {best_crash_free}")


if __name__ == "__main__":
    main()
