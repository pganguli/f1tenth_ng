#!/usr/bin/env python3
"""
Simulation script to test various reactive planners.
Supports: bubble, gap_follower, disparity, and dynamic.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from f110_planning.metrics import MetricAggregator
from f110_planning.reactive import (
    BubblePlanner,
    DisparityExtenderPlanner,
    DynamicWaypointPlanner,
    EdgeCloudPlanner,
    GapFollowerPlanner,
    LidarDNNPlanner,
)
from f110_planning.render_callbacks import (
    create_camera_tracking,
    create_dynamic_waypoint_renderer,
    create_heading_error_renderer,
    create_trace_renderer,
    create_cloud_call_renderer,
    create_waypoint_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.base import CloudScheduler
from f110_planning.schedulers import FixedIntervalScheduler
from f110_planning.utils import add_common_sim_args, load_waypoints, setup_env
from stable_baselines3 import PPO


# ------------------------------------------------------------------
# Cloud scheduler wrapper for policies saved by train_rl.py
# ------------------------------------------------------------------
class PolicyScheduler(CloudScheduler):
    """``CloudScheduler`` backed by an SB3 PPO policy"""

    def __init__(self, policy_path: str) -> None:
        self._model = PPO.load(policy_path)

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Any | None,
    ) -> bool:
        # the policy was trained on observations produced by the RL wrapper,
        # which include a few extra keys.  When running normal simulations the
        # raw env obs lack those fields, causing SB3 to raise a KeyError.  We
        # therefore create a shallow copy and supply reasonable defaults so the
        # policy can always run.
        obs_rl = {**obs}
        obs_rl.setdefault("cloud_request_pending", 0)
        obs_rl.setdefault("latest_cloud_action", np.array([0.0, 0.0]))
        obs_rl.setdefault("crosstrack_dist", np.array([0.0]))
        action, _ = self._model.predict(obs_rl, deterministic=True)
        return bool(action)

    def reset(self) -> None:  # type: ignore[override]
        """No persistent state to reset for a stateless policy."""


def _build_reactive_parser() -> argparse.ArgumentParser:
    """
    Build the reactive-planner argument parser without the ``--help`` action.

    Returns a parser with ``add_help=False`` so it can be reused as the
    ``parents`` entry for scripts that extend these arguments (e.g.
    ``tune_alphas``).  The arguments and defaults are exactly the same as in
    :func:`parse_args`.
    """
    parser = argparse.ArgumentParser(
        description="F1TENTH Reactive Planner Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    add_common_sim_args(parser)

    parser.add_argument(
        "--planner",
        type=str,
        choices=["bubble", "gap", "disparity", "dynamic", "dnn", "edge_cloud"],
        default="edge_cloud",
        help="Algorithm for obstacle avoidance and navigation.",
    )
    # scheduling strategy controls cloud calling logic when using edge_cloud
    parser.add_argument(
        "--cloud-strategy",
        type=str,
        choices=["always", "interval", "rl"],
        default="rl",
        help=(
            "Cloud calling strategy: 'always' issues a request every step, "
            "'interval' uses --cloud-interval spacing, and 'rl' loads a "
            "policy specified by --rl-scheduler."
        ),
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
        "--camera-tracking",
        dest="camera_tracking",
        action="store_true",
        default=False,
        help="Enable camera tracking of the car during rendering.",
    )

    parser.add_argument(
        "--left-wall-model",
        type=str,
        default="data/models/left_wall_dist_arch6.pt",
        help="Path to self-sufficient TorchScript .pt model for left wall distance.",
    )

    parser.add_argument(
        "--track-width-model",
        type=str,
        default="data/models/track_width_arch6.pt",
        help="Path to self-sufficient TorchScript .pt model for total track width.",
    )

    parser.add_argument(
        "--heading-model",
        type=str,
        default="data/models/heading_error_arch6.pt",
        help="Path to the self-sufficient TorchScript .pt heading-error model.",
    )

    # ---- edge-cloud specific ----
    ec = parser.add_argument_group(
        "edge-cloud", "Edge-Cloud DNN settings and RL scheduler"
    )
    ec.add_argument(
        "--cloud-latency",
        type=int,
        default=10,
        help="Round-trip latency in simulation steps for cloud inference.",
    )
    ec.add_argument(
        "--alpha-steer",
        type=float,
        default=0.1,
        help="Cloud weight for steering (0 = edge only, 1 = cloud only).",
    )
    ec.add_argument(
        "--alpha-speed",
        type=float,
        default=0.1,
        help="Cloud weight for speed (0 = edge only, 1 = cloud only).",
    )
    ec.add_argument(
        "--cloud-interval",
        type=int,
        default=1,
        help="Steps between cloud requests (1 = every step).",
    )
    ec.add_argument(
        "--rl-scheduler",
        type=str,
        default="data/models/cloud_scheduler.zip",
        help=(
            "Path to a PPO policy (.zip) for cloud scheduling.  If the file "
            "exists the policy will be used; otherwise the fixed-interval "
            "scheduler (--cloud-interval) is employed."
        ),
    )
    ec.add_argument(
        "--edge-left-wall-model",
        type=str,
        default="data/models/left_wall_dist_arch1.pt",
        help="Path to self-sufficient TorchScript .pt edge left wall distance model.",
    )
    ec.add_argument(
        "--edge-track-width-model",
        type=str,
        default="data/models/track_width_arch1.pt",
        help="Path to self-sufficient TorchScript .pt edge track width model.",
    )
    ec.add_argument(
        "--edge-heading-model",
        type=str,
        default="data/models/heading_error_arch1.pt",
        help="Path to self-sufficient TorchScript .pt edge heading-error model.",
    )
    ec.add_argument(
        "--cloud-left-wall-model",
        type=str,
        default="data/models/left_wall_dist_arch6.pt",
        help="Path to self-sufficient TorchScript .pt cloud left wall distance model.",
    )
    ec.add_argument(
        "--cloud-track-width-model",
        type=str,
        default="data/models/track_width_arch6.pt",
        help="Path to self-sufficient TorchScript .pt cloud track width model.",
    )
    ec.add_argument(
        "--cloud-heading-model",
        type=str,
        default="data/models/heading_error_arch6.pt",
        help="Path to self-sufficient TorchScript .pt cloud heading-error model.",
    )

    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Registers command-line arguments for reactive simulation experiments.

    Parameters
    ----------
    args : list[str] | None
        If provided, the list is passed to ``ArgumentParser.parse_args``.
        This is mainly used by tests to avoid interference with pytest's own
        arguments.
    """
    parser = argparse.ArgumentParser(
        parents=[_build_reactive_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args(args)


def _create_planner(args: argparse.Namespace, waypoints: np.ndarray) -> Any:  # pylint: disable=too-many-branches
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
            "left_model_path": args.left_wall_model,
            "track_width_model_path": args.track_width_model,
            "heading_model_path": args.heading_model,
            "lookahead_distance": args.lookahead,
            "lateral_gain": args.lateral_gain,
        }
        if args.speed is not None:
            kwargs["max_speed"] = args.speed
        return LidarDNNPlanner(**kwargs)

    if args.planner == "edge_cloud":
        # choose scheduler based on strategy
        if args.cloud_strategy == "always":
            scheduler = FixedIntervalScheduler(interval=1)
        elif args.cloud_strategy == "interval":
            scheduler = FixedIntervalScheduler(interval=args.cloud_interval)
        else:  # rl
            if args.rl_scheduler is not None and Path(args.rl_scheduler).exists():
                print(f"Using RL scheduler at {args.rl_scheduler}")
                scheduler = PolicyScheduler(args.rl_scheduler)
            else:
                # fall back to default interval behavior when policy missing
                scheduler = FixedIntervalScheduler(interval=args.cloud_interval)

        kwargs = {
            "cloud_latency": args.cloud_latency,
            "scheduler": scheduler,
            "alpha_steer": args.alpha_steer,
            "alpha_speed": args.alpha_speed,
            "lookahead_distance": args.lookahead,
            "lateral_gain": args.lateral_gain,
            "edge_left_wall_model_path": args.edge_left_wall_model,
            "edge_track_width_model_path": args.edge_track_width_model,
            "edge_heading_model_path": args.edge_heading_model,
            "cloud_left_wall_model_path": args.cloud_left_wall_model,
            "cloud_track_width_model_path": args.cloud_track_width_model,
            "cloud_heading_model_path": args.cloud_heading_model,
        }
        if args.speed is not None:
            kwargs["max_speed"] = args.speed
        return EdgeCloudPlanner(**kwargs)

    raise ValueError(f"Unsupported planner logic: {args.planner}")


def _setup_rendering(
    env: Any, args: argparse.Namespace, waypoints: np.ndarray, planner: Any
) -> None:
    """Configures environment render callbacks."""
    if args.camera_tracking:
        env.unwrapped.add_render_callback(create_camera_tracking(rotate=True))
    env.unwrapped.add_render_callback(render_lidar)
    env.unwrapped.add_render_callback(render_side_distances)
    env.unwrapped.add_render_callback(create_trace_renderer(agent_idx=0))

    if waypoints.size > 0:
        env.unwrapped.add_render_callback(create_waypoint_renderer(waypoints))
        env.unwrapped.add_render_callback(create_heading_error_renderer(waypoints, 0))

    if args.planner in ["dynamic", "dnn", "edge_cloud"]:
        env.unwrapped.add_render_callback(
            create_dynamic_waypoint_renderer(planner, agent_idx=0)
        )
    # when using the edge-cloud planner we can also visualize the cloud calls
    if args.planner == "edge_cloud":
        env.unwrapped.add_render_callback(
            create_cloud_call_renderer(planner, agent_idx=0)
        )


def _run_reactive_sim(
    env: Any,
    obs: dict[str, Any],
    planner: Any,
    r_mode: str | None,
    waypoints: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Executes the reactive simulation loop with metric collection."""
    wpts = waypoints if waypoints.size > 0 else None
    metrics = MetricAggregator.create_default(waypoints=wpts)
    metrics.on_reset(obs, waypoints=wpts)

    laptime, done = 0.0, False

    try:
        while not done:
            action = planner.plan(obs, ego_idx=0)
            obs, reward, terminated, truncated, _ = env.step(
                np.array([[action.steer, action.speed]])
            )
            done, laptime = (terminated or truncated), laptime + float(reward)
            metrics.on_step(obs, action, float(reward), ego_idx=0)

            if r_mode:
                env.render()
    except KeyboardInterrupt:
        print("\nSimulation aborted by user.")
    except RuntimeError as exc:
        print(f"\nSimulation stopped: {exc}")

    return laptime, metrics.report()


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
    start_time = time.time()
    laptime, _ = _run_reactive_sim(env, obs, planner, r_mode, waypoints)

    total_real_time = time.time() - start_time
    print("\n--- Simulation Summary ---")
    print(f"Planner:           {args.planner}")
    print(f"Real Wall Time:    {total_real_time:.3f}s")
    if total_real_time > 0:
        print(f"RT-Factor:         {laptime / total_real_time:.2f}x")

    env.close()


if __name__ == "__main__":
    main()
