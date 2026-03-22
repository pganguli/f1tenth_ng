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

try:
    from stable_baselines3 import PPO
except ModuleNotFoundError:
    PPO = None

from f110_planning.metrics import MetricAggregator
from f110_planning.reactive import (
    BubblePlanner,
    DisparityExtenderPlanner,
    DynamicWaypointPlanner,
    EdgeCloudPlanner,
    GapFollowerPlanner,
    LidarDNNPlanner,
    MultiTierEdgeCloudPlanner,
)
from f110_planning.render_callbacks import (
    create_camera_tracking,
    create_cloud_call_renderer,
    create_dynamic_waypoint_renderer,
    create_heading_error_renderer,
    create_trace_renderer,
    create_waypoint_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.base import CloudScheduler
from f110_planning.schedulers import (
    AlwaysCloudScheduler,
    BernoulliMaxMissConfig,
    BernoulliMaxMissPolicy,
    FixedIntervalScheduler,
    FixedBernoulliConfig,
    FixedBernoulliPolicy,
    PolicyDrivenScheduler,
    DeterministicDeviationConfig,
    DeterministicDeviationPolicy,
    LogisticRiskConfig,
    LogisticRiskPolicy,
    ExponentialRiskConfig,
    ExponentialRiskPolicy,
    PiecewiseLinearRampConfig,
    PiecewiseLinearRampPolicy,
    TieredPolicyDrivenScheduler,
    TieredProbabilisticRiskConfig,
    TieredProbabilisticRiskPolicy,
    TieredThresholdConfig,
    TieredThresholdPolicy,
)
from f110_planning.utils import (
    add_common_sim_args,
    load_waypoints,
    resolve_start_pose,
    setup_env,
)


# ------------------------------------------------------------------
# Cloud scheduler wrapper for policies saved by train_rl.py
# ------------------------------------------------------------------
class PolicyScheduler(CloudScheduler):
    """``CloudScheduler`` backed by an SB3 PPO policy"""

    def __init__(self, policy_path: str) -> None:
        if PPO is None:
            raise ModuleNotFoundError(
                "stable_baselines3 is required for --cloud-strategy rl"
            )
        self._model = PPO.load(policy_path)

    def should_call_cloud(
        self,
        step: int,
        obs: dict[str, Any],
        latest_cloud_action: Any | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        del step, latest_cloud_action, context
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
        choices=[
            "bubble",
            "gap",
            "disparity",
            "dynamic",
            "dnn",
            "edge_cloud",
            "multi_tier_edge_cloud",
        ],
        default="edge_cloud",
        help="Algorithm for obstacle avoidance and navigation.",
    )
    # scheduling strategy controls cloud calling logic when using edge_cloud
    parser.add_argument(
        "--cloud-strategy",
        type=str,
        choices=[
            "always",
            "hybrid",
            "interval",
            "rl",
            "deterministic",
            "fixed_bernoulli",
            "bernoulli_max_miss",
            "logistic",
            "exponential",
            "piecewise_ramp",
            "tiered_probabilistic",
            "tiered_threshold",
        ],
        default="rl",
        help=(
            "Cloud calling strategy: 'always' issues a request every step, "
            "'hybrid' also requests cloud every step, "
            "'interval' uses --cloud-interval spacing, and 'rl' loads a "
            "policy specified by --rl-scheduler. Safety strategies include "
            "'deterministic', 'fixed_bernoulli', 'bernoulli_max_miss', "
            "'logistic', 'exponential', and 'piecewise_ramp'. "
            "Multi-tier strategies include "
            "'tiered_probabilistic' and 'tiered_threshold'."
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
        default=0.5,
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
        "--safety-threshold",
        type=float,
        default=0.03,
        help="Deviation threshold for deterministic strategy.",
    )
    ec.add_argument(
        "--safety-min-interval",
        type=int,
        default=1,
        help="Minimum steps between cloud invocations for safety strategies.",
    )
    ec.add_argument(
        "--safety-warmup-steps",
        type=int,
        default=1,
        help="Always request cloud for first N steps in safety strategies.",
    )
    ec.add_argument(
        "--safety-fallback-interval",
        type=int,
        default=0,
        help="Force periodic cloud refresh every N steps when > 0.",
    )
    ec.add_argument(
        "--safety-max-miss",
        type=int,
        default=5,
        help="Maximum consecutive eligible misses for safety strategies.",
    )
    ec.add_argument(
        "--strategy-seed",
        type=int,
        default=7,
        help="Random seed for probabilistic strategies.",
    )
    ec.add_argument(
        "--fixed-bernoulli-p",
        type=float,
        default=0.6,
        help="Cloud-call probability for fixed Bernoulli scheduling.",
    )
    ec.add_argument(
        "--bernoulli-max-miss-p",
        type=float,
        default=0.6,
        help="Base cloud-call probability for Bernoulli+max-miss scheduling.",
    )
    ec.add_argument(
        "--logistic-center",
        type=float,
        default=0.03,
        help="Deviation center for logistic strategy.",
    )
    ec.add_argument(
        "--logistic-slope",
        type=float,
        default=40.0,
        help="Sigmoid slope for logistic strategy.",
    )
    ec.add_argument(
        "--logistic-p-min",
        type=float,
        default=0.0,
        help="Minimum cloud-call probability for logistic strategy.",
    )
    ec.add_argument(
        "--logistic-p-max",
        type=float,
        default=1.0,
        help="Maximum cloud-call probability for logistic strategy.",
    )
    ec.add_argument(
        "--exp-center",
        type=float,
        default=0.03,
        help="Deviation center for exponential strategy.",
    )
    ec.add_argument(
        "--exp-rate",
        type=float,
        default=25.0,
        help="Growth rate for exponential strategy.",
    )
    ec.add_argument(
        "--exp-p-min",
        type=float,
        default=0.0,
        help="Minimum cloud-call probability for exponential strategy.",
    )
    ec.add_argument(
        "--exp-p-max",
        type=float,
        default=1.0,
        help="Maximum cloud-call probability for exponential strategy.",
    )
    ec.add_argument(
        "--ramp-d-low",
        type=float,
        default=0.01,
        help="Lower deviation threshold for piecewise-ramp scheduling.",
    )
    ec.add_argument(
        "--ramp-d-high",
        type=float,
        default=0.05,
        help="Upper deviation threshold for piecewise-ramp scheduling.",
    )
    ec.add_argument(
        "--deviation-steer-weight",
        type=float,
        default=1.0,
        help="Weight on steering disagreement when forming the deviation signal.",
    )
    ec.add_argument(
        "--deviation-speed-weight",
        type=float,
        default=1.0,
        help="Weight on speed disagreement when forming the deviation signal.",
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
    ec.add_argument(
        "--medium-cloud-latency",
        type=int,
        default=6,
        help="Round-trip latency in simulation steps for medium cloud inference.",
    )
    ec.add_argument(
        "--large-cloud-latency",
        type=int,
        default=10,
        help="Round-trip latency in simulation steps for large cloud inference.",
    )
    ec.add_argument(
        "--alpha-steer-medium",
        type=float,
        default=0.5,
        help="Medium-cloud blend weight for steering.",
    )
    ec.add_argument(
        "--alpha-speed-medium",
        type=float,
        default=0.1,
        help="Medium-cloud blend weight for speed.",
    )
    ec.add_argument(
        "--alpha-steer-large",
        type=float,
        default=0.7,
        help="Large-cloud blend weight for steering.",
    )
    ec.add_argument(
        "--alpha-speed-large",
        type=float,
        default=0.2,
        help="Large-cloud blend weight for speed.",
    )
    ec.add_argument(
        "--tier-medium-threshold",
        type=float,
        default=0.03,
        help="Disagreement threshold above which medium cloud is requested.",
    )
    ec.add_argument(
        "--tier-large-threshold",
        type=float,
        default=0.08,
        help="Disagreement threshold above which large cloud is requested.",
    )
    ec.add_argument(
        "--tier-steer-weight",
        type=float,
        default=1.0,
        help="Weight on steering disagreement for tiered threshold routing.",
    )
    ec.add_argument(
        "--tier-speed-weight",
        type=float,
        default=0.0,
        help="Weight on speed disagreement for tiered threshold routing.",
    )
    ec.add_argument(
        "--tier-medium-prob",
        type=float,
        default=0.3,
        help="Fixed probability of calling the medium cloud on an eligible step.",
    )
    ec.add_argument(
        "--tier-large-prob",
        type=float,
        default=0.2,
        help="Fixed probability of calling the large cloud on an eligible step.",
    )
    ec.add_argument(
        "--edge2-left-wall-model",
        type=str,
        default="data/models/left_wall_dist_arch2.pt",
        help="Path to self-sufficient TorchScript .pt edge left wall distance model.",
    )
    ec.add_argument(
        "--edge2-track-width-model",
        type=str,
        default="data/models/track_width_arch2.pt",
        help="Path to self-sufficient TorchScript .pt edge track width model.",
    )
    ec.add_argument(
        "--edge2-heading-model",
        type=str,
        default="data/models/heading_error_arch2.pt",
        help="Path to self-sufficient TorchScript .pt edge heading-error model.",
    )
    ec.add_argument(
        "--medium-left-wall-model",
        type=str,
        default="data/models/left_wall_dist_arch5.pt",
        help="Path to self-sufficient TorchScript .pt medium cloud left wall distance model.",
    )
    ec.add_argument(
        "--medium-track-width-model",
        type=str,
        default="data/models/track_width_arch5.pt",
        help="Path to self-sufficient TorchScript .pt medium cloud track width model.",
    )
    ec.add_argument(
        "--medium-heading-model",
        type=str,
        default="data/models/heading_error_arch5.pt",
        help="Path to self-sufficient TorchScript .pt medium cloud heading-error model.",
    )
    ec.add_argument(
        "--large-left-wall-model",
        type=str,
        default="data/models/left_wall_dist_arch6.pt",
        help="Path to self-sufficient TorchScript .pt large cloud left wall distance model.",
    )
    ec.add_argument(
        "--large-track-width-model",
        type=str,
        default="data/models/track_width_arch6.pt",
        help="Path to self-sufficient TorchScript .pt large cloud track width model.",
    )
    ec.add_argument(
        "--large-heading-model",
        type=str,
        default="data/models/heading_error_arch6.pt",
        help="Path to self-sufficient TorchScript .pt large cloud heading-error model.",
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


def _create_planner(args: argparse.Namespace, waypoints: np.ndarray) -> Any:  # pylint: disable=too-many-branches,too-many-statements
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
        if args.cloud_strategy in ("always", "hybrid"):
            scheduler = AlwaysCloudScheduler()
        elif args.cloud_strategy == "interval":
            scheduler = FixedIntervalScheduler(interval=args.cloud_interval)
        elif args.cloud_strategy == "rl":
            if args.rl_scheduler is not None and Path(args.rl_scheduler).exists():
                print(f"Using RL scheduler at {args.rl_scheduler}")
                scheduler = PolicyScheduler(args.rl_scheduler)
            else:
                # fall back to default interval behavior when policy missing
                scheduler = FixedIntervalScheduler(interval=args.cloud_interval)
        elif args.cloud_strategy == "deterministic":
            scheduler = PolicyDrivenScheduler(
                DeterministicDeviationPolicy(
                    DeterministicDeviationConfig(
                        threshold=args.safety_threshold,
                        min_interval=args.safety_min_interval,
                        warmup_steps=args.safety_warmup_steps,
                        fallback_interval=args.safety_fallback_interval,
                    )
                )
            )
        elif args.cloud_strategy == "fixed_bernoulli":
            scheduler = PolicyDrivenScheduler(
                FixedBernoulliPolicy(
                    FixedBernoulliConfig(
                        p=getattr(args, "fixed_bernoulli_p", 0.6),
                        min_interval=args.safety_min_interval,
                        warmup_steps=args.safety_warmup_steps,
                        fallback_interval=args.safety_fallback_interval,
                        seed=args.strategy_seed,
                    )
                )
            )
        elif args.cloud_strategy == "bernoulli_max_miss":
            scheduler = PolicyDrivenScheduler(
                BernoulliMaxMissPolicy(
                    BernoulliMaxMissConfig(
                        p=getattr(args, "bernoulli_max_miss_p", 0.6),
                        min_interval=args.safety_min_interval,
                        warmup_steps=args.safety_warmup_steps,
                        fallback_interval=args.safety_fallback_interval,
                        max_miss=args.safety_max_miss,
                        seed=args.strategy_seed,
                    )
                )
            )
        elif args.cloud_strategy == "logistic":
            scheduler = PolicyDrivenScheduler(
                LogisticRiskPolicy(
                    LogisticRiskConfig(
                        center=args.logistic_center,
                        slope=args.logistic_slope,
                        p_min=args.logistic_p_min,
                        p_max=args.logistic_p_max,
                        min_interval=args.safety_min_interval,
                        warmup_steps=args.safety_warmup_steps,
                        fallback_interval=args.safety_fallback_interval,
                        max_miss=args.safety_max_miss,
                        seed=args.strategy_seed,
                    )
                )
            )
        elif args.cloud_strategy == "exponential":
            scheduler = PolicyDrivenScheduler(
                ExponentialRiskPolicy(
                    ExponentialRiskConfig(
                        center=args.exp_center,
                        rate=args.exp_rate,
                        p_min=args.exp_p_min,
                        p_max=args.exp_p_max,
                        min_interval=args.safety_min_interval,
                        warmup_steps=args.safety_warmup_steps,
                        fallback_interval=args.safety_fallback_interval,
                        max_miss=args.safety_max_miss,
                        seed=args.strategy_seed,
                    )
                )
            )
        elif args.cloud_strategy == "piecewise_ramp":
            scheduler = PolicyDrivenScheduler(
                PiecewiseLinearRampPolicy(
                    PiecewiseLinearRampConfig(
                        d_low=getattr(args, "ramp_d_low", 0.01),
                        d_high=getattr(args, "ramp_d_high", 0.05),
                        min_interval=args.safety_min_interval,
                        warmup_steps=args.safety_warmup_steps,
                        fallback_interval=args.safety_fallback_interval,
                        max_miss=args.safety_max_miss,
                        seed=args.strategy_seed,
                    )
                )
            )
        else:
            raise ValueError(
                "edge_cloud requires --cloud-strategy to be one of "
                "'always', 'hybrid', 'interval', 'rl', 'deterministic', "
                "'fixed_bernoulli', 'bernoulli_max_miss', 'logistic', "
                "'exponential', or 'piecewise_ramp'"
            )

        kwargs = {
            "cloud_latency": args.cloud_latency,
            "scheduler": scheduler,
            "alpha_steer": getattr(
                args,
                "alpha_steer",
                getattr(args, "alpha_heading", 0.5),
            ),
            "alpha_speed": getattr(
                args,
                "alpha_speed",
                getattr(args, "alpha_track", getattr(args, "alpha_left", 0.1)),
            ),
            "deviation_steer_weight": getattr(args, "deviation_steer_weight", 1.0),
            "deviation_speed_weight": getattr(args, "deviation_speed_weight", 1.0),
            "lookahead_distance": args.lookahead,
            "lateral_gain": args.lateral_gain,
            "edge_left_wall_model_path": args.edge_left_wall_model,
            "edge_track_width_model_path": args.edge_track_width_model,
            "edge_heading_model_path": args.edge_heading_model,
            "cloud_left_wall_model_path": args.cloud_left_wall_model,
            "cloud_track_width_model_path": args.cloud_track_width_model,
            "cloud_heading_model_path": args.cloud_heading_model,
        }
        for name in ("alpha_left", "alpha_track", "alpha_heading"):
            if hasattr(args, name):
                kwargs[name] = getattr(args, name)
        if args.speed is not None:
            kwargs["max_speed"] = args.speed
        return EdgeCloudPlanner(**kwargs)

    if args.planner == "multi_tier_edge_cloud":
        if args.cloud_strategy == "tiered_probabilistic":
            scheduler = TieredPolicyDrivenScheduler(
                TieredProbabilisticRiskPolicy(
                    TieredProbabilisticRiskConfig(
                        p_medium=args.tier_medium_prob,
                        p_large=args.tier_large_prob,
                        min_interval=args.safety_min_interval,
                        warmup_steps=args.safety_warmup_steps,
                        fallback_interval=args.safety_fallback_interval,
                        max_miss=args.safety_max_miss,
                        seed=args.strategy_seed,
                    )
                )
            )
            oracle_disagreement_routing = False
        elif args.cloud_strategy == "tiered_threshold":
            scheduler = TieredPolicyDrivenScheduler(
                TieredThresholdPolicy(
                    TieredThresholdConfig(
                        medium_threshold=args.tier_medium_threshold,
                        large_threshold=args.tier_large_threshold,
                        min_interval=args.safety_min_interval,
                        warmup_steps=args.safety_warmup_steps,
                        fallback_interval=args.safety_fallback_interval,
                        max_miss=args.safety_max_miss,
                    )
                )
            )
            oracle_disagreement_routing = True
        else:
            raise ValueError(
                "multi_tier_edge_cloud requires --cloud-strategy to be "
                "'tiered_probabilistic' or 'tiered_threshold'"
            )

        kwargs = {
            "scheduler": scheduler,
            "medium_cloud_latency": args.medium_cloud_latency,
            "large_cloud_latency": args.large_cloud_latency,
            "alpha_steer_medium": args.alpha_steer_medium,
            "alpha_speed_medium": args.alpha_speed_medium,
            "alpha_steer_large": args.alpha_steer_large,
            "alpha_speed_large": args.alpha_speed_large,
            "lookahead_distance": args.lookahead,
            "lateral_gain": args.lateral_gain,
            "edge_left_wall_model_path": args.edge2_left_wall_model,
            "edge_track_width_model_path": args.edge2_track_width_model,
            "edge_heading_model_path": args.edge2_heading_model,
            "medium_left_wall_model_path": args.medium_left_wall_model,
            "medium_track_width_model_path": args.medium_track_width_model,
            "medium_heading_model_path": args.medium_heading_model,
            "large_left_wall_model_path": args.large_left_wall_model,
            "large_track_width_model_path": args.large_track_width_model,
            "large_heading_model_path": args.large_heading_model,
            "oracle_disagreement_routing": oracle_disagreement_routing,
            "tier_steer_weight": args.tier_steer_weight,
            "tier_speed_weight": args.tier_speed_weight,
        }
        if args.speed is not None:
            kwargs["max_speed"] = args.speed
        return MultiTierEdgeCloudPlanner(**kwargs)

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

    if args.planner in ["dynamic", "dnn", "edge_cloud", "multi_tier_edge_cloud"]:
        env.unwrapped.add_render_callback(
            create_dynamic_waypoint_renderer(planner, agent_idx=0)
        )
    # when using the edge-cloud planner we can also visualize the cloud calls
    if args.planner in ["edge_cloud", "multi_tier_edge_cloud"]:
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
    pose = np.array([list(resolve_start_pose(args))])
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
