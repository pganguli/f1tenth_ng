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
    SelectiveEdgeCloudPlanner,
)
from f110_planning.render_callbacks import (
    create_camera_tracking,
    create_dynamic_waypoint_renderer,
    create_heading_error_renderer,
    create_trace_renderer,
    create_cloud_call_renderer,
    create_selective_cloud_call_renderer,
    create_waypoint_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.base import CloudScheduler
from f110_planning.schedulers import FixedIntervalScheduler, RoundRobinScheduler, SensitivityProportionalScheduler
from f110_planning.utils import add_common_sim_args, load_waypoints, resolve_start_pose, setup_env
from stable_baselines3 import PPO
import gymnasium.spaces as _gym_spaces


# ------------------------------------------------------------------
# Shared top-k helper (mirrors SelectiveCloudSchedulerEnv._resolve_topk)
# ------------------------------------------------------------------
def _resolve_topk(logits: np.ndarray, top_k: int) -> list[bool]:
    """Numerically-stable softmax → top-k bool mask."""
    logits = np.asarray(logits, dtype=np.float64).ravel()
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    top_k_idx = np.argsort(probs)[-top_k:]
    mask = [False] * len(logits)
    for i in top_k_idx:
        mask[i] = True
    return mask


# ------------------------------------------------------------------
# Cloud scheduler wrapper for *binary* policies saved by train_rl.py
# ------------------------------------------------------------------
class PolicyScheduler(CloudScheduler):
    """``CloudScheduler`` backed by an SB3 binary policy (Discrete / Box(1))."""

    def __init__(self, policy_path: str) -> None:
        self._model = PPO.load(policy_path)

    @classmethod
    def _from_model(cls, model: Any) -> "PolicyScheduler":
        """Construct from an already-loaded SB3 model (skips file I/O)."""
        inst = cls.__new__(cls)
        inst._model = model  # pylint: disable=protected-access
        return inst

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
        action, _ = self._model.predict(obs_rl, deterministic=True)
        return bool(action)

    def reset(self) -> None:  # type: ignore[override]
        """No persistent state to reset for a stateless policy."""


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Thin wrapper: SelectiveEdgeCloudPlanner + any get_call_mask() scheduler
# ------------------------------------------------------------------
class _SelectiveDumbPlanner:  # pylint: disable=too-few-public-methods
    """Wraps :class:`SelectiveEdgeCloudPlanner` with a deterministic mask scheduler.

    The scheduler must expose ``get_call_mask() -> list[bool]`` and
    ``reset()``.  Matches the public surface of
    :class:`SelectivePolicyPlanner` so rendering callbacks work unchanged.
    """

    def __init__(
        self,
        inner_planner: SelectiveEdgeCloudPlanner,
        scheduler: Any,
    ) -> None:
        self._planner = inner_planner
        self._scheduler = scheduler

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Any:
        call_mask = self._scheduler.get_call_mask()
        return self._planner.plan(obs, call_mask=call_mask, ego_idx=ego_idx)

    def reset(self) -> None:
        self._planner.reset()
        self._scheduler.reset()

    @property
    def last_call_mask(self) -> list[bool]:
        return self._planner.last_call_mask

    @property
    def last_target_point(self) -> Any:
        return self._planner.last_target_point


# Selective-policy planner (Box(3) logit action, per-DNN scheduling)
# ------------------------------------------------------------------
class SelectivePolicyPlanner:  # pylint: disable=too-few-public-methods
    """Wraps :class:`SelectiveEdgeCloudPlanner` + a trained SB3 policy.

    At each planning step the policy predicts logits over the m=3 DNN slots;
    the top-k indices are selected as the ``call_mask`` passed to the inner
    :class:`SelectiveEdgeCloudPlanner`.  The augmented RL observation is
    reconstructed from the planner's last outputs so the policy receives the
    same feature vector it was trained on.
    """

    def __init__(
        self,
        inner_planner: SelectiveEdgeCloudPlanner,
        model: Any,
        waypoints: np.ndarray,
        top_k: int = 1,
    ) -> None:
        self._planner = inner_planner
        self._model = model
        self._waypoints = waypoints
        self._top_k = top_k
        # Keys expected by this model's stored observation space — used to
        # gracefully handle models trained before cloud_age was added.
        self._model_obs_keys: frozenset[str] = frozenset(
            getattr(model, "observation_space", {}).spaces.keys()
            if hasattr(getattr(model, "observation_space", None), "spaces")
            else []
        )

    # ------------------------------------------------------------------
    # Public planner interface
    # ------------------------------------------------------------------
    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Any:
        obs_rl = self._build_rl_obs(obs)
        logits, _ = self._model.predict(obs_rl, deterministic=True)
        call_mask = _resolve_topk(np.asarray(logits), self._top_k)
        return self._planner.plan(obs, call_mask=call_mask, ego_idx=ego_idx)

    def reset(self) -> None:
        self._planner.reset()

    # Forward render-callback attributes
    @property
    def last_call_mask(self) -> list[bool]:
        return self._planner.last_call_mask

    @property
    def last_target_point(self) -> Any:
        return self._planner.last_target_point

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    # Keys that SelectiveCloudSchedulerEnv strips from the agent observation.
    # Must stay in sync with SelectiveCloudSchedulerEnv._OBS_EXCLUDED_KEYS.
    _OBS_EXCLUDED_KEYS: frozenset[str] = frozenset({
        "poses_x", "poses_y", "poses_theta",
        "ang_vels_z", "ego_idx", "collisions", "lap_times", "lap_counts",
    })

    def _build_rl_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct the augmented observation expected by the trained policy."""
        p = self._planner
        last = p.last_action
        rl_obs = {k: v for k, v in obs.items() if k not in self._OBS_EXCLUDED_KEYS}
        rl_obs["edge_left_dist"] = np.array([p.last_edge_left], dtype=np.float32)
        rl_obs["edge_track_width"] = np.array([p.last_edge_track], dtype=np.float32)
        rl_obs["edge_heading_error"] = np.array([p.last_edge_heading], dtype=np.float32)
        rl_obs["cloud_left_dist"] = np.array([p.last_cloud_left], dtype=np.float32)
        rl_obs["cloud_track_width"] = np.array([p.last_cloud_track], dtype=np.float32)
        rl_obs["cloud_heading_error"] = np.array([p.last_cloud_heading], dtype=np.float32)
        rl_obs["last_steer"] = np.array(
            [last.steer if last is not None else 0.0], dtype=np.float32
        )
        rl_obs["last_speed"] = np.array(
            [last.speed if last is not None else 0.0], dtype=np.float32
        )
        rl_obs["cloud_calls_mask"] = np.array(p.last_call_mask, dtype=np.int8)
        # cloud_age was added after some models were trained; only include it
        # when the loaded model's observation space actually expects it.
        if not self._model_obs_keys or "cloud_age" in self._model_obs_keys:
            rl_obs["cloud_age"] = np.clip(
                np.array(p.last_cloud_age, dtype=np.float32), 0.0, 999.0
            )
        return rl_obs


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
        choices=["always", "interval", "rl", "round_robin", "sensitivity"],
        default="rl",
        help=(
            "Cloud calling strategy: 'always' issues a request every step, "
            "'interval' uses --cloud-interval spacing, 'rl' loads a policy "
            "specified by --rl-scheduler, 'round_robin' cycles through DNNs "
            "sequentially (top-k per step), and 'sensitivity' calls DNNs "
            "proportionally to --call-weights."
        ),
    )
    parser.add_argument(
        "--call-weights",
        type=float,
        nargs="+",
        default=None,
        metavar="W",
        help=(
            "Per-DNN sensitivity weights for the 'sensitivity' cloud strategy "
            "(space-separated floats, one per DNN slot; normalised internally)."
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
        "--cloud-dot-history",
        type=int,
        default=10000,
        help="Maximum number of cloud-call dots retained on the plot before oldest are removed.",
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
        default=0.7,
        help="Cloud weight for steering (0 = edge only, 1 = cloud only).",
    )
    ec.add_argument(
        "--alpha-speed",
        type=float,
        default=0.2,
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
            "Path to a saved SB3 policy (.zip) for cloud scheduling.  If the "
            "file exists the policy will be used; otherwise the fixed-interval "
            "scheduler (--cloud-interval) is employed.  Both the old binary "
            "scheduler (Discrete/Box(1)) and the new selective per-DNN "
            "scheduler (Box(3) logits) are auto-detected from the model."
        ),
    )
    ec.add_argument(
        "--rl-agent",
        type=str,
        default="ppo",
        help="SB3 algorithm used to save --rl-scheduler (ppo | sac | td3 | a2c).",
    )
    ec.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of DNNs to call per step when using the selective RL policy.",
    )
    ec.add_argument(
        "--edge-left-wall-model",
        type=str,
        default="data/models/left_wall_dist_arch2.pt",
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
        default="data/models/heading_error_arch2.pt",
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
        # Shared planner kwargs (minus scheduler/top_k which depend on path)
        ec_kwargs: dict[str, Any] = {
            "cloud_latency": args.cloud_latency,
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
            ec_kwargs["max_speed"] = args.speed

        if args.cloud_strategy == "always":
            return EdgeCloudPlanner(scheduler=FixedIntervalScheduler(interval=1), **ec_kwargs)

        if args.cloud_strategy == "interval":
            return EdgeCloudPlanner(
                scheduler=FixedIntervalScheduler(interval=args.cloud_interval), **ec_kwargs
            )

        if args.cloud_strategy == "round_robin":
            top_k = getattr(args, "top_k", 1)
            inner = SelectiveEdgeCloudPlanner(top_k=top_k, **ec_kwargs)
            scheduler = RoundRobinScheduler(
                num_dnns=SelectiveEdgeCloudPlanner.NUM_DNNS, top_k=top_k
            )
            return _SelectiveDumbPlanner(inner, scheduler)

        if args.cloud_strategy == "sensitivity":
            top_k = getattr(args, "top_k", 1)
            weights = getattr(args, "call_weights", None)
            if not weights:
                raise ValueError(
                    "--call-weights must be provided for the 'sensitivity' cloud strategy."
                )
            inner = SelectiveEdgeCloudPlanner(top_k=top_k, **ec_kwargs)
            scheduler = SensitivityProportionalScheduler(weights=list(weights), top_k=top_k)
            return _SelectiveDumbPlanner(inner, scheduler)

        # rl: auto-detect policy type from saved action space
        policy_path = args.rl_scheduler
        if policy_path is not None and Path(policy_path).exists():
            print(f"Using RL scheduler at {policy_path}")
            # Use the module-level PPO for the default case so tests can
            # monkeypatch it.  Other algorithms go through REGISTRY.
            algo_name = getattr(args, "rl_agent", "ppo").lower()
            if algo_name == "ppo":
                algo_cls = PPO  # module-level, monkeypatchable
            else:
                from f110_scripts.train.agents import REGISTRY as _AGENT_REGISTRY  # pylint: disable=import-outside-toplevel
                if not _AGENT_REGISTRY:
                    from stable_baselines3 import A2C, SAC, TD3  # pylint: disable=import-outside-toplevel
                    _AGENT_REGISTRY.update({"sac": SAC, "td3": TD3, "a2c": A2C})
                algo_cls = _AGENT_REGISTRY.get(algo_name, PPO)
            model = algo_cls.load(policy_path)

            # Detect by action space: Box(3) → selective per-DNN policy
            action_space = getattr(model, "action_space", None)
            if (
                isinstance(action_space, _gym_spaces.Box)
                and action_space.shape == (3,)
            ):
                top_k = getattr(args, "top_k", 1)
                inner = SelectiveEdgeCloudPlanner(top_k=top_k, **ec_kwargs)
                return SelectivePolicyPlanner(inner, model, waypoints, top_k=top_k)

            # Fallback: old binary policy → PolicyScheduler + EdgeCloudPlanner
            return EdgeCloudPlanner(scheduler=PolicyScheduler._from_model(model), **ec_kwargs)

        # No valid policy file → fixed-interval fallback
        return EdgeCloudPlanner(
            scheduler=FixedIntervalScheduler(interval=args.cloud_interval), **ec_kwargs
        )

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
        max_pts = getattr(args, "cloud_dot_history", 10000)
        if hasattr(planner, "last_call_mask"):
            # SelectiveEdgeCloudPlanner: colour-coded per-DNN dots
            env.unwrapped.add_render_callback(
                create_selective_cloud_call_renderer(planner, agent_idx=0, max_points=max_pts)
            )
        else:
            # EdgeCloudPlanner: single orange dot on any cloud call
            env.unwrapped.add_render_callback(
                create_cloud_call_renderer(planner, agent_idx=0, max_points=max_pts)
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
    metrics = MetricAggregator.create_default(waypoints=wpts, selective_planner=planner)
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
