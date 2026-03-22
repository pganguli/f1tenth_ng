"""Unit tests for reactive simulation scripts."""

from argparse import Namespace
from typing import Any

import numpy as np

from f110_planning.base import Action
from f110_planning.schedulers import (
    AlwaysCloudScheduler,
    BernoulliMaxMissPolicy,
    FixedBernoulliPolicy,
    PiecewiseLinearRampPolicy,
    PolicyDrivenScheduler,
    TieredPolicyDrivenScheduler,
)
from f110_scripts.sim.reactive_planners import _create_planner, main


def _sim_obs() -> dict[str, Any]:
    """Full observation dict needed by the simulation loop."""
    return {
        "scans": np.zeros((1, 1080)),
        "poses_x": np.zeros(1),
        "poses_y": np.zeros(1),
        "poses_theta": np.zeros(1),
        "linear_vels_x": np.ones(1),
        "linear_vels_y": np.zeros(1),
        "ang_vels_z": np.zeros(1),
        "steering_angles": np.zeros(1),
        "collisions": np.zeros(1),
        "lap_times": np.zeros(1),
        "lap_counts": np.zeros(1),
    }


def test_reactive_planner_produces_valid_action() -> None:
    """Each planner factory creates a planner that returns a valid Action with
    plausible steer and speed when planning on a minimal observation."""
    obs = _sim_obs()
    waypoints = np.ones((4, 2)) * np.array([0.0, 0.0])
    waypoints = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    for planner_name, extra in [
        ("gap", {"speed": 2.0, "bubble_radius": 160}),
        ("bubble", {"speed": 2.0, "safety_radius": 1.3}),
    ]:
        args = Namespace(planner=planner_name, **extra)
        planner = _create_planner(args, waypoints)
        action = planner.plan(obs)
        assert isinstance(action, Action), f"planner {planner_name} must return an Action"
        assert isinstance(action.steer, (float, int))
        assert isinstance(action.speed, (float, int))


def test_simulation_step_logic(mocker: Any) -> None:
    """Smoke test for the simulation loop logic using mocks."""
    # Mock parse_args to return specific values
    # include new fields added by add_common_sim_args
    mocker.patch(
        "f110_scripts.sim.reactive_planners.parse_args",
        return_value=Namespace(
            planner="gap",
            speed=2.0,
            bubble_radius=160,
            waypoints=None,
            render_mode="None",
            max_laps=None,
            start_x=0.0,
            start_y=0.0,
            start_theta=0.0,
            map="Budapest",
        ),
    )

    # Mock dependencies
    mocker.patch(
        "f110_scripts.sim.reactive_planners.load_waypoints", return_value=np.zeros((0, 2))
    )
    mock_env = mocker.Mock()
    base_obs = _sim_obs()
    reset_val = (base_obs, {})
    mock_env.reset.return_value = reset_val
    step_val = (base_obs, 0.1, True, False, {})
    mock_env.step.return_value = step_val  # Immediately terminate
    mock_setup = mocker.patch(
        "f110_scripts.sim.reactive_planners.setup_env", return_value=mock_env
    )

    # Run main - should execute reset and one step before terminating
    main()

    mock_env.reset.assert_called_once()
    mock_env.step.assert_called_once()
    mock_env.close.assert_called_once()

    # ensure setup_env was invoked with our patched namespace and calculated
    # render override (None in this case)
    called_args, called_kwargs = mock_setup.call_args
    assert hasattr(called_args[0], "max_laps")
    assert called_args[0].max_laps is None
    assert called_kwargs == {}


def test_main_resets_with_resolved_start_pose(mocker: Any) -> None:
    """main() should use resolve_start_pose instead of raw start CLI values."""
    mocker.patch(
        "f110_scripts.sim.reactive_planners.parse_args",
        return_value=Namespace(
            planner="gap",
            speed=2.0,
            bubble_radius=160,
            waypoints=None,
            render_mode="None",
            max_laps=None,
            start_x=0.0,
            start_y=0.0,
            start_theta=None,
            map="Monza",
        ),
    )
    mocker.patch(
        "f110_scripts.sim.reactive_planners.load_waypoints", return_value=np.zeros((0, 2))
    )
    mocker.patch(
        "f110_scripts.sim.reactive_planners.resolve_start_pose",
        return_value=(0.0, 0.0, 1.472932),
    )
    mock_env = mocker.Mock()
    base_obs = _sim_obs()
    mock_env.reset.return_value = (base_obs, {})
    mock_env.step.return_value = (base_obs, 0.1, True, False, {})
    mocker.patch("f110_scripts.sim.reactive_planners.setup_env", return_value=mock_env)

    main()

    mock_env.reset.assert_called_once()
    options = mock_env.reset.call_args.kwargs["options"]
    assert np.allclose(options["poses"], np.array([[0.0, 0.0, 1.472932]]))


def test_edge_cloud_strategy_construction() -> None:
    """Factory should build edge-cloud planner for all safety strategies."""
    waypoints = np.zeros((0, 2))
    base = dict(
        planner="edge_cloud",
        speed=None,
        cloud_latency=10,
        alpha_steer=0.5,
        alpha_speed=0.1,
        lookahead=1.5,
        lateral_gain=1.0,
        cloud_interval=2,
        rl_scheduler="does/not/exist.zip",
        safety_threshold=0.03,
        safety_min_interval=1,
        safety_warmup_steps=1,
        safety_fallback_interval=0,
        safety_max_miss=2,
        strategy_seed=7,
        fixed_bernoulli_p=0.65,
        bernoulli_max_miss_p=0.6,
        logistic_center=0.03,
        logistic_slope=40.0,
        logistic_p_min=0.0,
        logistic_p_max=1.0,
        exp_center=0.03,
        exp_rate=25.0,
        exp_p_min=0.0,
        exp_p_max=1.0,
        ramp_d_low=0.01,
        ramp_d_high=0.05,
        deviation_steer_weight=1.0,
        deviation_speed_weight=1.0,
        edge_left_wall_model=None,
        edge_track_width_model=None,
        edge_heading_model=None,
        cloud_left_wall_model=None,
        cloud_track_width_model=None,
        cloud_heading_model=None,
    )

    for strategy in [
        "always",
        "hybrid",
        "deterministic",
        "fixed_bernoulli",
        "bernoulli_max_miss",
        "logistic",
        "exponential",
        "piecewise_ramp",
    ]:
        args = Namespace(**base, cloud_strategy=strategy)
        planner = _create_planner(args, waypoints)
        assert planner is not None
        if strategy in {"always", "hybrid"}:
            assert isinstance(planner.scheduler, AlwaysCloudScheduler)
        else:
            assert isinstance(planner.scheduler, PolicyDrivenScheduler)


def test_fixed_bernoulli_strategy_config() -> None:
    """Fixed Bernoulli strategy should wire the constant probability config."""
    waypoints = np.zeros((0, 2))
    args = Namespace(
        planner="edge_cloud",
        cloud_strategy="fixed_bernoulli",
        speed=None,
        cloud_latency=10,
        alpha_steer=0.5,
        alpha_speed=0.1,
        lookahead=1.5,
        lateral_gain=1.0,
        cloud_interval=2,
        rl_scheduler="does/not/exist.zip",
        safety_threshold=0.03,
        safety_min_interval=1,
        safety_warmup_steps=1,
        safety_fallback_interval=0,
        safety_max_miss=2,
        strategy_seed=7,
        fixed_bernoulli_p=0.65,
        bernoulli_max_miss_p=0.6,
        logistic_center=0.03,
        logistic_slope=40.0,
        logistic_p_min=0.0,
        logistic_p_max=1.0,
        exp_center=0.03,
        exp_rate=25.0,
        exp_p_min=0.0,
        exp_p_max=1.0,
        ramp_d_low=0.01,
        ramp_d_high=0.05,
        deviation_steer_weight=1.5,
        deviation_speed_weight=0.25,
        edge_left_wall_model=None,
        edge_track_width_model=None,
        edge_heading_model=None,
        cloud_left_wall_model=None,
        cloud_track_width_model=None,
        cloud_heading_model=None,
    )
    planner = _create_planner(args, waypoints)
    assert isinstance(planner.scheduler.policy, FixedBernoulliPolicy)  # type: ignore[attr-defined]
    cfg = planner.scheduler.policy.fixed_bernoulli_config  # type: ignore[attr-defined]
    assert cfg.p == 0.65
    assert planner.deviation_steer_weight == 1.5
    assert planner.deviation_speed_weight == 0.25


def test_bernoulli_max_miss_strategy_config() -> None:
    """Bernoulli+guard strategy should wire probability and max-miss controls."""
    waypoints = np.zeros((0, 2))
    args = Namespace(
        planner="edge_cloud",
        cloud_strategy="bernoulli_max_miss",
        speed=None,
        cloud_latency=10,
        alpha_steer=0.5,
        alpha_speed=0.1,
        lookahead=1.5,
        lateral_gain=1.0,
        cloud_interval=2,
        rl_scheduler="does/not/exist.zip",
        safety_threshold=0.03,
        safety_min_interval=1,
        safety_warmup_steps=1,
        safety_fallback_interval=0,
        safety_max_miss=2,
        strategy_seed=7,
        fixed_bernoulli_p=0.65,
        bernoulli_max_miss_p=0.6,
        logistic_center=0.03,
        logistic_slope=40.0,
        logistic_p_min=0.0,
        logistic_p_max=1.0,
        exp_center=0.03,
        exp_rate=25.0,
        exp_p_min=0.0,
        exp_p_max=1.0,
        ramp_d_low=0.01,
        ramp_d_high=0.05,
        deviation_steer_weight=1.0,
        deviation_speed_weight=1.0,
        edge_left_wall_model=None,
        edge_track_width_model=None,
        edge_heading_model=None,
        cloud_left_wall_model=None,
        cloud_track_width_model=None,
        cloud_heading_model=None,
    )
    planner = _create_planner(args, waypoints)
    assert isinstance(planner.scheduler.policy, BernoulliMaxMissPolicy)  # type: ignore[attr-defined]
    cfg = planner.scheduler.policy.bernoulli_max_miss_config  # type: ignore[attr-defined]
    assert cfg.p == 0.6
    assert cfg.max_miss == 2


def test_piecewise_ramp_strategy_config() -> None:
    """Piecewise ramp strategy should wire the ramp thresholds into the policy."""
    waypoints = np.zeros((0, 2))
    args = Namespace(
        planner="edge_cloud",
        cloud_strategy="piecewise_ramp",
        speed=None,
        cloud_latency=10,
        alpha_steer=0.5,
        alpha_speed=0.1,
        lookahead=1.5,
        lateral_gain=1.0,
        cloud_interval=2,
        rl_scheduler="does/not/exist.zip",
        safety_threshold=0.03,
        safety_min_interval=1,
        safety_warmup_steps=1,
        safety_fallback_interval=0,
        safety_max_miss=2,
        strategy_seed=7,
        fixed_bernoulli_p=0.65,
        bernoulli_max_miss_p=0.6,
        logistic_center=0.03,
        logistic_slope=40.0,
        logistic_p_min=0.0,
        logistic_p_max=1.0,
        exp_center=0.03,
        exp_rate=25.0,
        exp_p_min=0.0,
        exp_p_max=1.0,
        ramp_d_low=0.02,
        ramp_d_high=0.06,
        deviation_steer_weight=1.0,
        deviation_speed_weight=1.0,
        edge_left_wall_model=None,
        edge_track_width_model=None,
        edge_heading_model=None,
        cloud_left_wall_model=None,
        cloud_track_width_model=None,
        cloud_heading_model=None,
    )
    planner = _create_planner(args, waypoints)
    assert isinstance(planner.scheduler.policy, PiecewiseLinearRampPolicy)  # type: ignore[attr-defined]
    cfg = planner.scheduler.policy.piecewise_linear_ramp_config  # type: ignore[attr-defined]
    assert cfg.d_low == 0.02
    assert cfg.d_high == 0.06


def test_multi_tier_edge_cloud_strategy_construction() -> None:
    """Factory should build multi-tier planner for tiered strategies."""
    waypoints = np.zeros((0, 2))
    base = dict(
        planner="multi_tier_edge_cloud",
        speed=None,
        cloud_strategy="tiered_probabilistic",
        lookahead=1.5,
        lateral_gain=1.0,
        safety_min_interval=1,
        safety_warmup_steps=1,
        safety_fallback_interval=0,
        safety_max_miss=2,
        strategy_seed=7,
        tier_medium_prob=0.3,
        tier_large_prob=0.2,
        tier_medium_threshold=0.03,
        tier_large_threshold=0.08,
        tier_steer_weight=1.0,
        tier_speed_weight=0.0,
        medium_cloud_latency=6,
        large_cloud_latency=10,
        alpha_steer_medium=0.5,
        alpha_speed_medium=0.1,
        alpha_steer_large=0.7,
        alpha_speed_large=0.2,
        edge2_left_wall_model=None,
        edge2_track_width_model=None,
        edge2_heading_model=None,
        medium_left_wall_model=None,
        medium_track_width_model=None,
        medium_heading_model=None,
        large_left_wall_model=None,
        large_track_width_model=None,
        large_heading_model=None,
    )

    planner_prob = _create_planner(Namespace(**base), waypoints)
    assert isinstance(planner_prob.scheduler, TieredPolicyDrivenScheduler)

    planner_thresh = _create_planner(
        Namespace(**{**base, "cloud_strategy": "tiered_threshold"}),
        waypoints,
    )
    assert isinstance(planner_thresh.scheduler, TieredPolicyDrivenScheduler)
    assert planner_thresh.oracle_disagreement_routing
