"""Unit tests for reactive simulation scripts."""

from argparse import Namespace
from typing import Any

import numpy as np

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
    from f110_planning.base import Action

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
