"""
Unit tests for reactive simulation scripts.
"""

from argparse import Namespace
import numpy as np
from scripts.sim.reactive_planners import _create_planner, main


def test_reactive_planner_creation():
    """Tests the factory function for reactive planners."""
    waypoints = np.zeros((10, 2))

    # Gap Follower
    args_gap = Namespace(planner="gap", speed=2.0, bubble_radius=160)
    planner_gap = _create_planner(args_gap, waypoints)
    assert planner_gap.__class__.__name__ == "GapFollowerPlanner"

    # Bubble Planner
    args_bubble = Namespace(planner="bubble", speed=2.0, safety_radius=1.3)
    planner_bubble = _create_planner(args_bubble, waypoints)
    assert planner_bubble.__class__.__name__ == "BubblePlanner"

    # Dynamic Waypoint
    args_dyn = Namespace(planner="dynamic", speed=3.0, lookahead=1.5, lateral_gain=1.0)
    planner_dyn = _create_planner(args_dyn, waypoints)
    assert planner_dyn.__class__.__name__ == "DynamicWaypointPlanner"


def test_simulation_step_logic(mocker):
    """Smoke test for the simulation loop logic using mocks."""
    # Mock parse_args to return specific values
    mocker.patch("scripts.sim.reactive_planners.parse_args", return_value=Namespace(
        planner="gap", speed=2.0, bubble_radius=160, waypoints=None,
        render_mode="None", start_x=0.0, start_y=0.0, start_theta=0.0,
        map="Budapest"
    ))

    # Mock dependencies
    mocker.patch("scripts.sim.reactive_planners.load_waypoints", return_value=np.zeros((0, 2)))
    mock_env = mocker.Mock()
    # Scans should be [num_agents, 1080]
    reset_val = ({"scans": np.zeros((1, 1080))}, {})
    mock_env.reset.return_value = reset_val
    step_val = ({"scans": np.zeros((1, 1080))}, 0.1, True, False, {})
    mock_env.step.return_value = step_val  # Immediately terminate
    mocker.patch("scripts.sim.reactive_planners.setup_env", return_value=mock_env)

    # Run main - should execute reset and one step before terminating
    main()

    mock_env.reset.assert_called_once()
    mock_env.step.assert_called_once()
    mock_env.close.assert_called_once()
