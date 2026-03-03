"""Unit tests for tracking simulation scripts."""

import numpy as np
from f110_planning.base import Action
from f110_planning.tracking import PurePursuitPlanner

from f110_scripts.sim.tracking_planners import _init_planners


def test_init_planners_count_and_hybrid() -> None:
    """Without hybrid all planners are PurePursuitPlanner; the hybrid branch
    requires a display and is covered by the headless guard below."""
    wpts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    planners_no_hybrid = _init_planners(2, [wpts, wpts], enable_hybrid=False)

    # Without hybrid all planners are PurePursuitPlanner
    assert all(isinstance(p, PurePursuitPlanner) for p in planners_no_hybrid)
    assert len(planners_no_hybrid) == 2


def test_init_planners_plan_on_valid_obs() -> None:
    """Each planner initialised by _init_planners should produce a valid Action
    when planning on a minimal observation."""
    wpts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    planners = _init_planners(2, [wpts, wpts], enable_hybrid=False)
    obs = {
        "poses_x": np.array([0.0, 0.5]),
        "poses_y": np.array([0.0, 0.0]),
        "poses_theta": np.array([0.0, 0.0]),
        "linear_vels_x": np.array([1.0, 1.0]),
        "linear_vels_y": np.array([0.0, 0.0]),
        "ang_vels_z": np.array([0.0, 0.0]),
        "scans": np.zeros((2, 1080)),
        "steering_angles": np.zeros((2,)),
        "collisions": np.zeros((2,)),
    }
    for i, planner in enumerate(planners):
        action = planner.plan(obs, ego_idx=i)
        assert isinstance(action, Action)
