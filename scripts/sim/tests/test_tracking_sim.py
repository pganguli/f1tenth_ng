"""
Unit tests for tracking simulation scripts.
"""

import numpy as np
from f110_planning.tracking import PurePursuitPlanner

from scripts.sim.tracking_planners import _init_planners


def test_init_planners():
    """Test initialization of multiple tracking planners for multi-agent simulation."""
    num_agents = 2
    agent_waypoints = [np.array([[0, 0]]), np.array([[1, 1]])]
    planners = _init_planners(num_agents, agent_waypoints, enable_hybrid=False)

    assert len(planners) == 2
    assert isinstance(planners[0], PurePursuitPlanner)
    assert isinstance(planners[1], PurePursuitPlanner)
