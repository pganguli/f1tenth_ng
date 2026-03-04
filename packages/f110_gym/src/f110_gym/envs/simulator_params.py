"""
Configuration parameters for the F1TENTH Simulator.
"""

from dataclasses import dataclass
from typing import Any
from .integrator import Integrator


@dataclass
class SimulatorParams:
    """
    Configuration parameters for the Simulator.
    """

    vehicle_params: dict[str, Any]
    num_agents: int
    seed: int
    time_step: float = 0.01
    ego_idx: int = 0
    integrator: Integrator = Integrator.RK4
    lidar_dist: float = 0.0
