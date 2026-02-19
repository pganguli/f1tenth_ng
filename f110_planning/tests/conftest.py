"""
Shared pytest fixtures for f110_planning tests.
"""

from typing import Any

import numpy as np
import pytest


@pytest.fixture
def dummy_obs() -> dict[str, Any]:
    """Provides a dummy observation dictionary."""
    return {
        "poses_x": np.array([0.0]),
        "poses_y": np.array([0.0]),
        "poses_theta": np.array([0.0]),
        "linear_vels_x": np.array([1.0]),
        "linear_vels_y": np.array([0.0]),
        "ang_vels_z": np.array([0.0]),
    }
