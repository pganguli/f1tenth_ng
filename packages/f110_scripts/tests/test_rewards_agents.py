"""Unit tests for the reward factory, agent factory, and selective renderer."""

from typing import Any

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from f110_scripts.train.rewards import make_reward
from f110_scripts.train.agents import make_agent
from f110_planning.render_callbacks.trace import (
    _DNN_COLORS,
    _MULTI_CALL_COLOR,
    create_selective_cloud_call_renderer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

WPTS = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

OBS_ON_PATH: dict[str, Any] = {
    "poses_x": np.array([0.0]),
    "poses_y": np.array([0.1]),  # slight offset from waypoints
}


# ---------------------------------------------------------------------------
# Reward factory
# ---------------------------------------------------------------------------

def test_make_reward_cte_returns_callable() -> None:
    fn = make_reward("cte", WPTS)
    assert callable(fn)


def test_cte_reward_is_non_positive() -> None:
    fn = make_reward("cte", WPTS)
    r = fn(OBS_ON_PATH, [False, False, False])
    assert isinstance(r, float)
    assert r <= 0.0


def test_cte_ignores_call_mask() -> None:
    """cte reward must not change based on how many DNNs are called."""
    fn = make_reward("cte", WPTS)
    r_no_call = fn(OBS_ON_PATH, [False, False, False])
    r_all_call = fn(OBS_ON_PATH, [True, True, True])
    assert r_no_call == pytest.approx(r_all_call)


def test_make_reward_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        make_reward("this_reward_does_not_exist", WPTS)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

class _MockEnv(gym.Env):
    """Minimal Gymnasium env satisfying SB3 MultiInputPolicy + PPO requirements."""

    observation_space = spaces.Dict(
        {"obs": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)}
    )
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, **_: Any):
        return {"obs": np.zeros(4, dtype=np.float32)}, {}

    def step(self, _: Any):
        return {"obs": np.zeros(4, dtype=np.float32)}, 0.0, False, False, {}

    def render(self) -> None:  # pragma: no cover
        pass

    def close(self) -> None:
        pass


def test_make_agent_ppo_returns_learnable_instance() -> None:
    agent = make_agent("ppo", _MockEnv(), verbose=0)
    assert hasattr(agent, "learn")
    assert hasattr(agent, "save")


def test_make_agent_is_case_insensitive() -> None:
    """'PPO' and 'ppo' should resolve to the same algorithm class."""
    a1 = make_agent("ppo", _MockEnv(), verbose=0)
    a2 = make_agent("PPO", _MockEnv(), verbose=0)
    assert type(a1) is type(a2)


def test_make_agent_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        make_agent("nonexistent_algo", _MockEnv(), verbose=0)


# ---------------------------------------------------------------------------
# Color constants for create_selective_cloud_call_renderer
# ---------------------------------------------------------------------------

def test_dnn_colors_has_three_entries() -> None:
    """There must be exactly one color per DNN (3 DNNs)."""
    assert len(_DNN_COLORS) == 3


def test_all_dnn_colors_are_distinct() -> None:
    """Each DNN should have a unique color to be visually distinguishable."""
    assert len(set(_DNN_COLORS)) == 3


def test_multi_call_color_is_white() -> None:
    """White is reserved for the case where 2+ DNNs are called simultaneously."""
    assert _MULTI_CALL_COLOR == (255, 255, 255)


def test_dnn_colors_are_not_multi_call_color() -> None:
    """No single-DNN color should clash with the multi-call colour."""
    for color in _DNN_COLORS:
        assert color != _MULTI_CALL_COLOR


# ---------------------------------------------------------------------------
# create_selective_cloud_call_renderer — callable contract
# ---------------------------------------------------------------------------

def test_renderer_factory_returns_callable() -> None:
    class _FakePlanner:
        last_call_mask: list[bool] = [False, False, False]

    renderer_fn = create_selective_cloud_call_renderer(_FakePlanner())
    assert callable(renderer_fn)
