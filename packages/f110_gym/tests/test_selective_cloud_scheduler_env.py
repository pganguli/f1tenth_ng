"""Unit tests for SelectiveCloudSchedulerEnv."""

import numpy as np
import gymnasium as gym

import f110_gym  # noqa: F401  # pylint: disable=unused-import  # registers gym envs


def _make_env(top_k: int = 1) -> gym.Env:
    wpts = np.array([[0.0, 0.0], [1.0, 0.0]])
    return gym.make(
        "f110_gym:f110-selective-cloud-scheduler-v0",
        map="data/maps/F1/Oschersleben/Oschersleben_map",
        waypoints=wpts,
        num_agents=1,
        cloud_latency=2,
        top_k=top_k,
        render_mode=None,
    )


# ---------------------------------------------------------------------------
# _resolve_topk
# ---------------------------------------------------------------------------

def test_resolve_topk_selects_highest_logit() -> None:
    """top_k=1: only the index with the largest logit should be True."""
    env = _make_env(top_k=1)
    logits = np.array([0.1, 5.0, 0.3], dtype=np.float32)
    mask = env.unwrapped._resolve_topk(logits)  # pylint: disable=protected-access
    assert mask == [False, True, False]
    env.close()


def test_resolve_topk_selects_top2() -> None:
    """top_k=2: the two highest logits should be True."""
    env = _make_env(top_k=2)
    logits = np.array([0.1, 5.0, 3.0], dtype=np.float32)
    mask = env.unwrapped._resolve_topk(logits)  # pylint: disable=protected-access
    assert mask[0] is False
    assert mask[1] is True
    assert mask[2] is True
    env.close()


def test_resolve_topk_uniform_logits_returns_exactly_k() -> None:
    """Uniform logits: exactly top_k entries must be True."""
    env = _make_env(top_k=2)
    logits = np.zeros(3, dtype=np.float32)
    mask = env.unwrapped._resolve_topk(logits)  # pylint: disable=protected-access
    assert sum(mask) == 2
    env.close()


def test_resolve_topk_numerically_stable_large_logits() -> None:
    """Large logit differences (overflow-prone) should still produce a valid mask."""
    env = _make_env(top_k=1)
    logits = np.array([-1000.0, 1000.0, -1000.0], dtype=np.float32)
    # Numerically stable softmax should handle this without nan/inf
    mask = env.unwrapped._resolve_topk(logits)  # pylint: disable=protected-access
    assert mask == [False, True, False]
    env.close()


# ---------------------------------------------------------------------------
# Action / Observation spaces
# ---------------------------------------------------------------------------

def test_action_space_is_box_of_shape_3() -> None:
    env = _make_env()
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (3,)
    env.close()


def test_observation_space_has_extra_keys() -> None:
    """New DNN-related keys must be present after reset."""
    expected_keys = (
        "edge_left_dist",
        "edge_track_width",
        "edge_heading_error",
        "cloud_left_dist",
        "cloud_track_width",
        "cloud_heading_error",
        "last_steer",
        "last_speed",
        "cloud_calls_mask",
    )
    env = _make_env()
    obs, _ = env.reset()
    for key in expected_keys:
        assert key in obs, f"Missing observation key: {key}"
    env.close()


def test_observation_excludes_map_absolute_keys() -> None:
    """Map-absolute keys must not reach the RL agent; scans are kept."""
    excluded = (
        "poses_x", "poses_y", "poses_theta",
        "ang_vels_z", "ego_idx", "collisions", "lap_times", "lap_counts",
    )
    env = _make_env()
    obs, _ = env.reset()
    for key in excluded:
        assert key not in obs, f"Key '{key}' should be excluded from agent obs"
    assert "scans" in obs, "scans should be present in agent obs"
    env.close()


# ---------------------------------------------------------------------------
# step / reset
# ---------------------------------------------------------------------------

def test_step_returns_finite_non_positive_reward() -> None:
    env = _make_env()
    env.reset()
    _obs, reward, _term, _trunc, _info = env.step(np.zeros(3, dtype=np.float32))
    assert isinstance(reward, float)
    assert reward <= 0.0
    env.close()


def test_step_info_contains_call_mask() -> None:
    env = _make_env()
    env.reset()
    _obs, _r, _term, _trunc, info = env.step(np.zeros(3, dtype=np.float32))
    assert "call_mask" in info
    assert len(info["call_mask"]) == 3
    env.close()


def test_reset_clears_planner_call_mask() -> None:
    """After reset, cloud_calls_mask in the observation should be all-zero."""
    env = _make_env()
    env.reset()
    # Force a call by boosting one logit
    env.step(np.array([100.0, -100.0, -100.0], dtype=np.float32))
    obs, _ = env.reset()
    assert int(obs["cloud_calls_mask"].sum()) == 0
    env.close()


def test_observation_space_contains_step_obs() -> None:
    """Each step observation must satisfy the declared observation space."""
    env = _make_env()
    env.reset()
    obs, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert env.observation_space.contains(obs)
    env.close()


def test_top_k_enforced_in_call_mask() -> None:
    """With top_k=1 the call_mask returned in info must have exactly 1 True."""
    env = _make_env(top_k=1)
    env.reset()
    _obs, _r, _term, _trunc, info = env.step(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert sum(info["call_mask"]) == 1
    env.close()
