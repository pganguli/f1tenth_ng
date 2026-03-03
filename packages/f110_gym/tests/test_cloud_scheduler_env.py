"""Unit tests for the cloud scheduler RL environment."""

import numpy as np
import gymnasium as gym

import f110_gym  # ensures registration


def _make_env():
    # small dummy waypoint (straight line)
    wpts = np.array([[0.0, 0.0], [1.0, 0.0]])
    env = gym.make(
        "f110_gym:f110-cloud-scheduler-v0",
        map="data/maps/F1/Oschersleben/Oschersleben_map",
        waypoints=wpts,
        num_agents=1,
        cloud_latency=2,
        render_mode=None,
    )
    return env


def test_spaces():
    env = _make_env()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 2
    obs = env.reset()[0]
    # underlying observations plus extras
    assert "scans" in obs
    assert "latest_cloud_action" in obs
    assert obs["latest_cloud_action"].shape == (2,)
    assert "cloud_request_pending" in obs
    assert "crosstrack_dist" in obs
    assert obs["crosstrack_dist"].shape == (1,)
    # should be nonnegative
    assert obs["crosstrack_dist"][0] >= 0.0
    env.close()


def test_step_and_reward():
    env = _make_env()
    obs, _ = env.reset()
    # initially no cloud and reward should be non-positive
    assert obs["cloud_request_pending"] in (0, 1)
    r0 = env.step(0)[1]
    assert isinstance(r0, float)
    assert r0 <= 0
    # take a call and step two times to let latency materialize
    env.reset()
    env.step(1)
    obs2, r2, term2, trunc2, info2 = env.step(0)
    # since latency is 2, the latest_cloud_action should still be default
    assert "latest_cloud_action" in info2
    assert info2["latest_cloud_action"].shape == (2,)
    assert isinstance(r2, float)
    env.close()


def test_reset_clears_scheduler():
    env = _make_env()
    env.reset()
    env.step(1)
    # after reset scheduler action pending should be None/zero
    obs, _ = env.reset()
    assert obs["cloud_request_pending"] == 0
    env.close()


def test_cloud_latency_countdown() -> None:
    """With cloud_latency=2 the cloud action should not materialise before 2 steps."""
    env = _make_env()  # cloud_latency=2
    env.reset()

    # step 0: request cloud
    env.step(1)
    planner = env.unwrapped._planner
    assert planner._latest_cloud_action is None, "Cloud should not arrive before latency elapses"

    # step 1: no new cloud request; action still pending
    env.step(0)
    assert planner._latest_cloud_action is None, "Cloud still should not have arrived"

    # step 2: the pending request's arrival time is reached – cloud materialises
    env.step(0)
    assert planner._latest_cloud_action is not None, "Cloud action should have materialised by now"

    env.close()


def test_observation_space_contains_step_obs() -> None:
    """Each step observation must satisfy the declared observation space."""
    env = _make_env()
    env.reset()
    obs, _, _, _, _ = env.step(0)
    assert env.observation_space.contains(obs)
    env.close()
