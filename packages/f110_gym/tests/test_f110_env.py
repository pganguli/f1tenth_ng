"""
Unit tests for the F1TENTH Gymnasium environment.
"""

import gymnasium as gym
import numpy as np
from f110_gym.envs.f110_env import update_lap_counts

import f110_gym  # pylint: disable=unused-import


def test_observation_space():
    """Test observation space structure and types."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=1
    )
    obs, _ = env.reset()

    assert isinstance(obs, dict)
    assert "scans" in obs
    assert "poses_x" in obs
    assert "poses_y" in obs
    assert "poses_theta" in obs
    assert "linear_vels_x" in obs
    assert "linear_vels_y" in obs
    assert "ang_vels_z" in obs

    # Check shapes for 1 agent
    assert obs["scans"].shape == (1, 1080)
    assert obs["poses_x"].shape == (1,)
    env.close()


def test_action_space():
    """Test action space limits and shapes."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=2
    )
    assert env.action_space.shape == (2, 2)
    # Bounds defined by default_params in f110_env.py
    assert env.action_space.low[0, 0] <= -0.4  # steering min
    assert env.action_space.low[0, 1] <= -5.0  # speed min
    env.close()


def test_step():
    """Test a single environment step and basic physics."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=1
    )
    obs, _ = env.reset(options={"poses": np.array([[0.0, 0.0, 0.0]])})

    # Action: steer=0, speed=2.0
    action = np.array([[0.0, 2.0]])
    obs, reward, _, _, _ = env.step(action)

    assert obs["poses_x"][0] > 0.0  # Should have moved forward in x
    assert abs(obs["poses_y"][0]) < 1e-3  # Should not have moved much in y
    assert isinstance(reward, (float, np.float64))
    env.close()


def test_lap_times_update_beyond_two_laps() -> None:
    """Calling the internal utility should update lap_times regardless of
    toggle count (regression for issue where label froze at 2 laps)."""
    # create minimal arrays for a single agent
    poses_x = np.array([0.0])
    poses_y = np.array([0.0])
    start_xs = np.array([0.0])
    start_ys = np.array([0.0])
    start_rot = np.eye(2)
    num_agents = 1
    near_starts = np.array([False])
    toggle_list = np.array([4.0])  # already completed two laps
    lap_counts = np.array([2.0])
    lap_times = np.array([0.0])

    # pick an arbitrary current time
    current_time = 12.34
    update_lap_counts(
        poses_x,
        poses_y,
        start_xs,
        start_ys,
        start_rot,
        num_agents,
        current_time,
        near_starts,
        toggle_list,
        lap_counts,
        lap_times,
    )
    # even though toggle_list >= 4 we expect lap_times to be updated
    assert lap_times[0] == current_time


def test_collision_detection():
    """Test that moving into a wall triggers collision."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=1
    )
    # Reset to a known safe spot then drive into a wall
    env.reset(options={"poses": np.array([[0.0, 0.0, 0.0]])})

    # Drive fast into the side wall
    # We should eventually hit something
    # Oschersleben is a closed track, driving in one direction will hit a wall.
    action = np.array([[0.0, 7.0]])
    collided = False
    for _ in range(100):  # 1 second of simulation
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            collided = True
            break

    assert collided is True
    assert obs["collisions"][0] > 0
    assert obs["collisions"][0] == 1.0
    env.close()


def test_lidar_values():
    """Test that LiDAR scans return reasonable values."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=1
    )
    obs, _ = env.reset(options={"poses": np.array([[0.0, 0.0, 0.0]])})
    scan = obs["scans"][0]

    assert scan.shape == (1080,)
    assert np.all(scan >= 0.0)
    assert np.any(scan < 30.0)  # Should hit something eventually in Oschersleben
    env.close()


def test_multi_agent_observations() -> None:
    """Multi-agent env should expose per-agent arrays with correct shapes."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=2
    )
    obs, _ = env.reset()
    assert obs["scans"].shape == (2, 1080)
    assert obs["poses_x"].shape == (2,)
    assert obs["poses_y"].shape == (2,)
    assert obs["linear_vels_x"].shape == (2,)
    env.close()


def test_observation_space_contains_reset_obs() -> None:
    """The observation returned by reset() must satisfy the declared observation space."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=1
    )
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    env.close()


def test_reset_restores_initial_state() -> None:
    """After several steps, a reset() to the original pose should fully restore state."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=1
    )
    start_pose = np.array([[0.0, 0.0, 0.0]])
    obs1, _ = env.reset(options={"poses": start_pose})
    initial_scan = obs1["scans"].copy()
    initial_x = float(obs1["poses_x"][0])

    # drive forward for several steps so the physics state changes
    for _ in range(20):
        env.step(np.array([[0.0, 2.0]]))

    # reset back to the exact same starting pose
    obs2, _ = env.reset(options={"poses": start_pose})
    assert np.isclose(obs2["poses_x"][0], initial_x)
    assert np.allclose(obs2["scans"], initial_scan)
    env.close()


def test_max_laps_termination() -> None:
    """Episode terminates when toggle_list reaches the required lap count."""
    env = gym.make(
        "f110-v0",
        map="data/maps/F1/Oschersleben/Oschersleben_map",
        num_agents=1,
        max_laps=1,
    )
    env.reset(options={"poses": np.array([[0.0, 0.0, 0.0]])})
    # Directly set the toggle_list so that the next _check_done call sees 2
    # toggles (= 1 completed lap) without needing to drive around the track.
    inner = env.unwrapped
    inner.toggle_list = np.array([2.0])
    _, _, terminated, _, _ = env.step(np.array([[0.0, 0.0]]))
    assert terminated, "Episode should terminate after max_laps=1 is reached"
    env.close()
