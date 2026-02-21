"""
Unit tests for the F1TENTH Gymnasium environment.
"""

import gymnasium as gym
import numpy as np

import f110_gym  # pylint: disable=unused-import


def test_registration():
    """Test that the environment is correctly registered."""
    spec_ids = [spec.id for spec in gym.envs.registry.values()]
    assert "f110-v0" in spec_ids


def test_env_init():
    """Test environment initialization."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=1
    )
    assert env.unwrapped.num_agents == 1
    env.close()


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


def test_default_max_laps_is_none() -> None:
    """New default should not terminate after two laps."""
    env = gym.make(
        "f110-v0", map="data/maps/F1/Oschersleben/Oschersleben_map", num_agents=1
    )
    # ensure the internal field was set to None by default
    assert env.unwrapped.max_laps is None
    env.close()


def test_lap_times_update_beyond_two_laps() -> None:
    """Calling the internal utility should update lap_times regardless of
    toggle count (regression for issue where label froze at 2 laps)."""
    from f110_gym.envs.f110_env import update_lap_counts

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
