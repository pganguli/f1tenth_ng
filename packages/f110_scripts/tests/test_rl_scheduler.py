"""Unit tests for the RL-based scheduler option in the reactive script."""

from argparse import Namespace
from pathlib import Path

import numpy as np

from f110_scripts.sim.reactive_planners import _create_planner, PolicyScheduler
from f110_planning.schedulers import FixedIntervalScheduler


class DummyModel:
    @staticmethod
    def load(path):
        return DummyModel()

    def predict(self, obs, deterministic=True):
        # always return non-zero action
        return 1, None


def test_rl_scheduler_used_when_file_exists(tmp_path, monkeypatch):
    """Existing model file should trigger PolicyScheduler."""
    model_file = tmp_path / "sched.zip"
    model_file.write_text("dummy")
    monkeypatch.setattr("f110_scripts.sim.reactive_planners.PPO", DummyModel)

    args = Namespace(
        planner="edge_cloud",
        cloud_strategy="rl",
        rl_scheduler=str(model_file),
        cloud_interval=10,
        cloud_latency=0,
        alpha_steer=0.0,
        alpha_speed=0.0,
        lookahead=1.0,
        lateral_gain=1.0,
        edge_left_wall_model="",
        edge_track_width_model="",
        edge_heading_model="",
        cloud_left_wall_model="",
        cloud_track_width_model="",
        cloud_heading_model="",
        speed=None,
    )
    planner = _create_planner(args, waypoints=np.zeros((0, 2)))
    assert isinstance(planner.scheduler, PolicyScheduler)

    # also verify that the scheduler can be invoked with a bare observation
    # dictionary (no extra RL keys) without raising
    sched = planner.scheduler
    obs = {"poses_x": np.zeros(1), "poses_y": np.zeros(1)}
    assert isinstance(sched.should_call_cloud(0, obs, None), bool)


def test_no_rl_scheduler_default(tmp_path, monkeypatch):
    """Explicit None still yields fixed-interval scheduler."""
    monkeypatch.setattr("f110_scripts.sim.reactive_planners.PPO", DummyModel)
    args = Namespace(
        planner="edge_cloud",
        cloud_strategy="rl",
        rl_scheduler=None,
        cloud_interval=4,
        cloud_latency=0,
        alpha_steer=0.0,
        alpha_speed=0.0,
        lookahead=1.0,
        lateral_gain=1.1,
        edge_left_wall_model="",
        edge_track_width_model="",
        edge_heading_model="",
        cloud_left_wall_model="",
        cloud_track_width_model="",
        cloud_heading_model="",
        speed=None,
    )
    planner = _create_planner(args, waypoints=np.zeros((0, 2)))
    assert isinstance(planner.scheduler, FixedIntervalScheduler)


def test_fixed_interval_strategy(tmp_path, monkeypatch):
    """Explicit interval strategy should give fixed-interval regardless of rl file."""
    missing = tmp_path / "nope.zip"
    monkeypatch.setattr("f110_scripts.sim.reactive_planners.PPO", DummyModel)

    args = Namespace(
        planner="edge_cloud",
        rl_scheduler=str(missing),
        cloud_strategy="interval",
        cloud_interval=7,
        cloud_latency=0,
        alpha_steer=0.0,
        alpha_speed=0.0,
        lookahead=1.0,
        lateral_gain=1.1,
        edge_left_wall_model="",
        edge_track_width_model="",
        edge_heading_model="",
        cloud_left_wall_model="",
        cloud_track_width_model="",
        cloud_heading_model="",
        speed=None,
    )
    planner = _create_planner(args, waypoints=np.zeros((0, 2)))
    assert isinstance(planner.scheduler, FixedIntervalScheduler)


def test_fallback_to_fixed_interval_when_model_missing(tmp_path, monkeypatch):
    """Missing scheduler file should default to fixed‑interval."""
    missing = tmp_path / "nope.zip"
    monkeypatch.setattr("f110_scripts.sim.reactive_planners.PPO", DummyModel)

    args = Namespace(
        planner="edge_cloud",
        cloud_strategy="rl",
        rl_scheduler=str(missing),
        cloud_interval=7,
        cloud_latency=0,
        alpha_steer=0.0,
        alpha_speed=0.0,
        lookahead=1.0,
        lateral_gain=1.1,
        edge_left_wall_model="",
        edge_track_width_model="",
        edge_heading_model="",
        cloud_left_wall_model="",
        cloud_track_width_model="",
        cloud_heading_model="",
        speed=None,
    )
    planner = _create_planner(args, waypoints=np.zeros((0, 2)))
    assert isinstance(planner.scheduler, FixedIntervalScheduler)


def test_render_callback_attached(monkeypatch):
    """When building an edge-cloud planner the cloud-call renderer is added."""
    # create dummy env with minimal interface
    class DummyEnv:
        def __init__(self):
            self.unwrapped = self
            self.render_callbacks = []

        def add_render_callback(self, cb):
            self.render_callbacks.append(cb)

    dummy = DummyEnv()
    # build planner as before but ignore scheduler type
    args = Namespace(
        planner="edge_cloud",
        cloud_strategy="interval",
        rl_scheduler=str(Path("doesnotmatter.zip")),
        cloud_interval=1,
        cloud_latency=0,
        alpha_steer=0,
        alpha_speed=0,
        lookahead=0,
        lateral_gain=0,
        edge_left_wall_model="",
        edge_track_width_model="",
        edge_heading_model="",
        cloud_left_wall_model="",
        cloud_track_width_model="",
        cloud_heading_model="",
        speed=None,
    )
    planner = _create_planner(args, waypoints=np.zeros((0, 2)))
    # call setup rendering
    from f110_scripts.sim.reactive_planners import _setup_rendering
    _setup_rendering(dummy, args, np.zeros((0, 2)), planner)
    # ensure at least one callback named render_cloud exists
    names = [cb.__name__ for cb in dummy.render_callbacks]
    assert any("render_cloud" in n for n in names), f"callbacks={names}"
