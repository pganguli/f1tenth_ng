"""Smoke tests for the single-tier paper-strategy benchmark script."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

from f110_planning.base import Action


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[3]
    script_path = root / "scripts/benchmarks/benchmark_single_tier_paper_strategies.py"
    spec = spec_from_file_location("benchmark_single_tier_paper_strategies", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_oracle_module():
    root = Path(__file__).resolve().parents[3]
    script_path = root / "scripts/benchmarks/benchmark_single_tier_oracle_baseline.py"
    spec = spec_from_file_location("benchmark_single_tier_oracle_baseline", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_experiments_cover_paper_strategies() -> None:
    """Default benchmark sweep should cover all paper strategy families."""
    module = _load_benchmark_module()

    strategies = {exp.strategy for exp in module.default_experiments()}

    assert strategies == {
        "always",
        "fixed_bernoulli",
        "bernoulli_max_miss",
        "logistic",
        "exponential",
        "piecewise_ramp",
        "deterministic",
    }


def test_canonical_defaults_target_corrected_10_map_workflow(monkeypatch) -> None:
    """The canonical single-tier benchmark defaults should match the 10-map workflow."""
    module = _load_benchmark_module()
    monkeypatch.setattr(sys, "argv", ["benchmark_single_tier_paper_strategies.py"])

    args = module.parse_args()

    assert args.output_stem == "single_tier_paper_strategies_10maps"
    assert args.cloud_latency == module.DEFAULT_CLOUD_LATENCY == 5
    assert args.maps.split(",") == module.DEFAULT_MAPS


def test_oracle_defaults_use_zero_latency_full_cloud_reference(monkeypatch) -> None:
    """The optional reference benchmark should default to a full-cloud latency-0 setup."""
    module = _load_oracle_module()
    monkeypatch.setattr(sys, "argv", ["benchmark_single_tier_oracle_baseline.py"])

    args = module.parse_args()

    assert args.output_stem == "single_tier_oracle_baseline_10maps"
    assert args.cloud_latency == 0
    assert args.alpha_left == 1.0
    assert args.alpha_track == 1.0
    assert args.alpha_heading == 1.0


def test_run_episode_reports_scheduler_metrics(monkeypatch) -> None:
    """Benchmark smoke run should emit cloud-call and collision summary fields."""
    module = _load_benchmark_module()

    class DummyPlanner:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self.last_cloud_call = False

        def plan(self, obs, ego_idx=0):
            del obs, ego_idx
            self.last_cloud_call = not self.last_cloud_call
            return Action(steer=0.0, speed=1.0)

    class DummyEnv:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self._step = 0

        def reset(self, options=None):
            del options
            obs = {
                "poses_x": np.zeros(1),
                "poses_y": np.zeros(1),
                "poses_theta": np.zeros(1),
                "linear_vels_x": np.ones(1),
                "collisions": np.zeros(1),
                "lap_counts": np.zeros(1),
                "scans": np.ones((1, 1080)),
            }
            return obs, {}

        def step(self, action):
            del action
            self._step += 1
            obs = {
                "poses_x": np.zeros(1),
                "poses_y": np.zeros(1),
                "poses_theta": np.zeros(1),
                "linear_vels_x": np.ones(1),
                "collisions": np.array([1.0 if self._step > 1 else 0.0]),
                "lap_counts": np.array([1.0 if self._step > 1 else 0.0]),
                "scans": np.ones((1, 1080)),
            }
            return obs, 0.01, self._step > 1, False, {}

        def close(self):
            """No-op close for the dummy environment."""

    class DummyAggregator:  # pylint: disable=too-few-public-methods
        def on_reset(self, obs, waypoints=None):
            del obs, waypoints

        def on_step(self, obs, action, reward, ego_idx=0):
            del obs, action, reward, ego_idx

        def report(self):
            return {
                "lap_time_s": 0.02,
                "steps": 2.0,
                "collision": 1.0,
                "laps_completed": 1.0,
                "crosstrack_rmse_m": 0.12,
                "crosstrack_mean_m": 0.11,
                "crosstrack_max_m": 0.15,
                "heading_error_rmse_deg": 2.0,
                "wall_min_distance_m": 0.6,
                "speed_mean_m_s": 1.0,
                "steering_rate_mean_rad_s": 0.2,
            }

    monkeypatch.setattr(
        module,
        "load_waypoints",
        lambda _path: np.array([[0.0, 0.0], [1.0, 0.0]]),
    )
    monkeypatch.setattr(module, "build_planner", lambda *args, **kwargs: DummyPlanner())
    monkeypatch.setattr(module, "make_env", lambda *args, **kwargs: DummyEnv())
    monkeypatch.setattr(
        module.MetricAggregator,
        "create_default",
        lambda waypoints=None: DummyAggregator(),
    )

    exp = module.Experiment("fixed_bernoulli_p60", "fixed_bernoulli", {"p": 0.6, "seed": 7})
    track = module.TrackConfig("Oschersleben", "map_path", "waypoints_path")
    settings = module.PlannerSettings()

    result = module.run_episode(
        exp,
        track,
        cloud_latency=10,
        max_laps=1,
        settings=settings,
        run_idx=0,
    )

    assert result["total_cloud_calls"] == 1.0
    assert result["cloud_call_rate"] == 0.5
    assert result["mean_call_gap_steps"] == 0.0
    assert result["max_call_gap_steps"] == 0.0
    assert result["collision_count"] == 1.0


def test_run_episode_uses_track_start_pose(monkeypatch) -> None:
    """Benchmark episodes should reset the env with the track YAML start pose."""
    module = _load_benchmark_module()

    class DummyPlanner:  # pylint: disable=too-few-public-methods
        last_cloud_call = False

        def plan(self, obs, ego_idx=0):
            del obs, ego_idx
            return Action(steer=0.0, speed=1.0)

    class DummyEnv:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self.reset_options = None

        def reset(self, options=None):
            self.reset_options = options
            obs = {
                "poses_x": np.zeros(1),
                "poses_y": np.zeros(1),
                "poses_theta": np.zeros(1),
                "linear_vels_x": np.ones(1),
                "collisions": np.zeros(1),
                "lap_counts": np.zeros(1),
                "scans": np.ones((1, 1080)),
            }
            return obs, {}

        def step(self, action):
            del action
            obs = {
                "poses_x": np.zeros(1),
                "poses_y": np.zeros(1),
                "poses_theta": np.zeros(1),
                "linear_vels_x": np.ones(1),
                "collisions": np.array([1.0]),
                "lap_counts": np.array([1.0]),
                "scans": np.ones((1, 1080)),
            }
            return obs, 0.01, True, False, {}

        def close(self):
            """No-op close for the dummy environment."""

    class DummyAggregator:  # pylint: disable=too-few-public-methods
        def on_reset(self, obs, waypoints=None):
            del obs, waypoints

        def on_step(self, obs, action, reward, ego_idx=0):
            del obs, action, reward, ego_idx

        def report(self):
            return {
                "lap_time_s": 0.01,
                "steps": 1.0,
                "collision": 1.0,
                "laps_completed": 1.0,
                "crosstrack_rmse_m": 0.12,
                "crosstrack_mean_m": 0.11,
                "crosstrack_max_m": 0.15,
                "heading_error_rmse_deg": 2.0,
                "wall_min_distance_m": 0.6,
                "speed_mean_m_s": 1.0,
                "steering_rate_mean_rad_s": 0.2,
            }

    env = DummyEnv()
    monkeypatch.setattr(
        module,
        "load_waypoints",
        lambda _path: np.array([[0.0, 0.0], [1.0, 0.0]]),
    )
    monkeypatch.setattr(module, "build_planner", lambda *args, **kwargs: DummyPlanner())
    monkeypatch.setattr(module, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        module.MetricAggregator,
        "create_default",
        lambda waypoints=None: DummyAggregator(),
    )
    monkeypatch.setattr(
        module,
        "load_start_pose_from_yaml",
        lambda _map_path: (0.0, 0.0, 1.472932),
    )

    exp = module.Experiment("always_hit", "always", {})
    track = module.TrackConfig("Monza", "map_path", "waypoints_path")
    settings = module.PlannerSettings()

    module.run_episode(
        exp,
        track,
        cloud_latency=5,
        max_laps=1,
        settings=settings,
        run_idx=0,
    )

    assert env.reset_options is not None
    assert np.allclose(env.reset_options["poses"], np.array([[0.0, 0.0, 1.472932]]))
