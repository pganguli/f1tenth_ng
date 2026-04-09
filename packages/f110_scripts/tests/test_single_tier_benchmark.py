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
        "fixed_interval",
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
    assert args.age_decay_lambda == 0.0
    assert args.workers == 1


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
    assert args.sigma_proc_left is None
    assert args.sigma_proc_track is None
    assert args.sigma_proc_heading is None
    assert args.age_decay_lambda == 0.0


def test_oracle_nondefault_setup_uses_generic_reference_label(monkeypatch) -> None:
    """Non-canonical reference anchors should not pretend to be the full-cloud default."""
    module = _load_oracle_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_single_tier_oracle_baseline.py",
            "--cloud-latency",
            "5",
            "--alpha-left",
            "0.0",
        ],
    )

    args = module.parse_args()
    experiment_name = (
        "zero_latency_full_cloud_reference"
        if args.cloud_latency == 0
        and args.alpha_left == 1.0
        and args.alpha_track == 1.0
        and args.alpha_heading == 1.0
        else "reference_anchor"
    )
    assert experiment_name == "reference_anchor"


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
    assert settings.age_decay_lambda == 0.0

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


def test_summarize_adds_uncertainty_columns() -> None:
    """Benchmark summaries should expose std/stderr/CI fields for key metrics."""
    module = _load_benchmark_module()
    results = [
        {
            "map_name": "Monza",
            "cloud_latency": 5.0,
            "experiment": "always_hit",
            "strategy": "always",
            "run_idx": float(run_idx),
            "collision": 0.0,
            "step_cap_hit": 0.0,
            "collision_count": 0.0,
            "collision_steps": 0.0,
            "laps_completed": 1.0,
            "lap_time_s": 1.0 + 0.1 * run_idx,
            "crosstrack_rmse_m": 0.1 + 0.01 * run_idx,
            "crosstrack_mean_m": 0.09,
            "crosstrack_max_m": 0.12,
            "heading_error_rmse_deg": 1.0,
            "wall_min_distance_m": 0.5,
            "speed_mean_m_s": 1.0,
            "steering_rate_mean_rad_s": 0.1,
            "total_cloud_calls": 10.0,
            "cloud_call_rate": 0.6 + 0.05 * run_idx,
            "mean_call_gap_steps": 1.0,
            "max_call_gap_steps": 2.0,
            "rt_factor": 1.0,
        }
        for run_idx in range(3)
    ]

    _, summary_df, _ = module.summarize(results)
    row = summary_df.iloc[0]

    assert row["trials"] == 3
    assert row["lap_time_s_std"] > 0.0
    assert row["lap_time_s_stderr"] > 0.0
    assert row["lap_time_s_ci95"] > 0.0
    assert row["crosstrack_rmse_m_stderr"] > 0.0
    assert row["crosstrack_max_m_ci95"] == 0.0
    assert row["cloud_call_rate_ci95"] > 0.0
    assert row["collision_free_rate_ci95"] == 0.0


def test_build_planner_honors_experiment_level_age_decay_override(monkeypatch) -> None:
    """Planner construction should allow age_decay_lambda to be overridden per experiment."""
    module = _load_benchmark_module()
    captured: dict[str, float] = {}

    class DummyPlanner:  # pylint: disable=too-few-public-methods
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "EdgeCloudPlanner", DummyPlanner)

    exp = module.Experiment(
        "self_norm_tau1p0_n3__lambda_1p0",
        "self_normalizing_momentum",
        {"tau": 1.0, "staleness_multiplier": 3, "age_decay_lambda": 1.0},
    )
    settings = module.PlannerSettings(age_decay_lambda=24.0)

    module.build_planner(exp, cloud_latency=5, settings=settings)

    assert captured["age_decay_lambda"] == 1.0


def test_build_planner_supports_dual_signal_periodic(monkeypatch) -> None:
    """Planner construction should instantiate the dual-signal scheduler family."""
    module = _load_benchmark_module()
    captured: dict[str, object] = {}

    class DummyPlanner:  # pylint: disable=too-few-public-methods
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "EdgeCloudPlanner", DummyPlanner)

    exp = module.Experiment(
        "dual_signal_k3_bt70_tau1p0_devheavy__lambda_4p0",
        "dual_signal_periodic",
        {
            "base_interval": 3,
            "burst_threshold": 0.7,
            "tau": 1.0,
            "age_weight": 0.2,
            "deviation_weight": 0.6,
            "momentum_weight": 0.2,
            "deviation_cap": 0.10,
            "age_horizon_multiplier": 2,
            "force_age_multiplier": 3,
            "min_extra_gap": 1,
            "burst_queue_cap": 1,
            "age_decay_lambda": 4.0,
        },
    )

    module.build_planner(exp, cloud_latency=5, settings=module.PlannerSettings())

    assert captured["age_decay_lambda"] == 4.0
    assert isinstance(captured["scheduler"], module.DualSignalPeriodicScheduler)


def test_build_planner_supports_srpv2_and_never_query(monkeypatch) -> None:
    """Planner construction should instantiate the SRPv2 and never-query controls."""
    module = _load_benchmark_module()
    captured: list[dict[str, object]] = []

    class DummyPlanner:  # pylint: disable=too-few-public-methods
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(module, "EdgeCloudPlanner", DummyPlanner)

    module.build_planner(
        module.Experiment(
            "srpv2_tau1p0_n3__lambda_4p0",
            "srpv2",
            {"tau": 1.0, "nmax": 3, "age_decay_lambda": 4.0, "seed": 7},
        ),
        cloud_latency=10,
        settings=module.PlannerSettings(),
    )
    module.build_planner(
        module.Experiment(
            "never_query__lambda_0p0",
            "never_query",
            {"age_decay_lambda": 0.0},
        ),
        cloud_latency=10,
        settings=module.PlannerSettings(),
    )

    assert isinstance(captured[0]["scheduler"], module.ShiftResponsePolicyScheduler)
    assert captured[0]["scheduler"].config.causal_feature_scales
    assert captured[0]["scheduler"].config.causal_baseline
    assert captured[0]["age_decay_lambda"] == 4.0
    assert isinstance(captured[1]["scheduler"], module.NeverCloudScheduler)


def test_run_episode_records_scheduler_debug_counts_when_available(monkeypatch) -> None:
    """Benchmark episodes should surface scheduler debug counters when exposed."""
    module = _load_benchmark_module()

    class DummyScheduler:  # pylint: disable=too-few-public-methods
        def debug_state(self):
            return {
                "call_reason_counts": {
                    "bootstrap": 1,
                    "backbone": 2,
                    "burst": 3,
                    "force_age": 0,
                    "none": 4,
                }
            }

    class DummyPlanner:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self.last_cloud_call = False
            self.scheduler = DummyScheduler()

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

    exp = module.Experiment("dual_signal_demo", "dual_signal_periodic", {"base_interval": 3})
    track = module.TrackConfig("Oschersleben", "map_path", "waypoints_path")
    settings = module.PlannerSettings()

    result = module.run_episode(
        exp,
        track,
        cloud_latency=5,
        max_laps=1,
        settings=settings,
        run_idx=0,
    )

    assert result["scheduler_calls_bootstrap"] == 1.0
    assert result["scheduler_calls_backbone"] == 2.0
    assert result["scheduler_calls_burst"] == 3.0
    assert result["scheduler_calls_total"] == 6.0
    assert result["scheduler_burst_fraction"] == 0.5


def test_summarize_can_rank_by_max_cte() -> None:
    """Selecting max-CTE ranking should prioritize lower max CTE over lower RMSE."""
    module = _load_benchmark_module()
    results = [
        {
            "map_name": "Spa",
            "cloud_latency": 5.0,
            "experiment": "lower_rmse",
            "strategy": "fixed_bernoulli",
            "run_idx": 0.0,
            "collision": 0.0,
            "step_cap_hit": 0.0,
            "collision_count": 0.0,
            "collision_steps": 0.0,
            "laps_completed": 1.0,
            "lap_time_s": 1.0,
            "crosstrack_rmse_m": 0.04,
            "crosstrack_mean_m": 0.03,
            "crosstrack_max_m": 0.20,
            "heading_error_rmse_deg": 1.0,
            "wall_min_distance_m": 0.5,
            "speed_mean_m_s": 1.0,
            "steering_rate_mean_rad_s": 0.1,
            "total_cloud_calls": 10.0,
            "cloud_call_rate": 0.60,
            "mean_call_gap_steps": 1.0,
            "max_call_gap_steps": 2.0,
            "rt_factor": 1.0,
        },
        {
            "map_name": "Spa",
            "cloud_latency": 5.0,
            "experiment": "lower_max_cte",
            "strategy": "fixed_bernoulli",
            "run_idx": 0.0,
            "collision": 0.0,
            "step_cap_hit": 0.0,
            "collision_count": 0.0,
            "collision_steps": 0.0,
            "laps_completed": 1.0,
            "lap_time_s": 1.0,
            "crosstrack_rmse_m": 0.05,
            "crosstrack_mean_m": 0.03,
            "crosstrack_max_m": 0.10,
            "heading_error_rmse_deg": 1.0,
            "wall_min_distance_m": 0.5,
            "speed_mean_m_s": 1.0,
            "steering_rate_mean_rad_s": 0.1,
            "total_cloud_calls": 10.0,
            "cloud_call_rate": 0.60,
            "mean_call_gap_steps": 1.0,
            "max_call_gap_steps": 2.0,
            "rt_factor": 1.0,
        },
    ]

    _, summary_df, _ = module.summarize(results, selection_metric="max_cte")

    assert summary_df.iloc[0]["experiment"] == "lower_max_cte"


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
