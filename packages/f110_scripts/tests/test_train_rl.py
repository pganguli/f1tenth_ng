"""Tests for the RL training script argument parser and environment builder."""

import sys

from f110_scripts.train.train_rl import parse_args


def _parse(extra_args=None, monkeypatch=None):
    """Call parse_args() after injecting synthetic sys.argv."""
    # --map and --waypoints now accept nargs='+', so single values still work.
    argv = ["prog", "--map", "some/map", "--waypoints", "some/waypoints.tsv"]
    if extra_args:
        argv.extend(extra_args)
    if monkeypatch is not None:
        monkeypatch.setattr(sys, "argv", argv)
    else:
        old = sys.argv
        sys.argv = argv
        try:
            return parse_args()
        finally:
            sys.argv = old
    return parse_args()


def test_parse_args_defaults(monkeypatch):
    """parse_args() should expose all expected keys with sensible defaults."""
    args = _parse(monkeypatch=monkeypatch)
    assert hasattr(args, "cloud_latency")
    assert hasattr(args, "timesteps")
    assert hasattr(args, "save_path")
    assert args.cloud_latency > 0
    assert args.timesteps > 0


def test_parse_args_cloud_latency_override(monkeypatch):
    """User-supplied --cloud-latency should be respected."""
    args = _parse(["--cloud-latency", "42"], monkeypatch=monkeypatch)
    assert args.cloud_latency == 42


def test_parse_args_timesteps_override(monkeypatch):
    """User-supplied --timesteps should be respected."""
    args = _parse(["--timesteps", "5000"], monkeypatch=monkeypatch)
    assert args.timesteps == 5000


# ---------------------------------------------------------------------------
# New-feature tests
# ---------------------------------------------------------------------------

def test_map_and_waypoints_produce_lists(monkeypatch):
    """--map and --waypoints (nargs='+') should be stored as lists."""
    args = _parse(monkeypatch=monkeypatch)
    assert isinstance(args.map, list)
    assert isinstance(args.waypoints, list)
    assert args.map == ["some/map"]
    assert args.waypoints == ["some/waypoints.tsv"]


def test_multi_map_accepted(monkeypatch):
    """Multiple --map / --waypoints values should produce equal-length lists."""
    args = _parse(
        ["--map", "map_a", "map_b", "--waypoints", "wpts_a.tsv", "wpts_b.tsv"],
        monkeypatch=monkeypatch,
    )
    # nargs='+' but the extra --map/--waypoints replace the base argv entries
    # because they appear after the base; argparse accumulates all positional
    # tokens for the last occurrence of each nargs='+' flag.
    assert len(args.map) == len(args.waypoints)


def test_n_envs_default_is_one(monkeypatch):
    args = _parse(monkeypatch=monkeypatch)
    assert args.n_envs == 1


def test_n_envs_override(monkeypatch):
    args = _parse(["--n-envs", "8"], monkeypatch=monkeypatch)
    assert args.n_envs == 8


def test_device_default_is_auto(monkeypatch):
    args = _parse(monkeypatch=monkeypatch)
    assert args.device == "auto"


def test_device_cpu_accepted(monkeypatch):
    args = _parse(["--device", "cpu"], monkeypatch=monkeypatch)
    assert args.device == "cpu"


def test_resume_default_is_none(monkeypatch):
    args = _parse(monkeypatch=monkeypatch)
    assert args.resume is None


def test_resume_path_stored(monkeypatch):
    args = _parse(["--resume", "some/checkpoint.zip"], monkeypatch=monkeypatch)
    assert args.resume == "some/checkpoint.zip"


def test_hyperparameter_defaults_are_set(monkeypatch):
    """All hyperparameters must have explicit non-None defaults for reproducible runs."""
    args = _parse(monkeypatch=monkeypatch)
    expected = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2,
    }
    for attr, expected_val in expected.items():
        actual = getattr(args, attr)
        assert actual is not None, f"{attr} should not be None"
        assert abs(actual - expected_val) < 1e-9, (
            f"{attr}: expected {expected_val}, got {actual}"
        )


def test_learning_rate_override(monkeypatch):
    args = _parse(["--lr", "1e-3"], monkeypatch=monkeypatch)
    assert abs(args.learning_rate - 1e-3) < 1e-10


def test_eval_freq_default_zero(monkeypatch):
    args = _parse(monkeypatch=monkeypatch)
    assert args.eval_freq == 0


def test_progress_bar_flag(monkeypatch):
    args = _parse(["--progress-bar"], monkeypatch=monkeypatch)
    assert args.progress_bar is True

