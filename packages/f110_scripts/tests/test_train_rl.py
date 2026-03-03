"""Tests for the RL training script argument parser and environment builder."""

import sys
import numpy as np
import pytest


def _parse(extra_args=None, monkeypatch=None):
    """Call parse_args() after injecting synthetic sys.argv."""
    from f110_scripts.train.train_rl import parse_args

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
    # Defaults should be sane positive integers
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
