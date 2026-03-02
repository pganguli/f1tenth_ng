"""Tests for tuning script command-line behaviour."""

import builtins
import sys

from f110_scripts.tune import tune_alphas


def run_main_with_args(monkeypatch, args_list):
    """Invoke main() with a modified argv and capture its parallel arg."""
    old_argv = sys.argv.copy()
    sys.argv = [old_argv[0]] + args_list
    called = {}

    def fake_search(*_args, **kwargs):
        called.update(kwargs)
        return 0.0, 0.0, 0.0, False

    # stub out the expensive parts of ``main``. previous versions of this
    # test only replaced ``coarse_to_fine_search`` but the final sanity loop
    # in ``main`` still invoked ``evaluate_runner`` which in turn executed a
    # full simulation and triggered the cross-track error failure we saw
    # earlier. patching both functions keeps the test fast and deterministic.
    monkeypatch.setattr(tune_alphas, "coarse_to_fine_search", fake_search)
    monkeypatch.setattr(tune_alphas, "evaluate_runner", lambda *a, **k: lambda *_: (0.0, True))

    # suppress prints
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    try:
        tune_alphas.main()
    finally:
        sys.argv = old_argv
    return called.get("parallel")


def test_parallel_flag(monkeypatch):
    """Check that the --parallel/--no-parallel flags behave correctly."""
    # default (no render flag) should enable parallel
    p = run_main_with_args(monkeypatch, [])
    assert p is True

    # explicit no-parallel disables
    p = run_main_with_args(monkeypatch, ["--no-parallel"])
    assert p is False

    # even if parallel requested, a render mode turns it off
    p = run_main_with_args(monkeypatch, ["--parallel", "--render-mode", "human"])
    assert p is False

    # explicit parallel with no render-mode remains true
    p = run_main_with_args(monkeypatch, ["--parallel"])
    assert p is True
