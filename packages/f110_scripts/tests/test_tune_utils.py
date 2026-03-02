"""Unit tests for the tuning utilities."""

import concurrent.futures
from f110_scripts.tune.tune_utils import coarse_to_fine_search


def test_coarse_to_fine_parallel_equivalence():
    """Sequential and parallel searches yield same result."""

    def eval_fn(s, v):
        return (s * s + v * v, False)

    out_seq = coarse_to_fine_search(
        eval_fn, coarse_grid_size=3, fine_grid_size=3, verbose=False, parallel=False
    )
    out_par = coarse_to_fine_search(
        eval_fn, coarse_grid_size=3, fine_grid_size=3, verbose=False, parallel=True
    )
    assert out_seq == out_par


def test_parallel_threads_are_used(monkeypatch):
    """Parallel flag actually triggers ThreadPoolExecutor."""

    used = {"called": False}

    # pylint: disable=too-few-public-methods
    class DummyExecutor(concurrent.futures.ThreadPoolExecutor):
        """ThreadPoolExecutor subclass that records entry."""
        def __enter__(self):
            used["called"] = True
            return super().__enter__()

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", DummyExecutor)

    def eval_fn(s, v):
        return s + v

    coarse_to_fine_search(
        eval_fn, coarse_grid_size=2, fine_grid_size=2, verbose=False, parallel=True
    )
    assert used["called"], "ThreadPoolExecutor not invoked"


def test_crash_flag_propagation():
    """Search should detect at least one crash-free point if provided."""

    def eval_fn(s, v):
        return (0.0, True) if s == 0.0 and v == 0.0 else (1.0, False)

    _, _, _, any_crash = coarse_to_fine_search(
        eval_fn, coarse_grid_size=3, fine_grid_size=3, verbose=False, parallel=False
    )
    assert any_crash


def test_steer_speed_ranges_respected():
    """Grid samples never escape user-specified bounds."""
    calls = []

    def eval_fn(s, v):
        calls.append((s, v))
        return (0.0, False)

    coarse_to_fine_search(
        eval_fn,
        coarse_grid_size=2,
        fine_grid_size=1,
        verbose=False,
        parallel=False,
        steer_min=0.2,
        steer_max=0.4,
        speed_min=0.3,
        speed_max=0.5,
    )
    assert all(0.2 <= s <= 0.4 for s, _ in calls)
    assert all(0.3 <= v <= 0.5 for _, v in calls)
