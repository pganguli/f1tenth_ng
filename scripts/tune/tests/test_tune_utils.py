import math
import numpy as np
from scripts.tune.tune_utils import coarse_to_fine_search


def test_coarse_to_fine_parallel_equivalence():
    """Results should be identical whether parallelisation is enabled or not."""

    # simple evaluation: score = steer^2 + speed^2 (min at 0,0)
    def eval_fn(s, v):
        # return a tuple to test handling of crash flag
        return (s * s + v * v, False)

    out_seq = coarse_to_fine_search(eval_fn, coarse_grid_size=3, fine_grid_size=3, verbose=False, parallel=False)
    out_par = coarse_to_fine_search(eval_fn, coarse_grid_size=3, fine_grid_size=3, verbose=False, parallel=True)
    assert out_seq == out_par


def test_parallel_threads_are_used(monkeypatch):
    """Make sure that when parallel=True the executor actually runs multiple tasks.
    We do this by patching ThreadPoolExecutor to record whether it was constructed.
    """
    import concurrent.futures

    used = {"called": False}

    class DummyExecutor(concurrent.futures.ThreadPoolExecutor):
        def __enter__(self):
            used["called"] = True
            return super().__enter__()

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", DummyExecutor)

    def eval_fn(s, v):
        return s + v  # doesn't matter

    # run with a few candidates to trigger executor creation
    coarse_to_fine_search(eval_fn, coarse_grid_size=2, fine_grid_size=2, verbose=False, parallel=True)
    assert used["called"], "ThreadPoolExecutor was not invoked when parallel=True"


def test_crash_flag_propagation():
    """When the evaluation returns (score, crash_free) the search should
    report a crash_free run and include it in the returned flag."""


def test_steer_speed_ranges_respected():
    """Search grids should only sample within the user-specified ranges."""
    calls = []

    def eval_fn(s, v):
        calls.append((s, v))
        return (0.0, False)

    # coarse grid size 2 will sample exactly at min and max boundaries
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

    # define a grid where one point is crash-free
    def eval_fn(s, v):
        if math.isclose(s, 0.0) and math.isclose(v, 0.0):
            return (0.0, True)  # best configuration, crash_free
        return (1.0, False)

    steer, speed, score, any_crash = coarse_to_fine_search(
        eval_fn, coarse_grid_size=3, fine_grid_size=3, verbose=False, parallel=False
    )

    assert any_crash is True
    assert math.isclose(steer, 0.0) and math.isclose(speed, 0.0)
    assert math.isclose(score, 0.0)
