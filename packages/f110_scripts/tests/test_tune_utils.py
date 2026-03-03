"""Unit tests for the tuning utilities."""

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


def test_coarse_to_fine_flat_objective():
    """When all evaluations return the same score the result is deterministic
    (first grid mid-point wins when there are no better alternatives)."""

    def eval_fn(s, v):
        return (0.0, False)

    out1 = coarse_to_fine_search(
        eval_fn, coarse_grid_size=3, fine_grid_size=3, verbose=False, parallel=False
    )
    out2 = coarse_to_fine_search(
        eval_fn, coarse_grid_size=3, fine_grid_size=3, verbose=False, parallel=False
    )
    # Repeated calls must yield the same best values
    assert out1[0] == out2[0], "best_steer should be deterministic for flat objective"
    assert out1[1] == out2[1], "best_speed should be deterministic for flat objective"
