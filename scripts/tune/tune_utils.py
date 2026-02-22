"""
Utilities for systematically tuning hyperparameters.
"""

import concurrent.futures
import math
from typing import Callable, Tuple

import numpy as np


def coarse_to_fine_search(
    evaluate_fn: Callable[[float, float], float],
    coarse_grid_size: int = 3,
    fine_grid_size: int = 3,
    verbose: bool = True,
    parallel: bool = True,
    steer_min: float = 0.0,
    steer_max: float = 1.0,
    speed_min: float = 0.0,
    speed_max: float = 1.0,
) -> Tuple[float, float, float, bool]:
    """Coarse-to-fine grid search with optional parallelism.

    many parameters allow restricting search ranges.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,too-many-branches,too-many-statements

    # coarse grid
    if coarse_grid_size > 1:
        steer_grid = list(np.linspace(steer_min, steer_max, coarse_grid_size))
        speed_grid = list(np.linspace(speed_min, speed_max, coarse_grid_size))
    else:
        # use midpoints of provided ranges
        steer_grid = [(steer_min + steer_max) / 2.0]
        speed_grid = [(speed_min + speed_max) / 2.0]

    # combine into candidate pairs
    coarse_pairs = [(s, v) for s in steer_grid for v in speed_grid]

    best_score = float("inf")
    best_steer = 0.5
    best_speed = 0.5
    any_crash_free = False

    if verbose:
        print("--- Starting Coarse Grid Search ---")

    # coarse_pairs already computed above from steer_grid/speed_grid

    # evaluate all candidates, possibly concurrently
    if parallel and len(coarse_pairs) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            coarse_results = list(executor.map(lambda p: evaluate_fn(*p), coarse_pairs))
    else:
        coarse_results = [evaluate_fn(s, v) for s, v in coarse_pairs]

    for (steer, speed), result in zip(coarse_pairs, coarse_results):
        if isinstance(result, tuple) and len(result) == 2:
            score, crash_free = result
        else:
            score = result
            crash_free = False
        if verbose:
            print(
                f"  [Coarse] {steer=:.2f} {speed=:.2f} | {score=:.2f} {crash_free=}"
            )
        if crash_free:
            any_crash_free = True
        if score < best_score:
            best_score = score
            best_steer = steer
            best_speed = speed

    if verbose:
        print(
            f"coarse winner: {best_steer:.2f},{best_speed:.2f} score={best_score:.2f}"
        )

    # fine grid
    if coarse_grid_size > 1:
        # radius based on spacing of coarse grid within provided steer range
        fine_search_radius = (steer_max - steer_min) / (coarse_grid_size - 1) / 2.0
    else:
        fine_search_radius = (steer_max - steer_min) / 5.0

    # candidate bounds must obey the user-specified ranges
    if fine_grid_size > 1:
        steer_cands = list(
            np.linspace(
                max(steer_min, best_steer - fine_search_radius),
                min(steer_max, best_steer + fine_search_radius),
                fine_grid_size,
            )
        )
        speed_cands = list(
            np.linspace(
                max(speed_min, best_speed - fine_search_radius),
                min(speed_max, best_speed + fine_search_radius),
                fine_grid_size,
            )
        )
    else:
        steer_cands, speed_cands = [best_steer], [best_speed]

    # Remove duplicates and round to avoid float precision issues
    steer_cands = sorted({round(x, 4) for x in steer_cands})
    speed_cands = sorted({round(x, 4) for x in speed_cands})

    if verbose:
        print("fine search")

    # prepare candidate list skipping center
    fine_pairs = []
    for steer in steer_cands:
        for speed in speed_cands:
            if math.isclose(steer, best_steer, abs_tol=1e-5) and math.isclose(
                speed, best_speed, abs_tol=1e-5
            ):
                continue
            fine_pairs.append((steer, speed))

    if parallel and len(fine_pairs) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fine_results = list(executor.map(lambda p: evaluate_fn(*p), fine_pairs))
    else:
        fine_results = [evaluate_fn(s, v) for s, v in fine_pairs]

    for (steer, speed), result in zip(fine_pairs, fine_results):
        if isinstance(result, tuple) and len(result) == 2:
            score, crash_free = result
        else:
            score = result
            crash_free = False
        if verbose:
            print(
                f"  fine {steer:.2f},{speed:.2f} score={score:.2f} crash={crash_free}"
            )
        if crash_free:
            any_crash_free = True
        if score < best_score:
            best_score = score
            best_steer = steer
            best_speed = speed

    if verbose:
        print("optimal", best_steer, best_speed, best_score)

    return best_steer, best_speed, best_score, any_crash_free
