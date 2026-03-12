"""Estimate per-feature process-noise standard deviations from dataset files.

For each of the three DNN output features (left_wall_dist, track_width,
heading_error) the process noise σ_proc is estimated as the standard deviation
of successive first differences across all trajectories in
``data/datasets/*.npz``.

Under a random-walk model  y(t) = y(t-1) + ε(t),  σ_proc = std(ε(t)).  Feed
the printed values into SelectiveEdgeCloudPlanner as ``sigma_proc_left``,
``sigma_proc_track``, and ``sigma_proc_heading`` to enable age-dependent alpha
blending.

Usage
-----
    python packages/f110_scripts/src/f110_scripts/tune/estimate_sigma_proc.py
    python packages/f110_scripts/src/f110_scripts/tune/estimate_sigma_proc.py \\
        --data-dir data/datasets  --glob "*.npz"
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np


# Keys expected inside each .npz file.  They must contain 1-D arrays of equal
# length.  Adjust if the dataset schema changes.
_FEATURE_KEYS = {
    "left_wall_dist": "left_wall_dist",
    "track_width":    "track_width",
    "heading_error":  "heading_error",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate σ_proc for left_wall_dist, track_width, heading_error.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/datasets",
        help="Directory containing the .npz dataset files.",
    )
    parser.add_argument(
        "--glob", type=str, default="lidar_tracking_*.npz",
        help=(
            "Glob pattern relative to --data-dir.  Defaults to "
            "'lidar_tracking_*.npz' so that combined/combined_all.npz files "
            "are never accidentally included."
        ),
    )
    parser.add_argument(
        "--exclude-glob", type=str, default=None,
        metavar="PATTERN",
        help=(
            "Additional glob pattern to exclude (matched against the full "
            "path).  Useful for skipping aggregate files that would otherwise "
            "duplicate data, e.g. '*combined*.npz'."
        ),
    )
    return parser.parse_args()


def estimate_sigma_proc(
    data_dir: str,
    glob_pattern: str = "lidar_tracking_*.npz",
    exclude_glob: str | None = None,
) -> dict[str, float]:
    """Load all matching .npz files and compute σ_proc for each feature.

    Each file is treated as one independent trajectory.  First differences are
    computed *within* each file so no artificial discontinuity is introduced at
    trajectory boundaries.  The pooled std across all trajectories gives a
    single global process-noise estimate suitable for use across all maps.

    Parameters
    ----------
    data_dir : str
        Directory to search.
    glob_pattern : str
        Glob pattern matched against file names in ``data_dir``.  Defaults to
        ``lidar_tracking_*.npz`` so combined aggregate files are excluded.
    exclude_glob : str | None
        Optional additional glob pattern; any files whose path matches will be
        removed from the candidate list.  E.g. ``'*combined*.npz'``.

    Returns
    -------
    dict mapping feature name → σ_proc (float).
    """
    pattern = str(Path(data_dir) / glob_pattern)
    files = sorted(glob.glob(pattern))

    # Remove any files that match the exclude pattern.
    if exclude_glob is not None:
        import fnmatch  # stdlib, only imported when needed
        files = [f for f in files if not fnmatch.fnmatch(f, exclude_glob)]

    if not files:
        raise FileNotFoundError(
            f"No files matched '{pattern}' (after exclusions).  "
            "Check --data-dir and --glob arguments."
        )

    # Collect first-differences for each feature across all trajectories.
    diffs: dict[str, list[np.ndarray]] = {k: [] for k in _FEATURE_KEYS}
    missing: set[str] = set()

    for path in files:
        data = np.load(path, allow_pickle=False)
        for feat, key in _FEATURE_KEYS.items():
            if key not in data:
                missing.add(f"{Path(path).name}:{key}")
                continue
            arr = data[key].ravel().astype(np.float64)
            if arr.size < 2:
                continue
            diffs[feat].append(np.diff(arr))

    if missing:
        keys_shown = sorted(missing)[:5]
        suffix = f" (+{len(missing)-5} more)" if len(missing) > 5 else ""
        print(
            f"WARNING: {len(missing)} (file, key) pairs missing: "
            f"{keys_shown}{suffix}",
            file=sys.stderr,
        )

    result: dict[str, float] = {}
    for feat in _FEATURE_KEYS:
        all_diffs = diffs[feat]
        if not all_diffs:
            print(
                f"WARNING: no data found for feature '{feat}'; σ_proc set to 0.0",
                file=sys.stderr,
            )
            result[feat] = 0.0
        else:
            combined = np.concatenate(all_diffs)
            result[feat] = float(np.std(combined, ddof=1))

    return result


def main() -> None:
    args = _parse_args()
    sigmas = estimate_sigma_proc(args.data_dir, args.glob, args.exclude_glob)

    print("Estimated process-noise standard deviations (σ_proc):")
    print(f"  sigma_proc_left    = {sigmas['left_wall_dist']:.6f}")
    print(f"  sigma_proc_track   = {sigmas['track_width']:.6f}")
    print(f"  sigma_proc_heading = {sigmas['heading_error']:.6f}")
    print()
    print("Use these in train_rl.py / reactive_planners.py as:")
    print(f"  --sigma-proc-left    {sigmas['left_wall_dist']:.6g}")
    print(f"  --sigma-proc-track   {sigmas['track_width']:.6g}")
    print(f"  --sigma-proc-heading {sigmas['heading_error']:.6g}")


if __name__ == "__main__":
    main()
