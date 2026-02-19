"""
Utilities for loading and processing waypoints.
"""

import numpy as np
import pandas as pd


def load_waypoints(file_path: str, delimiter: str = "\t") -> np.ndarray:
    """
    Loads waypoints from a CSV or TSV file.

    The function attempts to find 'x_m' and 'y_m' columns first, then falls
    back to taking the first two columns if those headers are missing.

    Args:
        file_path: Path to the waypoint file on disk.
        delimiter: The character separating values in the file.

    Returns:
        A numpy array of shape (N, 2) containing [x, y] coordinates.
    """
    try:
        df = pd.read_csv(file_path, sep=delimiter)
        if "x_m" in df.columns and "y_m" in df.columns:
            waypoints = df[["x_m", "y_m"]].to_numpy()
        else:
            waypoints = df.iloc[:, 0:2].to_numpy()
    except (OSError, KeyError, IndexError) as e:
        print(f"Could not load waypoints from {file_path}: {e}")
        waypoints = np.array([]).reshape(0, 2)
    return waypoints
