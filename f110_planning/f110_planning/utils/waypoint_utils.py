import numpy as np


def load_waypoints(file_path, delimiter=";", skiprows=3):
    """
    Loads waypoints from a CSV file and reorders columns to [x, y, v, th].

    Args:
        file_path (str): Path to the waypoint CSV file.
        delimiter (str): Delimiter used in the CSV file.
        skiprows (int): Number of header rows to skip.

    Returns:
        np.ndarray: Waypoints as a numpy array with columns [x, y, v, th].
    """
    try:
        waypoints = np.loadtxt(file_path, delimiter=delimiter, skiprows=skiprows)
        # Reorder columns:
        #   CSV is [s, x, y, th, kappa, v, a]
        #   Planner/Renderer expect [x, y, v, th]
        #   User indices: x=1, y=2, th=3, v=5
        waypoints = waypoints[:, [1, 2, 5, 3]]
    except Exception as e:
        print(f"Could not load waypoints from {file_path}: {e}")
        waypoints = np.array([])
    return waypoints
