"""
Unit tests for the dataset combination script.
"""

import os

import numpy as np

from scripts.datagen.combine_datasets import combine_datasets


def test_combine_datasets(tmp_path):
    """Test that multiple datasets can be merged correctly."""
    # Create two dummy datasets
    data1 = {"scans": np.random.rand(10, 1080), "heading_error": np.random.rand(10)}
    data2 = {"scans": np.random.rand(5, 1080), "heading_error": np.random.rand(5)}

    p1 = tmp_path / "data1.npz"
    p2 = tmp_path / "data2.npz"
    out = tmp_path / "combined.npz"

    np.savez(p1, **data1)
    np.savez(p2, **data2)

    combine_datasets([str(p1), str(p2)], str(out), deduplicate=False)

    assert os.path.exists(out)
    loaded = np.load(out)
    assert len(loaded["heading_error"]) == 15
