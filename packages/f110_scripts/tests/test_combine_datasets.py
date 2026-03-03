"""Unit tests for the dataset combination script."""

from pathlib import Path

import numpy as np


from f110_scripts.datagen.combine_datasets import combine_datasets


def test_combine_datasets(tmp_path: Path) -> None:
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

    assert out.exists()
    loaded = np.load(out)
    assert len(loaded["heading_error"]) == 15


def test_combine_datasets_key_mismatch(tmp_path: Path) -> None:
    """A file with different keys should be silently skipped."""
    # First file has heading_error; second has a different key set
    data1 = {"scans": np.random.rand(10, 5), "heading_error": np.random.rand(10)}
    data2 = {"scans": np.random.rand(3, 5), "left_dist": np.random.rand(3)}

    p1 = tmp_path / "good.npz"
    p2 = tmp_path / "bad.npz"
    out = tmp_path / "out.npz"

    np.savez(p1, **data1)
    np.savez(p2, **data2)

    combine_datasets([str(p1), str(p2)], str(out), deduplicate=False)

    # Only the valid file should contribute its rows
    assert out.exists()
    loaded = np.load(out)
    assert loaded["heading_error"].shape[0] == 10


def test_combine_datasets_with_deduplication(tmp_path: Path) -> None:
    """Rows that are identical within epsilon should be collapsed by deduplication."""
    # Create one block of unique rows and a second block that exactly duplicates the first
    unique_rows = np.random.rand(5, 10).astype(np.float32)
    dup_rows = unique_rows.copy()  # exact copies -> distance=0 < epsilon

    p1 = tmp_path / "unique.npz"
    p2 = tmp_path / "dup.npz"
    out = tmp_path / "deduped.npz"

    np.savez(p1, scans=unique_rows, label=np.zeros(5, dtype=np.float32))
    np.savez(p2, scans=dup_rows, label=np.zeros(5, dtype=np.float32))

    combine_datasets([str(p1), str(p2)], str(out), deduplicate=True, epsilon=1e-6)

    loaded = np.load(out)
    # After deduplication we expect only the 5 unique rows to remain
    assert loaded["scans"].shape[0] < 10, (
        f"Expected < 10 rows after deduplication, got {loaded['scans'].shape[0]}"
    )
