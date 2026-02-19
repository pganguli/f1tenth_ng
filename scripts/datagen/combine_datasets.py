#!/usr/bin/env python3
"""
Utility to combine multiple F1TENTH .npz datasets into a single file
with deduplication based on an epsilon threshold.
"""

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


def combine_datasets(
    input_files: List[str],
    output_file: str,
    epsilon: float = 1e-2,
    deduplicate: bool = False,
) -> None:
    """
    Combines multiple .npz datasets and optionally removes redundant rows.

    This function loads multiple NPZ files, checks for key consistency across them,
    concatenates the data, and then optionally performs a deduplication step based on an
    epsilon threshold for all features in a row.

    Args:
        input_files: List of paths to the input .npz files.
        output_file: Path where the combined .npz file will be saved.
        epsilon: Maximum absolute difference for two rows to be considered duplicates.
        deduplicate: Whether to perform deduplication (default: False).
    """
    if not input_files:
        print("No input files provided.")
        return

    data_list: List[Dict[str, np.ndarray]] = []
    keys: Optional[List[str]] = None

    print(f"Loading {len(input_files)} datasets...")
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        try:
            loaded = np.load(file_path)
            # Initialize keys from first file
            if keys is None:
                keys = sorted(list(loaded.keys()))
                print(f"Detected keys: {keys}")
            else:
                # Check consistency
                if sorted(list(loaded.keys())) != keys:
                    print(
                        f"Error: File {file_path} has different keys. Expected {keys}, got {list(loaded.keys())}"
                    )
                    continue

            # Create a dictionary of arrays for this file
            file_data = {k: loaded[k] for k in keys}
            data_list.append(file_data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not data_list or keys is None:
        print("No valid data loaded.")
        return

    # 1. Concatenate all data
    merged_data: Dict[str, np.ndarray] = {}
    for k in keys:
        merged_data[k] = np.concatenate([d[k] for d in data_list], axis=0)

    total_rows = merged_data[keys[0]].shape[0]
    print(f"Total rows after concatenation: {total_rows}")

    # 2. Optional Deduplication logic
    if deduplicate:
        print("Starting deduplication (this may take a while for large datasets)...")
    
        # Combine all columns into a single large matrix for distance calculation
        feature_matrices: List[np.ndarray] = []
        for k in keys:
            # If the key is already 2D (like scans), it's (N, 1080)
            # If it's 1D, we need to make it (N, 1)
            arr = merged_data[k]
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            feature_matrices.append(arr)

        # Big matrix (N, total_features)
        full_matrix = np.concatenate(feature_matrices, axis=1)

        indices_to_keep: List[int] = []

        # Greedy approach: keep first occurrence, skip subsequent almost-duplicates
        seen_mask = np.zeros(total_rows, dtype=bool)

        for i in tqdm(range(total_rows), desc="Deduplicating"):
            if seen_mask[i]:
                continue

            indices_to_keep.append(i)

            # Find all rows that are within epsilon of current row i
            # np.all(abs(matrix - row) < eps, axis=1)
            # We only check rows after i
            diff = np.abs(full_matrix[i + 1 :] - full_matrix[i])
            # A row matches if ALL its columns are < epsilon
            matches = np.all(diff < epsilon, axis=1)

            # Mark matched rows as seen
            if np.any(matches):
                seen_mask[i + 1 :][matches] = True

        print(f"Kept {len(indices_to_keep)} unique rows out of {total_rows}.")

        # 3. Create final filtered dataset
        final_data: Dict[str, np.ndarray] = {}
        for k in keys:
            final_data[k] = merged_data[k][indices_to_keep]
    else:
        print("Deduplication skipped (default behavior).")
        final_data = merged_data

    # Save
    np.savez_compressed(output_file, allow_pickle=False, **final_data)
    print(f"Saved combined dataset to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple .npz datasets with deduplication."
    )
    parser.add_argument("inputs", nargs="+", help="Input .npz files")
    parser.add_argument("--output", "-o", required=True, help="Output .npz file")
    parser.add_argument(
        "--epsilon",
        "-e",
        type=float,
        default=1e-5,
        help="Epsilon threshold for deduplication",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Enable deduplication (default: False)",
    )

    args = parser.parse_args()
    combine_datasets(args.inputs, args.output, args.epsilon, args.dedup)
