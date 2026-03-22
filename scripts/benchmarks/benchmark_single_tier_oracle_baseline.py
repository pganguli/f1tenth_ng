#!/usr/bin/env python3
"""Canonical reference benchmark: idealized always-hit latency-0 baseline.

Reads: the same corrected 10-map setup as the single-tier benchmark.
Writes: optional idealized-reference artifacts under ``data/benchmarks``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from benchmark_single_tier_paper_strategies import (  # pylint: disable=import-error
    DEFAULT_TRIALS,
    Experiment,
    PlannerSettings,
    _json_default,
    available_map_names,
    maybe_write_xlsx,
    run_episode,
    summarize,
    track_config,
)


DEFAULT_MAPS = [
    "Austin",
    "BrandsHatch",
    "Hockenheim",
    "MexicoCity",
    "Montreal",
    "Monza",
    "Oschersleben",
    "Shanghai",
    "Spa",
    "Spielberg",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Optional idealized reference benchmark for the corrected 10-map workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Status: canonical optional reference. Example: "
            "python scripts/benchmarks/benchmark_single_tier_oracle_baseline.py"
        ),
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=",".join(DEFAULT_MAPS),
        help="Comma-separated map names under data/maps/F1.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of repeated trials per map.",
    )
    parser.add_argument(
        "--max-laps",
        type=int,
        default=1,
        help="Maximum laps per episode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15000,
        help="Safety cap on simulator steps per episode.",
    )
    parser.add_argument(
        "--cloud-latency",
        type=int,
        default=0,
        help="Latency for the oracle baseline, defaulting to 0 steps.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="single_tier_oracle_baseline_10maps",
        help="Base filename stem for outputs under data/benchmarks.",
    )
    parser.add_argument(
        "--alpha-left",
        type=float,
        default=0.2,
        help="Cloud blend weight for the left-distance feature.",
    )
    parser.add_argument(
        "--alpha-track",
        type=float,
        default=0.2,
        help="Cloud blend weight for the track-width feature.",
    )
    parser.add_argument(
        "--alpha-heading",
        type=float,
        default=0.7,
        help="Cloud blend weight for the heading-error feature.",
    )
    parser.add_argument(
        "--deviation-steer-weight",
        type=float,
        default=1.0,
        help="Weight on steering disagreement when forming scheduler deviation.",
    )
    parser.add_argument(
        "--deviation-speed-weight",
        type=float,
        default=1.0,
        help="Weight on speed disagreement when forming scheduler deviation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the oracle baseline sweep and write benchmark artifacts."""
    args = parse_args()
    requested_maps = [name.strip() for name in args.maps.split(",") if name.strip()]
    available = set(available_map_names())
    missing = sorted(set(requested_maps).difference(available))
    if missing:
        raise ValueError(f"Unknown map names: {', '.join(missing)}")

    tracks = [track_config(name) for name in requested_maps]
    settings = PlannerSettings(
        alpha_left=args.alpha_left,
        alpha_track=args.alpha_track,
        alpha_heading=args.alpha_heading,
        deviation_steer_weight=args.deviation_steer_weight,
        deviation_speed_weight=args.deviation_speed_weight,
    )
    experiment = Experiment("always_hit", "always", {})

    results = [
        run_episode(
            experiment,
            track,
            cloud_latency=args.cloud_latency,
            max_laps=args.max_laps,
            settings=settings,
            run_idx=run_idx,
            max_steps=args.max_steps,
        )
        for track in tracks
        for run_idx in range(args.trials)
    ]

    experiments_df, summary_df, near_target_df = summarize(results)
    del near_target_df

    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tracks": [track.__dict__ for track in tracks],
        "cloud_latency": float(args.cloud_latency),
        "cloud_latencies": [float(args.cloud_latency)],
        "settings": settings.__dict__,
        "experiments": results,
        "summary": summary_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    experiments_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    engine = maybe_write_xlsx(
        {
            "experiments": experiments_df,
            "summary": summary_df,
        },
        xlsx_path,
    )

    print(summary_df.to_string(index=False))
    print(f"\nWrote JSON report to {json_path}")
    print(f"Wrote experiment CSV to {csv_path}")
    print(f"Wrote summary CSV to {summary_csv_path}")
    if engine is None:
        print("Skipped XLSX export because neither openpyxl nor xlsxwriter is installed.")
    else:
        print(f"Wrote XLSX workbook to {xlsx_path} using {engine}.")


if __name__ == "__main__":
    main()
