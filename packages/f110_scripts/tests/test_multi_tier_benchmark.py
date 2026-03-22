"""Smoke tests for the corrected multi-tier benchmark defaults."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_multi_tier_module():
    root = Path(__file__).resolve().parents[3]
    script_path = root / "scripts/benchmarks/benchmark_multi_tier_cloud.py"
    spec = spec_from_file_location("benchmark_multi_tier_cloud", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_multi_tier_defaults_target_corrected_10_map_workflow(monkeypatch) -> None:
    """The multi-tier benchmark defaults should match the corrected 10-map workflow."""
    module = _load_multi_tier_module()
    monkeypatch.setattr(sys, "argv", ["benchmark_multi_tier_cloud.py"])

    args = module.parse_args()

    assert args.output_stem == "multi_tier_cloud_benchmark_10maps_corrected"
    assert args.maps.split(",") == module.DEFAULT_MAPS
    assert args.delay_pairs == "3:5"
