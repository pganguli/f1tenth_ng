"""Smoke tests for the curated 10-map paper plot script."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_plot_test_data import write_plot_fixture_csvs


def _load_module():
    root = Path(__file__).resolve().parents[3]
    script_dir = root / "scripts/benchmarks"
    sys.path.insert(0, str(script_dir))
    script_path = script_dir / "plot_curated_paper_figures_10maps.py"
    spec = spec_from_file_location("plot_curated_paper_figures_10maps", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_curated_plot_smoke_generates_png_and_pdf(tmp_path: Path) -> None:
    """The curated plot script should render the reduced figure suite."""
    module = _load_module()
    fixture_paths = write_plot_fixture_csvs(tmp_path)
    outputs = module.run(
        single_summary_csv=str(fixture_paths["single"]),
        output_dir=str(tmp_path),
        dpi=72,
        formats=["png", "pdf"],
        target_low=0.55,
        target_high=0.65,
    )

    expected_stems = {
        "curated_target_band_winners",
        "curated_strategy_wins",
        "curated_target_band_head_to_head",
    }
    produced = {(path.stem, path.suffix) for path in outputs}
    for stem in expected_stems:
        assert (stem, ".png") in produced
        assert (stem, ".pdf") in produced
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()
