# Benchmark Workflow

This directory contains the corrected spawn-aware evaluation workflow for the
`feature-alpha-coins` branch.

## Canonical Pipeline

Use this path for reproducible paper-facing runs from the repository root:

1. `benchmark_single_tier_paper_strategies.py`
2. `benchmark_single_tier_oracle_baseline.py` (optional idealized reference)
3. `benchmark_multi_tier_cloud.py` (optional supporting/appendix study)
4. `plot_curated_paper_figures_10maps.py`

The canonical single-tier study is the corrected 10-map subset with fixed
latency `5`. The canonical curated output is the three-figure set under
`data/benchmarks/paper_figures_10maps_curated/`.

## Canonical Scripts

| Script | Role | Primary outputs |
|---|---|---|
| `benchmark_single_tier_paper_strategies.py` | Main single-tier Strategy 1-5 benchmark | `single_tier_paper_strategies_10maps*` |
| `benchmark_single_tier_oracle_baseline.py` | Optional idealized always-cloud reference | `single_tier_oracle_baseline_10maps*` |
| `benchmark_multi_tier_cloud.py` | Optional corrected multi-tier comparison | `multi_tier_cloud_benchmark_10maps_corrected*` |
| `plot_curated_paper_figures_10maps.py` | Publication-facing curated figures | `paper_figures_10maps_curated/` |

## Exploratory Scripts

These scripts are kept for analysis, not as the default reproducible path:

- `plot_paper_results_10maps.py`
- `plot_single_tier_paper_strategies.py`

They can be useful for broader diagnosis or internal review, but they are not
the recommended paper-facing output path for this branch.

## Naming Rule

- `benchmark_*`: experiment generation and summary export
- `plot_curated_*`: recommended publication-facing plots
- exploratory plotters: internal analysis only, even if they generate polished
  figures

## Example Commands

```bash
python scripts/benchmarks/benchmark_single_tier_paper_strategies.py
python scripts/benchmarks/benchmark_single_tier_oracle_baseline.py
python scripts/benchmarks/benchmark_multi_tier_cloud.py
python scripts/benchmarks/plot_curated_paper_figures_10maps.py
```
