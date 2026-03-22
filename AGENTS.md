# `feature-alpha-coins` Working Notes

This repository's canonical research branch is `feature-alpha-coins`.

## Branch intent
- `origin/feature_alpha` is the experimental base:
  per-map `start_pose`, simulator wiring, and the supervisor-aligned model setup.
- `feature-alpha-coins` extends that base with:
  Strategy 1-5 scheduler implementations, the corrected deviation signal,
  canonical benchmark scripts, and curated paper figures.

## Canonical evaluation workflow
Use the workflow documented in [/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/README.md](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/README.md):

1. Run the corrected single-tier benchmark:
   `python scripts/benchmarks/benchmark_single_tier_paper_strategies.py`
2. Optionally run the idealized zero-latency reference:
   `python scripts/benchmarks/benchmark_single_tier_oracle_baseline.py`
3. Optionally run the corrected multi-tier benchmark:
   `python scripts/benchmarks/benchmark_multi_tier_cloud.py`
4. Generate publication-facing figures:
   `python scripts/benchmarks/plot_curated_paper_figures_10maps.py`

## Artifact policy
- Generated benchmark outputs live under `data/benchmarks/`.
- They are local-only artifacts and must remain untracked, except for
  [/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/README.md](/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/README.md).
- Before sharing results, prefer rerunning the canonical scripts instead of
  relying on older local outputs.

## Legacy / compatibility paths
- [/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_paper_results_10maps.py](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_paper_results_10maps.py)
  and [/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_single_tier_paper_strategies.py](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_single_tier_paper_strategies.py)
  are exploratory/internal, not the canonical paper path.
- [/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/src/f110_scripts/sim/run_experiments.sh](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/src/f110_scripts/sim/run_experiments.sh)
  is a legacy helper. Do not treat it as the authoritative benchmark route.
- Older scheduler aliases should remain compatible when feasible, but the
  current paper work should be evaluated through the canonical benchmark scripts.
