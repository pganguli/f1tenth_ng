# Sigma-Age Single-Tier Integration Handoff

This document is a copy-paste-ready prompt for another AI to perform an adversarial review of the current uncommitted changes in this repository.

The goal is to review whether the existing sigma-based age-decay mechanism from the selective planner was integrated correctly into the single-tier `EdgeCloudPlanner`, and whether the observed degradation in the 2-map benchmark is due to a bug or a real modeling issue.

## Current repo state

- Repository: `/Users/cembaykal/Desktop/f1tenth_ng`
- Active branch: `feature-alpha-coins`
- The working tree is **not clean**. There are pre-existing uncommitted local changes in addition to the sigma-age work.
- Full test suite currently passes on the working tree:
  - `PYTHONPATH=packages/f110_gym/src:packages/f110_planning/src:packages/f110_scripts/src ./.venv/bin/python -m pytest -q`
  - Result: `203 passed, 1 warning`

## What changed in this task

The user asked to "just use the same one that is existing" in reference to the repository's existing sigma-based age-decay implementation.

The implementation target was the single-tier `EdgeCloudPlanner`, which previously used static feature-level alphas. The intent was to reuse the same age-decay formula already used elsewhere in the repo.

### Existing implementation being reused

The existing age-decay implementation lives in:

- [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/src/f110_planning/reactive/selective_edge_cloud_planner.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/src/f110_planning/reactive/selective_edge_cloud_planner.py)

The key formula is:

`alpha(age) = sigma_e^2 / (sigma_e^2 + sigma_c^2 + sigma_proc^2 * age)`

In the selective planner, this is used together with **edge-delta stale-cloud correction** before blending.

## Files intentionally changed for this task

Please focus first on these files. They are the main scope of the sigma-age integration:

- [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/src/f110_planning/reactive/edge_cloud_planner.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/src/f110_planning/reactive/edge_cloud_planner.py)
- [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/tests/test_reactive_planners.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/tests/test_reactive_planners.py)
- [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py)
- [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_reactive_sim.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_reactive_sim.py)
- [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_single_tier_benchmark.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_single_tier_benchmark.py)
- [`/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_single_tier_paper_strategies.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_single_tier_paper_strategies.py)
- [`/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_single_tier_oracle_baseline.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_single_tier_oracle_baseline.py)

## Important scope note

The working tree also contains **older unrelated local diffs** that were already present before this sigma-age task. They should not be confused with the new work.

These broader local diffs include:

- [`/Users/cembaykal/Desktop/f1tenth_ng/.gitignore`](/Users/cembaykal/Desktop/f1tenth_ng/.gitignore)
- [`/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_paper_results_10maps.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_paper_results_10maps.py)
- [`/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_single_tier_paper_strategies.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_single_tier_paper_strategies.py)
- deleted local file:
  - [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/src/f110_scripts/sim/run_experiments.sh`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/src/f110_scripts/sim/run_experiments.sh)
- untracked local scripts:
  - [`/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_eval_best_configs.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_eval_best_configs.py)
  - [`/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/map_split.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/map_split.py)
  - [`/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/optimize_hyperparameters.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/optimize_hyperparameters.py)
- untracked PDF:
  - [`/Users/cembaykal/Desktop/f1tenth_ng/docs/DAC_2026___Safety_driven_Dynamic_Communication_Scheduling_for_Edge_Cloud_CPS.pdf`](/Users/cembaykal/Desktop/f1tenth_ng/docs/DAC_2026___Safety_driven_Dynamic_Communication_Scheduling_for_Edge_Cloud_CPS.pdf)

Some of the sigma-related files also contain older local benchmark-default changes, so please distinguish the **sigma-age hunks** from any earlier local edits in the same file.

## What the sigma-age integration currently does

### In `EdgeCloudPlanner`

The single-tier planner now:

- accepts optional `sigma_proc_left`, `sigma_proc_track`, and `sigma_proc_heading`
- stores static fallback alphas separately from sigma-age settings
- tracks cloud age via `self._cloud_last_updated` and `self.last_cloud_age`
- computes per-feature alphas with the same formula used in the selective planner
- applies **edge-delta correction** to stale cloud features on arrival, using the edge features recorded at enqueue time and the current edge features at arrival time
- keeps static alpha behavior when `sigma_proc_*` values are `None`

### Sigma constants used

The current implementation uses these per-feature MSE anchors in `EdgeCloudPlanner`:

- Edge:
  - left: `0.028020`
  - track: `0.036518`
  - heading: `0.019371`
- Cloud:
  - left: `0.000518`
  - track: `0.001539`
  - heading: `0.001140`

These correspond to:

- edge: `left arch1`, `track arch2`, `heading arch2`
- cloud: `left arch5`, `track arch7`, `heading arch6`

These values were chosen to align with the later supervisor-provided per-feature table rather than the older uniform `arch2/arch6` pairing.

### Process sigmas used for the rerun

The 2-map sigma-age rerun used:

- `sigma_proc_left = 0.044961`
- `sigma_proc_track = 0.067937`
- `sigma_proc_heading = 0.033182`

At age `0`, the resulting cloud weights are approximately:

- left: `0.9200`
- track: `0.9586`
- heading: `0.9444`

At age `5`, approximately:

- left: `0.5060`
- track: `0.5970`
- heading: `0.7446`

## Tests added or updated

Please inspect whether these tests are sufficient and actually validate the intended semantics:

- [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/tests/test_reactive_planners.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/tests/test_reactive_planners.py)
  - age-decay monotonicity
  - zero-latency same-step cloud feature use
  - edge-delta stale-cloud correction uses current edge features
- [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_reactive_sim.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_reactive_sim.py)
  - CLI support for `sigma_proc_*`
- [`/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_single_tier_benchmark.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_single_tier_benchmark.py)
  - benchmark defaults and sigma-proc defaults

## Experiments rerun

Representative rerun only, to keep it fast:

- maps: `Monza`, `Oschersleben`
- trials: `1`

Static-alpha baseline:

- [`/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/single_tier_paper_strategies_2maps_summary.csv`](/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/single_tier_paper_strategies_2maps_summary.csv)

Sigma-age rerun:

- [`/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/single_tier_paper_strategies_2maps_sigma_age_summary.csv`](/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/single_tier_paper_strategies_2maps_sigma_age_summary.csv)
- [`/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/single_tier_paper_strategies_2maps_sigma_age_best_target_band.csv`](/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/single_tier_paper_strategies_2maps_sigma_age_best_target_band.csv)

Curated figures for the sigma-age run:

- [`/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/paper_figures_2maps_sigma_age_curated/curated_target_band_winners.png`](/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/paper_figures_2maps_sigma_age_curated/curated_target_band_winners.png)
- [`/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/paper_figures_2maps_sigma_age_curated/curated_strategy_wins.png`](/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/paper_figures_2maps_sigma_age_curated/curated_strategy_wins.png)
- [`/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/paper_figures_2maps_sigma_age_curated/curated_target_band_head_to_head.png`](/Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/paper_figures_2maps_sigma_age_curated/curated_target_band_head_to_head.png)

## Observed outcome

The sigma-age rerun is clearly worse than the current static-alpha baseline on both sampled maps.

### Static-alpha best target-band results

- Monza:
  - experiment: `fixed_bernoulli_p65`
  - `CCR = 0.6424`
  - `RMSE = 0.0749`
- Oschersleben:
  - experiment: `bernoulli_max_miss_p55_m3`
  - `CCR = 0.5733`
  - `RMSE = 0.1166`

### Sigma-age best target-band results

- Monza:
  - experiment: `bernoulli_max_miss_p60_m5`
  - `CCR = 0.5955`
  - `RMSE = 0.0912`
- Oschersleben:
  - experiment: `bernoulli_max_miss_p55_m5`
  - `CCR = 0.5520`
  - `RMSE = 0.1445`

### Sigma-age best overall results

- Monza:
  - experiment: `fixed_interval_k5`
  - `CCR = 0.2001`
  - `RMSE = 0.0888`
- Oschersleben:
  - experiment: `fixed_interval_k5`
  - `CCR = 0.2000`
  - `RMSE = 0.1405`

Interpretation so far:

- the sigma-age integration appears to be functionally correct
- the selective-planner age formula does **not** improve the current single-tier controller on these two maps
- the likely reason is that the raw sigma-derived age-0 weights are too cloud-heavy for the current closed-loop behavior

## One independent review finding already surfaced

An independent internal review found one important nuance:

- the first sigma-age port reused the alpha-decay rule but initially did **not** include the selective planner's stale-cloud edge-delta correction
- that has now been corrected in `EdgeCloudPlanner`
- even after correcting that semantic mismatch, the 2-map sigma-age rerun is still worse than the static-alpha baseline

So the remaining question is not "did we forget the stale correction?" That part is now present.

## Main review questions

Please answer these in a skeptical, adversarial way:

1. Was the existing sigma-age implementation transplanted correctly into `EdgeCloudPlanner`?
2. Does the single-tier planner now genuinely match the selective planner's semantics closely enough, or is there still a meaningful mismatch?
3. Are the chosen MSE constants and `sigma_proc_*` parameters being applied correctly?
4. Is the benchmark wiring correct, or could the sigma-age experiment be silently evaluating something different from what is intended?
5. Do the new tests actually protect the semantics we care about, or are there missing cases?
6. Does the poor sigma-age result look like:
   - a real modeling outcome
   - a parameterization issue
   - or a remaining implementation bug?
7. If the implementation is correct, what is the most elegant next step?
   - keep sigma decay but cap fresh-cloud alphas
   - anchor decay to the tuned static alphas instead of raw sigma-optimal age-0 weights
   - keep the paper baseline static and report sigma-age as a negative result

## Adversarial review prompt

Copy-paste the prompt below into another AI:

```text
You are reviewing uncommitted local changes in this repository:

/Users/cembaykal/Desktop/f1tenth_ng

Branch context:
- Active branch is feature-alpha-coins
- The worktree is dirty and includes both new sigma-age changes and older unrelated local diffs
- Focus primarily on the sigma-age integration scope listed below

Your job:
- Perform an adversarial code review
- Prioritize real bugs, semantic mismatches, or invalid experiment conclusions
- Be skeptical
- Do not assume the implementation is correct just because tests pass

Review scope:
- /Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/src/f110_planning/reactive/edge_cloud_planner.py
- /Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/src/f110_planning/reactive/selective_edge_cloud_planner.py
- /Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/tests/test_reactive_planners.py
- /Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py
- /Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_reactive_sim.py
- /Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/tests/test_single_tier_benchmark.py
- /Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_single_tier_paper_strategies.py
- /Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_single_tier_oracle_baseline.py

Supporting experiment outputs:
- /Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/single_tier_paper_strategies_2maps_summary.csv
- /Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/single_tier_paper_strategies_2maps_sigma_age_summary.csv
- /Users/cembaykal/Desktop/f1tenth_ng/data/benchmarks/single_tier_paper_strategies_2maps_sigma_age_best_target_band.csv

Background:
- We reused the existing sigma age-decay formula from SelectiveEdgeCloudPlanner:
  alpha(age) = sigma_e^2 / (sigma_e^2 + sigma_c^2 + sigma_proc^2 * age)
- We added optional sigma_proc_left / sigma_proc_track / sigma_proc_heading support to the single-tier EdgeCloudPlanner path
- We also added stale-cloud edge-delta correction on arrival to mimic the selective planner’s stale-cache behavior
- Full test suite passes, but the sigma-age 2-map benchmark got worse than the current static-alpha baseline

Important details:
- Current per-feature sigma^2 constants used in EdgeCloudPlanner:
  - edge = (0.028020, 0.036518, 0.019371)
  - cloud = (0.000518, 0.001539, 0.001140)
- Current sigma_proc values used in the rerun:
  - left = 0.044961
  - track = 0.067937
  - heading = 0.033182
- That implies very cloud-heavy age-0 weights:
  - left ≈ 0.92
  - track ≈ 0.9586
  - heading ≈ 0.9444

Observed benchmark outcome:
- Static target-band winners:
  - Monza: fixed_bernoulli_p65, CCR 0.6424, RMSE 0.0749
  - Oschersleben: bernoulli_max_miss_p55_m3, CCR 0.5733, RMSE 0.1166
- Sigma-age target-band winners:
  - Monza: bernoulli_max_miss_p60_m5, CCR 0.5955, RMSE 0.0912
  - Oschersleben: bernoulli_max_miss_p55_m5, CCR 0.5520, RMSE 0.1445

Deliverables:
1. Findings first, ordered by severity
2. For each finding, cite exact files and lines if possible
3. Separate “real bug / semantic mismatch” from “parameterization issue” from “expected negative result”
4. State clearly whether the sigma-age integration appears correct
5. State clearly whether the poor performance is more likely due to:
   - implementation bug
   - wrong stale semantics
   - overly aggressive cloud weighting
   - benchmark mismatch
   - or simply an honest negative experimental result
6. Recommend the smallest clean next step

If you find no real implementation bugs, say so explicitly.
```

## Current recommendation before review

My current best read is:

- the sigma-age implementation is probably correct now
- the poor result is likely real
- the main issue is that the raw sigma-optimal age-0 cloud weights are too aggressive for this single-tier controller

But this should be independently stress-tested by a skeptical reviewer.
