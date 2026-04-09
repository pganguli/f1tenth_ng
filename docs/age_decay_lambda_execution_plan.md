# Age-Decay Lambda Execution Plan

This document captures the current agreed plan and the relevant technical context for adding anchored age-dependent fusion decay, tuning it properly on training maps, evaluating all strategies on held-out maps, and producing paper-ready figures and reports.

## Goal

Add one global planner hyperparameter, `age_decay_lambda`, to the single-tier edge-cloud controller so that:

- the current tuned static feature alphas are preserved at cloud age `0`
- trust in held cloud predictions decays as they become stale
- the new hyperparameter is tuned only on training maps
- all strategy families are then evaluated on held-out evaluation maps
- all reported results include uncertainty/error bars from repeated trials
- the final outputs include publication-facing figures and methodology/results reports

## Why this formulation

The direct sigma-age rule currently implemented in the selective planner is:

\[
\alpha_i(A) = \frac{\sigma^2_{e,i}}{\sigma^2_{e,i} + \sigma^2_{c,i} + A \sigma^2_{proc,i}}
\]

In the single-tier planner, that rule made age-0 cloud weights extremely large and empirically worsened Monza and Oschersleben. The core issue is that the raw MSE-based age-0 anchor is too cloud-heavy for the current closed-loop controller.

The chosen fix is to keep the sigma/process-noise structure only for the **decay shape**, while preserving the already tuned static age-0 alphas.

## Chosen anchored decay rule

For each feature `i ∈ {left, track, heading}` and cloud age `A`:

\[
\alpha_i(A)=\alpha^{static}_i \cdot
\frac{\sigma^2_{e,i}+\sigma^2_{c,i}}
{\sigma^2_{e,i}+\sigma^2_{c,i}+ \lambda \, A \, \sigma^2_{proc,i}}
\]

Where:

- `alpha_static_i` is the existing tuned cloud weight for feature `i`
- `sigma_e^2` is the measured edge-model feature error proxy
- `sigma_c^2` is the measured cloud-model feature error proxy
- `sigma_proc` is the measured process-noise standard deviation
- `A` is cloud age in steps since arrival
- `lambda >= 0` is one new global hyperparameter shared across all strategies

This guarantees:

- `A = 0` gives exactly the current static alpha
- `lambda = 0` gives exactly the current static controller behavior
- larger `lambda` means faster decay of cloud trust as age grows

## Fixed inputs and defaults

### Static feature alphas

These remain fixed during this study:

- `alpha_left = 0.2`
- `alpha_track = 0.2`
- `alpha_heading = 0.7`

### Sigma inputs

Keep the existing sigma values fixed during this phase:

- `sigma_e^2` and `sigma_c^2` come from the current per-feature edge/cloud model choices
- `sigma_proc_*` come from the existing process-noise estimation workflow

No re-estimation of model variances or static alphas is included in this phase.

### Train/eval split

Use the existing canonical split in [`scripts/benchmarks/map_split.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/map_split.py).

#### Eval maps

Keep the current canonical 10-map paper set for continuity:

- `Austin`
- `BrandsHatch`
- `Hockenheim`
- `MexicoCity`
- `Montreal`
- `Monza`
- `Oschersleben`
- `Shanghai`
- `Spa`
- `Spielberg`

#### Train maps

Use the remaining 13 F1 maps for tuning:

- `Budapest`
- `Catalunya`
- `IMS`
- `Melbourne`
- `MoscowRaceway`
- `Nuerburgring`
- `Sakhir`
- `SaoPaulo`
- `Sepang`
- `Silverstone`
- `Sochi`
- `YasMarina`
- `Zandvoort`

## Workload and parallel execution

### Current config counts from the repo

The current benchmark code implies:

- default canonical strategy suite: `27` configurations
- strategy families: `8`
- expanded train-time optimization grid: `118` configurations

#### Default 27-config suite

- `always`: `1`
- `fixed_interval`: `3`
- `fixed_bernoulli`: `3`
- `bernoulli_max_miss`: `6`
- `logistic`: `4`
- `exponential`: `4`
- `piecewise_ramp`: `3`
- `deterministic`: `3`

#### Expanded 118-config optimization grid

- `always`: `1`
- `fixed_interval`: `6`
- `fixed_bernoulli`: `6`
- `bernoulli_max_miss`: `30`
- `logistic`: `25`
- `exponential`: `25`
- `piecewise_ramp`: `19`
- `deterministic`: `6`

### Planned sweep totals

With the current plan, the total workload is:

1. lambda coarse tuning:
   - `8 lambda values × 27 configs × 13 maps × 1 trial = 2,808 episodes`
2. lambda refined tuning:
   - `3 lambda values × 27 configs × 13 maps × 3 trials = 3,159 episodes`
3. strategy optimization coarse pass:
   - `118 configs × 13 maps × 1 trial = 1,534 episodes`
4. strategy optimization confirmation pass:
   - `8 winning configs × 13 maps × 3 trials = 312 episodes`
5. held-out static baseline eval:
   - `27 configs × 10 maps × 5 trials = 1,350 episodes`
6. held-out tuned-lambda full-suite eval:
   - `27 configs × 10 maps × 5 trials = 1,350 episodes`
7. held-out tuned best-config eval:
   - `8 configs × 10 maps × 10 trials = 800 episodes`

Grand total:

- `11,313` simulator episodes

### Canonical parallelization model

This workload must be run in parallel.

Use **map-level process parallelism** as the canonical execution strategy:

- parallelize across maps, not across trials within a map
- one worker handles one map batch at a time
- each worker runs all requested configs/trials for that map sequentially

Why:

- map episodes are independent
- model and environment setup can be reused within each subprocess
- aggregation stays simple
- this matches the current pattern already used in `optimize_hyperparameters.py`

### Required worker support

The following scripts must support `--workers` and parallelize across maps:

- `scripts/benchmarks/optimize_age_decay_lambda.py`
- `scripts/benchmarks/benchmark_single_tier_paper_strategies.py`
- `scripts/benchmarks/benchmark_eval_best_configs.py`

`scripts/benchmarks/optimize_hyperparameters.py` already supports map-level process parallelism and should remain the reference implementation pattern.

### Parallel execution rules

- default `--workers=1` for reproducibility
- recommended runtime value:
  - `workers = min(number_of_maps_in_stage, practical_cpu_budget)`
- do not nest parallel pools
- do not parallelize both across maps and across trials simultaneously
- if memory pressure appears due to model loading, reduce workers rather than changing the parallelism pattern

### Practical runtime stages

The heaviest stages are:

- lambda tuning: `5,967` episodes total
- held-out eval: `3,500` episodes total

So the implementation should prioritize robust parallelism in:

1. lambda tuning
2. held-out full-suite eval
3. best-config held-out eval

## Implementation plan

### 1. Planner and CLI

Add `age_decay_lambda: float = 0.0` to the single-tier [`EdgeCloudPlanner`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_planning/src/f110_planning/reactive/edge_cloud_planner.py).

Preserve current behavior exactly when:

- `age_decay_lambda = 0.0`, or
- `sigma_proc_*` is unset

Keep the existing stale-cloud edge-delta correction unchanged.

Add `--age-decay-lambda` to:

- [`packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py`](/Users/cembaykal/Desktop/f1tenth_ng/packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py)
- [`scripts/benchmarks/benchmark_single_tier_paper_strategies.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_single_tier_paper_strategies.py)
- [`scripts/benchmarks/benchmark_single_tier_oracle_baseline.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_single_tier_oracle_baseline.py)
- [`scripts/benchmarks/optimize_hyperparameters.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/optimize_hyperparameters.py)
- [`scripts/benchmarks/benchmark_eval_best_configs.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_eval_best_configs.py)
- add `--workers` to any benchmark script that does not already expose it

### 2. Lambda tuning on training maps

Add a new canonical tuning script:

- [`scripts/benchmarks/optimize_age_decay_lambda.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/optimize_age_decay_lambda.py)

It must mirror the current map-level `ProcessPoolExecutor` pattern already used in `optimize_hyperparameters.py`.

Tune one global `lambda` over:

- `{0, 0.25, 0.5, 1, 2, 4, 8, 16}`

#### Stage 1

- run the canonical default single-tier strategy suite on all 13 training maps
- `1` trial per configuration

#### Stage 2

- rerun the best `3` lambda candidates from Stage 1
- `3` trials per configuration on all 13 training maps

#### Lambda selection rule

Rank lambda values by:

1. highest mean `collision_free_rate`
2. highest fraction of rows in the `0.55-0.65` cloud-call-rate band
3. lowest mean `crosstrack_rmse_m_mean`
4. lowest mean absolute distance from target `CCR = 0.60`

#### Tuning outputs

Write:

- `data/benchmarks/age_decay_lambda_train_13maps*`
- `data/benchmarks/age_decay_lambda_train_13maps_refined*`
- `data/benchmarks/best_age_decay_lambda.json`

### 3. Strategy hyperparameter optimization on training maps

After selecting `lambda`, freeze it and run the existing expanded strategy-family optimization on all 13 training maps.

Use [`scripts/benchmarks/optimize_hyperparameters.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/optimize_hyperparameters.py) with the frozen `--age-decay-lambda`.

Keep the current expanded search spaces for:

- `always`
- `fixed_interval`
- `fixed_bernoulli`
- `bernoulli_max_miss`
- `logistic`
- `exponential`
- `piecewise_ramp`
- `deterministic`

#### Optimization passes

- coarse pass: `1` trial per config
- confirmation pass: rerun the winning config for each strategy family with `3` trials

#### Optimization outputs

Write:

- `data/benchmarks/hyperparam_optimization_train_13maps_lambda*`
- `data/benchmarks/best_configs.json`

### 4. Held-out eval on all strategies

Run three held-out eval studies on the 10 eval maps:

1. static baseline canonical sweep with `lambda = 0`, `5` trials per config
2. tuned-lambda canonical sweep of **all strategies**, `5` trials per config
3. tuned-lambda eval of frozen best per-strategy configs, `10` trials per config

All three eval studies must run with map-level parallel execution support.

#### Eval output stems

- `single_tier_paper_strategies_10maps_static*`
- `single_tier_paper_strategies_10maps_lambda*`
- `eval_best_configs_10maps_lambda*`

## Error bars and uncertainty

Repeated trials are required for every paper-facing result.

For every held-out eval summary row, compute and retain:

- mean
- standard deviation
- standard error
- 95% confidence interval

Primary metrics with uncertainty:

- `crosstrack_rmse_m`
- `cloud_call_rate`
- `collision_free_rate`
- `lap_time_s`

Use binomial confidence intervals for collision-free rate.

### Plotting rules

All paper-facing comparison figures must include visible uncertainty:

- vertical error bars for RMSE
- horizontal error bars for cloud-call rate where applicable
- confidence intervals in captions or summary annotations

Do not use single-trial results in any paper-facing figure.

## Paper-ready figures

Keep [`scripts/benchmarks/plot_curated_paper_figures_10maps.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_curated_paper_figures_10maps.py) as the canonical plot path and extend it, or add one small canonical companion plotter, for:

- `lambda_train_sweep`
- `static_vs_lambda_target_band`
- `strategy_family_tradeoff`
- refreshed curated 10-map figures from the tuned-lambda full eval sweep
- `strategy_win_summary`

### Figure requirements

- evaluate **all strategies**
- show uncertainty/error bars
- export both `png` and `pdf`
- `600 dpi` PNG
- colorblind-safe palette
- direct labels where possible
- uncluttered layout
- publication-oriented sizing and typography

The final paper-facing figure set should clearly show:

- how `lambda` was chosen
- how tuned lambda compares against static fusion
- how all strategy families trade off communication vs tracking error
- uncertainty on all reported held-out comparisons

## Reports and deliverables

Generate both Markdown and LaTeX source:

- `docs/age_decay_lambda_methodology.md`
- `docs/age_decay_lambda_results.md`
- `docs/age_decay_lambda_methodology.tex`
- `docs/age_decay_lambda_results.tex`

Required report contents:

- train/eval split and rationale
- exact config counts per sweep stage
- total episode count and parallel execution model
- exact anchored formula
- fixed static alphas and sigma inputs
- lambda sweep grid and selection rule
- strategy-family optimization grids and winner-selection rule
- exact trial counts for every stage
- exact benchmark commands or output stems
- held-out eval results for all strategies
- uncertainty methodology and how error bars were computed
- figure captions and interpretation

### PDF note

No local TeX engine is currently installed. Therefore:

- Markdown reports are required
- matching `.tex` sources are required
- compiled PDF is optional and can be produced later when a TeX engine is available

## Tests and validation

### Planner tests

- `lambda=0` reproduces current static alpha behavior exactly
- `age=0` reproduces current static alphas exactly for any `lambda`
- alpha decreases monotonically with age when `lambda>0`
- alpha decreases as `lambda` increases for fixed age
- `sigma_proc_* is None` ignores `lambda`

### Benchmark tests

- parse and thread `--age-decay-lambda`
- lambda tuner writes the expected JSON/CSV schema and selected-lambda record
- eval summaries include std, stderr, and confidence intervals

### Plot tests

- uncertainty-aware plots render from synthetic fixture data
- all-strategy tradeoff figure renders with error bars
- curated plotter still works with tuned-lambda outputs

### Final validation

- full `pytest -q`
- train tuning artifacts exist
- best-config optimization artifacts exist
- held-out eval artifacts exist for static, tuned full sweep, and tuned best-config runs
- paper-facing figures exist in both `png` and `pdf`
- Markdown and LaTeX reports exist

## Current repository context relevant to this plan

- Branch: `feature-alpha-coins`
- Existing canonical single-tier script:
  - [`scripts/benchmarks/benchmark_single_tier_paper_strategies.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_single_tier_paper_strategies.py)
- Existing local optimizer helper:
  - [`scripts/benchmarks/optimize_hyperparameters.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/optimize_hyperparameters.py)
- Existing local held-out eval helper:
  - [`scripts/benchmarks/benchmark_eval_best_configs.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/benchmark_eval_best_configs.py)
- Existing canonical plotter:
  - [`scripts/benchmarks/plot_curated_paper_figures_10maps.py`](/Users/cembaykal/Desktop/f1tenth_ng/scripts/benchmarks/plot_curated_paper_figures_10maps.py)
- Existing adversarial-review handoff for the earlier sigma-age experiment:
  - [`docs/adversarial_review_sigma_age_handoff.md`](/Users/cembaykal/Desktop/f1tenth_ng/docs/adversarial_review_sigma_age_handoff.md)

## Assumptions

- `age_decay_lambda` is one global planner-level hyperparameter shared across all strategies.
- Static feature alphas remain fixed throughout this study.
- Existing `sigma_e^2`, `sigma_c^2`, and `sigma_proc` values remain fixed.
- The current 10-map paper set remains the held-out eval set for continuity.
- Generated benchmark outputs remain untracked by git.
- Code, tests, plots, Markdown docs, and LaTeX sources are tracked.
