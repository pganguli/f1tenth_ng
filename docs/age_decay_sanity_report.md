# Age-Decay Sanity Study Report

## Study Goal
Run a fast sanity-check of anchored age decay using a strict non-train boundary and `max CTE` as the primary selection metric.

## Split
- Train maps: Austin, Budapest, Melbourne, Sakhir, Zandvoort
- Eval maps: Nuerburgring, Sochi, Spa

## Fixed Settings
- Cloud latency: `5`
- Lambda grid: `0, 1, 4, 16, 64`
- Selection metric: `max_cte`
- Boundary hit: `True`

## What Ran
- `lambda_sweep`: `PYTHONPATH=packages/f110_gym/src:packages/f110_planning/src:packages/f110_scripts/src .venv/bin/python scripts/benchmarks/sweep_age_decay_lambda.py --maps Austin,Budapest,Melbourne,Sakhir,Zandvoort --trials 2 --workers 5 --selection-metric max_cte --lambda-values 0,1,4,16,64 --sigma-proc-left 0.044961 --sigma-proc-track 0.067937 --sigma-proc-heading 0.033182 --output-stem lambda_sweep_sanity_5train --selected-lambda-json data/benchmarks/lambda_sweep_sanity_optimal.json`
- `train_family_search`: `PYTHONPATH=packages/f110_gym/src:packages/f110_planning/src:packages/f110_scripts/src .venv/bin/python scripts/benchmarks/optimize_hyperparameters.py --maps Austin,Budapest,Melbourne,Sakhir,Zandvoort --trials 1 --workers 5 --selection-metric max_cte --grid-profile sanity --age-decay-lambda 0 --output-stem strategy_sanity_5train`
- `heldout_static_full_suite`: `PYTHONPATH=packages/f110_gym/src:packages/f110_planning/src:packages/f110_scripts/src .venv/bin/python scripts/benchmarks/benchmark_single_tier_paper_strategies.py --maps Nuerburgring,Sochi,Spa --trials 3 --workers 3 --selection-metric max_cte --output-stem single_tier_sanity_3eval_static`
- `heldout_lambda_full_suite`: `Reused single_tier_sanity_3eval_static outputs under single_tier_sanity_3eval_lambda because selected lambda was 0.0, making anchored decay identical to static fusion in this sanity study.`
- `heldout_best_configs`: `PYTHONPATH=packages/f110_gym/src:packages/f110_planning/src:packages/f110_scripts/src .venv/bin/python scripts/benchmarks/benchmark_eval_best_configs.py --configs-json data/benchmarks/strategy_sanity_5train_best_configs.json --maps Nuerburgring,Sochi,Spa --trials 5 --workers 3 --selection-metric max_cte --age-decay-lambda 0 --output-stem eval_best_configs_sanity_3eval_lambda`
- `figure_generation`: `PYTHONPATH=packages/f110_gym/src:packages/f110_planning/src:packages/f110_scripts/src .venv/bin/python scripts/benchmarks/plot_age_decay_sanity_figures.py --lambda-raw-csv data/benchmarks/lambda_sweep_sanity_5train.csv --selected-lambda-json data/benchmarks/lambda_sweep_sanity_optimal.json --static-raw-csv data/benchmarks/single_tier_sanity_3eval_static.csv --lambda-raw-heldout-csv data/benchmarks/single_tier_sanity_3eval_lambda.csv --best-config-raw-csv data/benchmarks/eval_best_configs_sanity_3eval_lambda.csv --output-dir data/benchmarks/paper_figures_sanity_3eval`

## Selected Lambda
- `lambda* = 0.0`

## Best Train-Time Family Configs
- `always`: `always_hit` with params `{}` (train max CTE 0.3223, CCR 1.000)
- `bernoulli_max_miss`: `bernoulli_max_miss_p65_m6` with params `{"max_miss": 6, "p": 0.65, "seed": 7}` (train max CTE 0.3198, CCR 0.644)
- `deterministic`: `deterministic_t0p11` with params `{"threshold": 0.11}` (train max CTE 0.3201, CCR 0.867)
- `exponential`: `exponential_c0p08_r5` with params `{"center": 0.08, "rate": 5.0, "seed": 7}` (train max CTE 0.3158, CCR 0.655)
- `fixed_bernoulli`: `fixed_bernoulli_p65` with params `{"p": 0.65, "seed": 7}` (train max CTE 0.3186, CCR 0.644)
- `fixed_interval`: `fixed_interval_k2` with params `{"interval": 2}` (train max CTE 0.3237, CCR 0.500)
- `logistic`: `logistic_c0p08_s10` with params `{"center": 0.08, "seed": 7, "slope": 10.0}` (train max CTE 0.3224, CCR 0.851)
- `piecewise_ramp`: `piecewise_ramp_0p05_0p12` with params `{"d_high": 0.12, "d_low": 0.05, "seed": 7}` (train max CTE 0.3185, CCR 0.899)

## Held-Out Best-Family Results
| strategy | experiment | cloud_call_rate_mean | crosstrack_max_m_mean | collision_free_rate |
| --- | --- | --- | --- | --- |
| always | always_hit | 1.0 | 0.3076 | 1.0 |
| bernoulli_max_miss | bernoulli_max_miss_p65_m6 | 0.65062 | 0.36904 | 1.0 |
| deterministic | deterministic_t0p11 | 0.8690999999999999 | 0.309 | 1.0 |
| exponential | exponential_c0p08_r5 | 0.6759000000000001 | 0.32762 | 1.0 |
| fixed_bernoulli | fixed_bernoulli_p65 | 0.64948 | 0.32644 | 1.0 |
| fixed_interval | fixed_interval_k2 | 0.5 | 0.3233 | 1.0 |
| logistic | logistic_c0p08_s10 | 0.8644399999999999 | 0.32682 | 1.0 |
| piecewise_ramp | piecewise_ramp_0p05_0p12 | 0.9037 | 0.30952 | 1.0 |

## Figure Inventory
- `data/benchmarks/paper_figures_sanity_3eval/aggregate_strategy_pareto.pdf`
- `data/benchmarks/paper_figures_sanity_3eval/per_map_strategy_pareto.pdf`
- `data/benchmarks/paper_figures_sanity_3eval/family_comparison_leaderboard.pdf`
- `data/benchmarks/paper_figures_sanity_3eval/appendix_lambda_sweep.pdf`

## Manifest
- `data/benchmarks/age_decay_sanity_manifest.json`
