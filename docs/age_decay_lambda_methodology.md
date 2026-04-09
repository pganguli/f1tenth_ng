# Anchored Age-Decay Methodology

## Objective
Tune one global planner hyperparameter, `age_decay_lambda`, for the anchored single-tier cloud-fusion rule

```text
alpha_i(A) = alpha_static_i * (sigma_e_i^2 + sigma_c_i^2) / (sigma_e_i^2 + sigma_c_i^2 + lambda * A * sigma_proc_i^2)
```

This preserves the tuned static cloud weights at cloud age `A = 0` and only changes how quickly stale cloud features lose trust after arrival.

## Fixed Inputs
- Static feature alphas: `alpha_left = 0.2`, `alpha_track = 0.2`, `alpha_heading = 0.7`
- Edge variances: `sigma_e^2 = (0.028020, 0.036518, 0.019371)`
- Cloud variances: `sigma_c^2 = (0.000518, 0.001539, 0.001140)`
- Process-noise sigmas: `sigma_proc = (0.044961, 0.067937, 0.033182)`
- Cloud latency: `5` simulator steps

## Train / Eval Split
- Training maps (13): Budapest, Catalunya, IMS, Melbourne, MoscowRaceway, Nuerburgring, Sakhir, SaoPaulo, Sepang, Silverstone, Sochi, YasMarina, Zandvoort
- Eval maps (10): Austin, BrandsHatch, Hockenheim, MexicoCity, Montreal, Monza, Oschersleben, Shanghai, Spa, Spielberg

The eval split remains the canonical 10-map paper set to preserve continuity with the earlier figures. All tuning was restricted to the remaining 13 maps.

## Lambda Sweep
- Candidate grid: `0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0`
- Representative strategies:
  - `always_hit`
  - `fixed_interval_k5`
  - `fixed_bernoulli_p60`
  - `bernoulli_max_miss_p60_m5`
- Trials per representative configuration: `3`
- Selection rule:
  1. if any lambda has zero mean collision rate, only compare those lambdas
  2. minimize mean crosstrack RMSE
  3. break ties by minimizing mean absolute distance from target CCR `0.60`

Selected value: `lambda* = 16`

### Lambda Sweep Summary
| λ | Mean Collision | Mean RMSE | Mean CCR | Target-Band Fraction |
|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.0973 | 0.600 | 0.500 |
| 0.25 | 0.000 | 0.0965 | 0.600 | 0.500 |
| 0.5 | 0.000 | 0.0959 | 0.600 | 0.500 |
| 1 | 0.000 | 0.0948 | 0.600 | 0.500 |
| 2 | 0.000 | 0.0933 | 0.600 | 0.500 |
| 4 | 0.000 | 0.0918 | 0.600 | 0.500 |
| 8 | 0.000 | 0.0903 | 0.600 | 0.500 |
| 16 | 0.000 | 0.0902 | 0.600 | 0.500 |

## Strategy Optimization
- Expanded training grid from `scripts/benchmarks/optimize_hyperparameters.py`
- Total configs in current repo: `118`
- Coarse stage: `1` trial per config on all 13 training maps
- Confirmation stage: rerun the winning configuration for each of the 8 strategy families with `3` trials on all 13 training maps

### Best Training Config Per Strategy Family
- `always`: `always_hit` with params `{}` (train RMSE 0.0973, CCR 1.000, collision 0.000)
- `bernoulli_max_miss`: `bernoulli_max_miss_p60_m3` with params `{"max_miss": 3, "p": 0.6, "seed": 7}` (train RMSE 0.0966, CCR 0.612, collision 0.000)
- `deterministic`: `deterministic_t0p07` with params `{"threshold": 0.07}` (train RMSE 0.0972, CCR 0.920, collision 0.000)
- `exponential`: `exponential_c0p05_r10` with params `{"center": 0.05, "rate": 10.0, "seed": 7}` (train RMSE 0.0973, CCR 0.816, collision 0.000)
- `fixed_bernoulli`: `fixed_bernoulli_p65` with params `{"p": 0.65, "seed": 7}` (train RMSE 0.0969, CCR 0.644, collision 0.000)
- `fixed_interval`: `fixed_interval_k2` with params `{"interval": 2}` (train RMSE 0.0970, CCR 0.500, collision 0.000)
- `logistic`: `logistic_c0p05_s20` with params `{"center": 0.05, "seed": 7, "slope": 20.0}` (train RMSE 0.0977, CCR 0.923, collision 0.000)
- `piecewise_ramp`: `piecewise_ramp_0p03_0p08` with params `{"d_high": 0.08, "d_low": 0.03, "seed": 7}` (train RMSE 0.0971, CCR 0.946, collision 0.000)

## Held-Out Evaluation
- Static baseline full canonical suite: `27` configs, `5` trials each on the 10 eval maps
- Anchored-decay full canonical suite: `27` configs, `5` trials each on the 10 eval maps
- Anchored-decay best-config eval: `8` configs, `10` trials each on the 10 eval maps

All paper-facing held-out figures use repeated trials and uncertainty bars derived from the per-config variance across those repeated runs.
