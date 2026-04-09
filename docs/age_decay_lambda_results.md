# Anchored Age-Decay Results

## Headline
- Selected global decay parameter: `lambda* = 16`
- Held-out static target-band winners: `10` maps
- Held-out anchored-decay target-band winners: `10` maps
- Best-config strategy families evaluated on held-out maps: `8`

## Held-Out Target-Band Winners

### Static Fusion
| map_name | experiment | strategy | cloud_call_rate_mean | crosstrack_rmse_m_mean |
| --- | --- | --- | --- | --- |
| Austin | bernoulli_max_miss_p55_m3 | bernoulli_max_miss | 0.57412 | 0.09836 |
| BrandsHatch | bernoulli_max_miss_p55_m3 | bernoulli_max_miss | 0.5751999999999999 | 0.07916 |
| Hockenheim | bernoulli_max_miss_p60_m5 | bernoulli_max_miss | 0.60196 | 0.09702 |
| MexicoCity | bernoulli_max_miss_p55_m5 | bernoulli_max_miss | 0.5549799999999999 | 0.10072 |
| Montreal | bernoulli_max_miss_p55_m5 | bernoulli_max_miss | 0.55766 | 0.15928 |
| Monza | bernoulli_max_miss_p60_m5 | bernoulli_max_miss | 0.60248 | 0.08166 |
| Oschersleben | fixed_bernoulli_p55 | fixed_bernoulli | 0.5522 | 0.14164 |
| Shanghai | fixed_bernoulli_p55 | fixed_bernoulli | 0.55138 | 0.1031 |
| Spa | fixed_bernoulli_p55 | fixed_bernoulli | 0.55088 | 0.09296 |
| Spielberg | bernoulli_max_miss_p55_m3 | bernoulli_max_miss | 0.57614 | 0.0877199999999999 |

### Anchored Age Decay
| map_name | experiment | strategy | cloud_call_rate_mean | crosstrack_rmse_m_mean |
| --- | --- | --- | --- | --- |
| Austin | bernoulli_max_miss_p55_m3 | bernoulli_max_miss | 0.57412 | 0.09836 |
| BrandsHatch | bernoulli_max_miss_p55_m3 | bernoulli_max_miss | 0.5751999999999999 | 0.07916 |
| Hockenheim | bernoulli_max_miss_p60_m5 | bernoulli_max_miss | 0.60196 | 0.09702 |
| MexicoCity | bernoulli_max_miss_p55_m5 | bernoulli_max_miss | 0.5549799999999999 | 0.10072 |
| Montreal | bernoulli_max_miss_p55_m5 | bernoulli_max_miss | 0.55766 | 0.15928 |
| Monza | bernoulli_max_miss_p60_m5 | bernoulli_max_miss | 0.60248 | 0.08166 |
| Oschersleben | fixed_bernoulli_p55 | fixed_bernoulli | 0.5522 | 0.14164 |
| Shanghai | fixed_bernoulli_p55 | fixed_bernoulli | 0.55138 | 0.1031 |
| Spa | fixed_bernoulli_p55 | fixed_bernoulli | 0.55088 | 0.09296 |
| Spielberg | bernoulli_max_miss_p55_m3 | bernoulli_max_miss | 0.57614 | 0.0877199999999999 |

## Held-Out Best Configs
| strategy | experiment | cloud_call_rate_mean | crosstrack_rmse_m_mean | collision_free_rate |
| --- | --- | --- | --- | --- |
| always | always_hit | 1.0 | 0.1591 | 1.0 |
| bernoulli_max_miss | bernoulli_max_miss_p60_m3 | 0.61475 | 0.0794 | 1.0 |
| deterministic | deterministic_t0p07 | 0.9365 | 0.1008 | 1.0 |
| exponential | exponential_c0p05_r10 | 0.8852 | 0.15892 | 1.0 |
| fixed_bernoulli | fixed_bernoulli_p65 | 0.64803 | 0.08292 | 1.0 |
| fixed_interval | fixed_interval_k2 | 0.5 | 0.0965 | 1.0 |
| logistic | logistic_c0p05_s20 | 0.93661 | 0.07952 | 1.0 |
| piecewise_ramp | piecewise_ramp_0p03_0p08 | 0.91748 | 0.08352 | 1.0 |

## Figure Inventory
- `data/benchmarks/paper_figures_10maps_age_decay/lambda_train_sweep.*`
- `data/benchmarks/paper_figures_10maps_age_decay/static_vs_lambda_target_band.*`
- `data/benchmarks/paper_figures_10maps_age_decay/strategy_family_tradeoff.*`
- `data/benchmarks/paper_figures_10maps_age_decay/strategy_win_summary.*`
- `data/benchmarks/paper_figures_10maps_age_decay/alpha_decay_curves.*`
- `data/benchmarks/paper_figures_10maps_lambda_curated/*` if the curated plotter is rerun on the anchored-decay held-out summary

## Notes
- `collision_free_rate_ci95`, `crosstrack_rmse_m_ci95`, `cloud_call_rate_ci95`, and `lap_time_s_ci95` are exported in the held-out summary CSVs for uncertainty-aware downstream plotting.
- The canonical exploratory/oracle plotter was not used for these publication-facing figures.
