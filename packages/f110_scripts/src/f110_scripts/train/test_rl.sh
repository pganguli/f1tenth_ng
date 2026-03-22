#!/usr/bin/env bash
set -euo pipefail

#for City in "MexicoCity" "Monza" "Silverstone" "Spa"; do
#for City in "Nuerburgring" "Shanghai" "Sochi"; do
for City in "Spa" "Nuerburgring" "Sochi"; do
  # Fixed interval, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --cloud-strategy interval \
      --cloud-latency 0;
  # Logistic deviation-aware, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --cloud-strategy logistic --logistic-center 0.03 --logistic-slope 40 \
      --cloud-latency 0;
  # cte, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/ppo_cte_k1_aL0.999658_aT0.998227_aH0.996549_lat0.zip \
      --cloud-latency 0;
  ## cte_sensitivity_staleness, lat=0
  #python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
  #    --map "data/maps/F1/${City}/${City}_map" \
  #    --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
  #    --planner edge_cloud --max-laps 2 --render-mode None \
  #    --rl-scheduler data/models/ppo_cte_sensitivity_staleness_k1_aL0.999658_aT0.998227_aH0.996549_lat0.zip \
  #    --cloud-latency 0 --top-k 1;
  # cte_sensitivity_annealed, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/ppo_cte_sensitivity_annealed_k1_aL0.999658_aT0.998227_aH0.996549_lat0.zip \
      --cloud-latency 0;
  # Fixed interval, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --cloud-strategy interval \
      --cloud-latency 10;
  # Logistic deviation-aware, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --cloud-strategy logistic --logistic-center 0.03 --logistic-slope 40 \
      --cloud-latency 10;
  # cte, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/ppo_cte_k1_aL0.999658_aT0.998227_aH0.996549_lat10.zip \
      --cloud-latency 10;
  ## cte_sensitivity_staleness, lat=10
  #python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
  #    --map "data/maps/F1/${City}/${City}_map" \
  #    --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
  #    --planner edge_cloud --max-laps 2 --render-mode None \
  #    --rl-scheduler data/models/ppo_cte_sensitivity_staleness_k1_aL0.999658_aT0.998227_aH0.996549_lat10.zip \
  #    --cloud-latency 10 --top-k 1;
  # cte_sensitivity_annealed, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/ppo_cte_sensitivity_annealed_k1_aL0.999658_aT0.998227_aH0.996549_lat10.zip \
      --cloud-latency 10;
done
