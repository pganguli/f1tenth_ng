#!/usr/bin/env bash
set -euo pipefail

for City in "MexicoCity" "Monza" "Silverstone" "Spa"; do
  # Run 1: cte, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_2/ppo_cte_k1_as0.7_asp0.2_lat0.zip \
      --cloud-latency 0 --alpha-steer 0.7 --alpha-speed 0.2 --top-k 1;
  # Run 2: cte_sensitivity_staleness, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_1/ppo_cte_sensitivity_staleness_k1_as0.7_asp0.2_lat0.zip \
      --cloud-latency 0 --alpha-steer 0.7 --alpha-speed 0.2 --top-k 1;
  # Run 3: cte_sensitivity_annealed, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_2/ppo_cte_sensitivity_annealed_k1_as0.7_asp0.2_lat0.zip \
      --cloud-latency 0 --alpha-steer 0.7 --alpha-speed 0.2 --top-k 1;
  # Run 4: cte, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_2/ppo_cte_k1_as0.7_asp0.2_lat10.zip \
      --cloud-latency 10 --alpha-steer 0.7 --alpha-speed 0.2 --top-k 1;
  # Run 5: cte_sensitivity_staleness, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
    --map "data/maps/F1/${City}/${City}_map" \
    --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
    --planner edge_cloud --max-laps 2 --render-mode None \
    --rl-scheduler data/models/PPO_1/ppo_cte_sensitivity_staleness_k1_as0.7_asp0.2_lat10.zip \
    --cloud-latency 10 --alpha-steer 0.7 --alpha-speed 0.2 --top-k 1;
  # Run 6: cte_sensitivity_annealed, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_2/ppo_cte_sensitivity_annealed_k1_as0.7_asp0.2_lat10.zip \
      --cloud-latency 10 --alpha-steer 0.7 --alpha-speed 0.2 --top-k 1;
done
