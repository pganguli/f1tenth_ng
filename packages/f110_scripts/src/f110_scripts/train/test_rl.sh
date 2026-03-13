#!/usr/bin/env bash
set -euo pipefail

for City in "MexicoCity" "Monza" "Silverstone" "Spa"; do
  # Round-robin, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --cloud-strategy round_robin \
      --cloud-latency 0;
  # Sensitivity-proportional, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --cloud-strategy sensitivity --call-weights 0.36876279 0.36876279 0.26247441 \
      --cloud-latency 0;
  # cte, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_4/ppo_cte_k1_aL0.9995553221_aT0.9986653465_aH0.9896705275_lat0.zip \
      --cloud-latency 0;
  # cte_sensitivity_staleness, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_4/ppo_cte_sensitivity_staleness_k1_aL0.9995553221_aT0.9986653465_aH0.9896705275_lat0.zip \
      --cloud-latency 0;
  # cte_sensitivity_annealed, lat=0
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_4/ppo_cte_sensitivity_annealed_k1_aL0.9995553221_aT0.9986653465_aH0.9896705275_lat0.zip \
      --cloud-latency 0 --alpha-left 0.9995553221 --alpha-track 0.9986653465 --alpha-heading 0.9896705275 --top-k 1;
  # Round-robin, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --cloud-strategy round_robin \
      --cloud-latency 10;
  # Sensitivity-proportional, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --cloud-strategy sensitivity --call-weights 0.36876279 0.36876279 0.26247441 \
      --cloud-latency 10;
  # cte, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_4/ppo_cte_k1_aL0.9995553221_aT0.9986653465_aH0.9896705275_lat10.zip \
      --cloud-latency 10 --alpha-left 0.9995553221 --alpha-track 0.9986653465 --alpha-heading 0.9896705275 --top-k 1;
  # cte_sensitivity_staleness, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_4/ppo_cte_sensitivity_staleness_k1_aL0.9995553221_aT0.9986653465_aH0.9896705275_lat10.zip \
      --cloud-latency 10;
  # cte_sensitivity_annealed, lat=10
  python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py \
      --map "data/maps/F1/${City}/${City}_map" \
      --waypoints "data/maps/F1/${City}/${City}_centerline.tsv" \
      --planner edge_cloud --max-laps 2 --render-mode None \
      --rl-scheduler data/models/PPO_4/ppo_cte_sensitivity_annealed_k1_aL0.9995553221_aT0.9986653465_aH0.9896705275_lat10.zip \
      --cloud-latency 10;
done

