#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_RL="$SCRIPT_DIR/sbatch_rl.sh"

B="data/maps/F1"
MAP_STR="$B/Austin/Austin_map $B/BrandsHatch/BrandsHatch_map $B/Budapest/Budapest_map $B/Catalunya/Catalunya_map $B/Hockenheim/Hockenheim_map $B/IMS/IMS_map $B/Melbourne/Melbourne_map $B/Montreal/Montreal_map $B/MoscowRaceway/MoscowRaceway_map $B/Oschersleben/Oschersleben_map $B/Sakhir/Sakhir_map $B/SaoPaulo/SaoPaulo_map $B/Sepang/Sepang_map $B/Spielberg/Spielberg_map $B/YasMarina/YasMarina_map $B/Zandvoort/Zandvoort_map"
WP_STR="$B/Austin/Austin_centerline.tsv $B/BrandsHatch/BrandsHatch_centerline.tsv $B/Budapest/Budapest_centerline.tsv $B/Catalunya/Catalunya_centerline.tsv $B/Hockenheim/Hockenheim_centerline.tsv $B/IMS/IMS_centerline.tsv $B/Melbourne/Melbourne_centerline.tsv $B/Montreal/Montreal_centerline.tsv $B/MoscowRaceway/MoscowRaceway_centerline.tsv $B/Oschersleben/Oschersleben_centerline.tsv $B/Sakhir/Sakhir_centerline.tsv $B/SaoPaulo/SaoPaulo_centerline.tsv $B/Sepang/Sepang_centerline.tsv $B/Spielberg/Spielberg_centerline.tsv $B/YasMarina/YasMarina_centerline.tsv $B/Zandvoort/Zandvoort_centerline.tsv"
EVAL_MAP_STR="$B/Nuerburgring/Nuerburgring_map $B/Shanghai/Shanghai_map $B/Sochi/Sochi_map"
EVAL_WP_STR="$B/Nuerburgring/Nuerburgring_centerline.tsv $B/Shanghai/Shanghai_centerline.tsv $B/Sochi/Sochi_centerline.tsv"
SENS="0.36876279 0.36876279 0.26247441"

COMMON=(--map-str "$MAP_STR" --waypoints-str "$WP_STR" --eval-map-str "$EVAL_MAP_STR" --eval-waypoints-str "$EVAL_WP_STR")

# ── 6 runs: 3 rewards × 2 latencies ──────────────────────────────────────────
# lat=0  (alpha_steer=0.2, alpha_speed=0.3)
bash "$SBATCH_RL" "${COMMON[@]}" --reward cte                       --cloud-latency 0  --alpha-steer 0.2 --alpha-speed 0.3
bash "$SBATCH_RL" "${COMMON[@]}" --reward cte_sensitivity_annealed  --cloud-latency 0  --alpha-steer 0.2 --alpha-speed 0.3 --call-weights-str "$SENS"
bash "$SBATCH_RL" "${COMMON[@]}" --reward cte_sensitivity_staleness --cloud-latency 0  --alpha-steer 0.2 --alpha-speed 0.3 --call-weights-str "$SENS"

# lat=10  (alpha_steer=0.7, alpha_speed=0.2)
bash "$SBATCH_RL" "${COMMON[@]}" --reward cte                       --cloud-latency 10 --alpha-steer 0.7 --alpha-speed 0.2
bash "$SBATCH_RL" "${COMMON[@]}" --reward cte_sensitivity_annealed  --cloud-latency 10 --alpha-steer 0.7 --alpha-speed 0.2 --call-weights-str "$SENS"
bash "$SBATCH_RL" "${COMMON[@]}" --reward cte_sensitivity_staleness --cloud-latency 10 --alpha-steer 0.7 --alpha-speed 0.2 --call-weights-str "$SENS"
