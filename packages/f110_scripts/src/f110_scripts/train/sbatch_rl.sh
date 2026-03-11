#!/usr/bin/env bash
# sbatch_rl.sh — partition selector and launcher for train_rl.sl
#
# Usage:
#   ./sbatch_rl.sh --reward cte --map-str "..." --waypoints-str "..."
#   ./sbatch_rl.sh --reward cte_sensitivity_annealed --call-weights-str "0.369 0.369 0.262"
#   ./sbatch_rl.sh --reward cte_sensitivity_reg --target-call-rates-str "0.369 0.369 0.262"
#   ./sbatch_rl.sh --resume path/to.zip
#
# Any unrecognised arguments are forwarded to sbatch (e.g. --time=24:00:00).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/train_rl.sl"

# ── Parse flags ───────────────────────────────────────────────────────────────
RESUME_PATH=""
REWARD="cte"
CALL_WEIGHTS=""
TARGET_CALL_RATES=""
MAP_STR=""
WAYPOINTS_STR=""
EVAL_MAP_STR=""
EVAL_WAYPOINTS_STR=""
EXTRA_ARGS=""
SBATCH_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)               RESUME_PATH="$2";       shift 2 ;;
        --reward)               REWARD="$2";             shift 2 ;;
        --call-weights-str)     CALL_WEIGHTS="$2";       shift 2 ;;
        --target-call-rates-str) TARGET_CALL_RATES="$2"; shift 2 ;;
        --map-str)              MAP_STR="$2";            shift 2 ;;
        --waypoints-str)        WAYPOINTS_STR="$2";      shift 2 ;;
        --eval-map-str)         EVAL_MAP_STR="$2";       shift 2 ;;
        --eval-waypoints-str)   EVAL_WAYPOINTS_STR="$2"; shift 2 ;;
        --cloud-latency)        CLOUD_LATENCY="$2";      shift 2 ;;
        --extra)                EXTRA_ARGS="$2";         shift 2 ;;
        *)                      SBATCH_EXTRA+=("$1");    shift ;;
    esac
done

# ── Partition selection ───────────────────────────────────────────────────────
mapfile -t PARTITIONS < <(sinfo -h -o "%P %a" | awk '$2=="up"{gsub(/\*$/,"",$1); print $1}' | sort -u)

if [[ ${#PARTITIONS[@]} -eq 0 ]]; then
    echo "No partitions available." >&2
    exit 1
fi

echo "Available partitions:"
for i in "${!PARTITIONS[@]}"; do
    printf "  [%d] %s\n" "$((i+1))" "${PARTITIONS[$i]}"
done

read -rp "Select partition (1-${#PARTITIONS[@]}): " CHOICE
if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || (( CHOICE < 1 || CHOICE > ${#PARTITIONS[@]} )); then
    echo "Invalid selection: $CHOICE" >&2
    exit 1
fi
PARTITION="${PARTITIONS[$((CHOICE-1))]}"

# ── Export env vars for train_rl.sl ──────────────────────────────────────────
export REWARD="$REWARD"
[[ -n "$RESUME_PATH"      ]] && export RESUME="$RESUME_PATH"
[[ -n "$CALL_WEIGHTS"     ]] && export CALL_WEIGHTS="$CALL_WEIGHTS"
[[ -n "$TARGET_CALL_RATES" ]] && export TARGET_CALL_RATES="$TARGET_CALL_RATES"
[[ -n "$MAP_STR"          ]] && export MAP="$MAP_STR"
[[ -n "$WAYPOINTS_STR"    ]] && export WAYPOINTS="$WAYPOINTS_STR"
[[ -n "$EVAL_MAP_STR"     ]] && export EVAL_MAP="$EVAL_MAP_STR"
[[ -n "$EVAL_WAYPOINTS_STR" ]] && export EVAL_WAYPOINTS="$EVAL_WAYPOINTS_STR"
[[ -n "$CLOUD_LATENCY"    ]] && export CLOUD_LATENCY="$CLOUD_LATENCY"
[[ -n "$EXTRA_ARGS"       ]] && export EXTRA_ARGS="$EXTRA_ARGS"

# ── Submit ────────────────────────────────────────────────────────────────────
DOMAIN=$(hostname -d)
[[ "$DOMAIN" == *.*.* ]] && DOMAIN="${DOMAIN#*.}"
EMAIL="${USER}@${DOMAIN}"

echo "Submitting with --partition=$PARTITION REWARD=$REWARD --mail-user=$EMAIL ..."
sbatch --partition="$PARTITION" --mail-user="$EMAIL" "${SBATCH_EXTRA[@]}" "$SLURM_SCRIPT"
