#!/usr/bin/env bash
# sbatch_rl.sh — interactive partition selector and launcher for train_rl.sl
#
# Usage:
#   ./sbatch_rl.sh                        # guided interactive mode
#   ./sbatch_rl.sh --resume path/to.zip   # resume from checkpoint
#   ./sbatch_rl.sh --reward cte_sensitivity_reg --call-weights-str "0.36876279 0.36876279 0.26247441"
#
# Reward functions:
#   cte                     pure -CTE^2 baseline
#   cte_sensitivity_reg     permanent L2 penalty toward target call rates
#   cte_sensitivity_annealed  annealed bonus toward sensitivity-weighted calls
#
# Sensitivity defaults (offline analysis, slot order: left_wall track_width heading):
#   DEFAULT_SENSITIVITY="0.36876279 0.36876279 0.26247441"
#
# Any remaining arguments are forwarded to sbatch (e.g. --time=24:00:00).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/train_rl.sl"

# ── Parse script-level options ───────────────────────────────────────────────
# Offline sensitivity weights (left_wall track_width heading)
DEFAULT_SENSITIVITY="0.36876279 0.36876279 0.26247441"

RESUME_PATH=""
EXTRA_ARGS=""
REWARD=""
CLI_CALL_WEIGHTS=""      # --call-weights value passed on cli
CLI_TARGET_CALL_RATES="" # --target-call-rates value passed on cli
CLI_MAP_STR=""           # --map-str: space-separated train map YAML paths
CLI_WAYPOINTS_STR=""     # --waypoints-str: space-separated train waypoint TSVs
CLI_EVAL_MAP_STR=""      # --eval-map-str: space-separated eval map YAML paths
CLI_EVAL_WAYPOINTS_STR="" # --eval-waypoints-str: space-separated eval waypoint TSVs
SBATCH_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_PATH="$2"; shift 2 ;;
        --extra)
            EXTRA_ARGS="$2"; shift 2 ;;
        --reward)
            REWARD="$2"; shift 2 ;;
        # convenience: pre-fill sensitivity weights non-interactively
        --call-weights-str)
            CLI_CALL_WEIGHTS="$2"; shift 2 ;;
        --target-call-rates-str)
            CLI_TARGET_CALL_RATES="$2"; shift 2 ;;
        # map / waypoints
        --map-str)
            CLI_MAP_STR="$2"; shift 2 ;;
        --waypoints-str)
            CLI_WAYPOINTS_STR="$2"; shift 2 ;;
        --eval-map-str)
            CLI_EVAL_MAP_STR="$2"; shift 2 ;;
        --eval-waypoints-str)
            CLI_EVAL_WAYPOINTS_STR="$2"; shift 2 ;;
        *)
            SBATCH_EXTRA+=("$1"); shift ;;
    esac
done

# ── Partition selection ───────────────────────────────────────────────────────
mapfile -t PARTITIONS < <(sinfo -h -o "%P %a" | awk '$2=="up"{gsub(/\*$/,"",$1); print $1}' | sort -u)

if [[ ${#PARTITIONS[@]} -eq 0 ]]; then
    echo "No partitions currently available (sinfo returned nothing)." >&2
    exit 1
fi

echo "Available partitions:"
for i in "${!PARTITIONS[@]}"; do
    printf "  [%d] %s\n" "$((i+1))" "${PARTITIONS[$i]}"
done
echo ""
echo "Enter partition numbers separated by spaces to select multiple (e.g. 1 3)."
read -rp "Select partition(s) (1-${#PARTITIONS[@]}): " -a CHOICES

SELECTED=()
for CHOICE in "${CHOICES[@]}"; do
    if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || (( CHOICE < 1 || CHOICE > ${#PARTITIONS[@]} )); then
        echo "Invalid selection: $CHOICE" >&2
        exit 1
    fi
    SELECTED+=("${PARTITIONS[$((CHOICE-1))]}")
done

PARTITION=$(printf '%s\n' "${SELECTED[@]}" | sort -u | paste -sd ',')

# ── Optional: N_ENVS ─────────────────────────────────────────────────────────
read -rp "Number of parallel envs/CPUs [default: 4, max ~7 for --cpus-per-task=8]: " N_ENVS_INPUT
N_ENVS="${N_ENVS_INPUT:-4}"

# ── Optional: resume path (if not passed as --resume flag) ───────────────────
if [[ -z "$RESUME_PATH" ]]; then
    read -rp "Resume from checkpoint .zip (leave blank to start fresh): " RESUME_INPUT
    RESUME_PATH="${RESUME_INPUT:-}"
fi

# ── Reward selection ─────────────────────────────────────────────────────────
if [[ -z "$REWARD" ]]; then
    REWARD_OPTIONS=("cte" "cte_sensitivity_reg" "cte_sensitivity_annealed")
    REWARD_DESCS=(
        "pure -CTE^2 baseline (no call-rate guidance)"
        "permanent L2 penalty toward target call rates (sensitivity prior)"
        "annealed bonus toward sensitivity-weighted calls (headstart -> pure cte)"
    )
    echo ""
    echo "Available reward functions:"
    for i in "${!REWARD_OPTIONS[@]}"; do
        printf "  [%d] %-30s  %s\n" "$((i+1))" "${REWARD_OPTIONS[$i]}" "${REWARD_DESCS[$i]}"
    done
    read -rp "Select reward (1-${#REWARD_OPTIONS[@]}) [default: 1]: " REWARD_CHOICE
    REWARD_CHOICE="${REWARD_CHOICE:-1}"
    if ! [[ "$REWARD_CHOICE" =~ ^[0-9]+$ ]] || (( REWARD_CHOICE < 1 || REWARD_CHOICE > ${#REWARD_OPTIONS[@]} )); then
        echo "Invalid reward selection: $REWARD_CHOICE" >&2
        exit 1
    fi
    REWARD="${REWARD_OPTIONS[$((REWARD_CHOICE-1))]}"
fi
echo "  Reward: $REWARD"

# ── Sensitivity weights (only needed for sensitivity-guided rewards) ──────────
CALL_WEIGHTS_EXPORT=""
TARGET_CALL_RATES_EXPORT=""

if [[ "$REWARD" == "cte_sensitivity_annealed" ]]; then
    if [[ -n "$CLI_CALL_WEIGHTS" ]]; then
        CALL_WEIGHTS_EXPORT="$CLI_CALL_WEIGHTS"
    else
        echo ""
        echo "  Slot order: left_wall_dist  track_width  heading_error"
        read -rp "  CALL_WEIGHTS (3 space-separated floats) [default: $DEFAULT_SENSITIVITY]: " CW_INPUT
        CALL_WEIGHTS_EXPORT="${CW_INPUT:-$DEFAULT_SENSITIVITY}"
    fi
    echo "  CALL_WEIGHTS: $CALL_WEIGHTS_EXPORT"
elif [[ "$REWARD" == "cte_sensitivity_reg" ]]; then
    if [[ -n "$CLI_TARGET_CALL_RATES" ]]; then
        TARGET_CALL_RATES_EXPORT="$CLI_TARGET_CALL_RATES"
    else
        echo ""
        echo "  Slot order: left_wall_dist  track_width  heading_error"
        read -rp "  TARGET_CALL_RATES (3 space-separated floats) [default: $DEFAULT_SENSITIVITY]: " TCR_INPUT
        TARGET_CALL_RATES_EXPORT="${TCR_INPUT:-$DEFAULT_SENSITIVITY}"
    fi
    echo "  TARGET_CALL_RATES: $TARGET_CALL_RATES_EXPORT"
fi

# ── Optional: extra train_rl.py flags ────────────────────────────────────────
if [[ -z "$EXTRA_ARGS" ]]; then
    read -rp "Extra train_rl.py flags (leave blank for defaults): " EXTRA_INPUT
    EXTRA_ARGS="${EXTRA_INPUT:-}"
fi

# ── Export vars into the environment (same pattern as sbatch_nn.sh) ──────────
export N_ENVS="$N_ENVS"
export REWARD="$REWARD"
[[ -n "$RESUME_PATH"            ]] && export RESUME="$RESUME_PATH"
[[ -n "$CALL_WEIGHTS_EXPORT"    ]] && export CALL_WEIGHTS="$CALL_WEIGHTS_EXPORT"
[[ -n "$TARGET_CALL_RATES_EXPORT" ]] && export TARGET_CALL_RATES="$TARGET_CALL_RATES_EXPORT"
[[ -n "$CLI_MAP_STR"            ]] && export MAP="$CLI_MAP_STR"
[[ -n "$CLI_WAYPOINTS_STR"      ]] && export WAYPOINTS="$CLI_WAYPOINTS_STR"
[[ -n "$CLI_EVAL_MAP_STR"       ]] && export EVAL_MAP="$CLI_EVAL_MAP_STR"
[[ -n "$CLI_EVAL_WAYPOINTS_STR" ]] && export EVAL_WAYPOINTS="$CLI_EVAL_WAYPOINTS_STR"
[[ -n "$EXTRA_ARGS"             ]] && export EXTRA_ARGS="$EXTRA_ARGS"

# ── Submit ────────────────────────────────────────────────────────────────────
echo ""
echo "Submitting with:"
echo "  --partition=${PARTITION}"
echo "  REWARD=${REWARD}"
[[ -n "$CALL_WEIGHTS_EXPORT"      ]] && echo "  CALL_WEIGHTS=${CALL_WEIGHTS_EXPORT}"
[[ -n "$TARGET_CALL_RATES_EXPORT" ]] && echo "  TARGET_CALL_RATES=${TARGET_CALL_RATES_EXPORT}"
[[ ${#SBATCH_EXTRA[@]} -gt 0 ]] && echo "  extra sbatch flags: ${SBATCH_EXTRA[*]}"

sbatch --partition="$PARTITION" "${SBATCH_EXTRA[@]}" "$SLURM_SCRIPT"
