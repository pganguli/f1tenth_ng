#!/usr/bin/env bash
# sbatch_rl.sh — interactive partition selector and launcher for train_rl.sl
#
# Usage:
#   ./sbatch_rl.sh                        # prompts for partition only
#   ./sbatch_rl.sh --resume path/to.zip   # resume from checkpoint
#   ./sbatch_rl.sh --extra "--reward cte_plus_call_cost --top-k 2"
#
# Any additional arguments are forwarded to sbatch (e.g. --time=24:00:00).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/train_rl.sl"

# ── Parse script-level options ───────────────────────────────────────────────
RESUME_PATH=""
EXTRA_ARGS=""
SBATCH_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_PATH="$2"; shift 2 ;;
        --extra)
            EXTRA_ARGS="$2"; shift 2 ;;
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
read -rp "Number of parallel envs/CPUs [default: 8, max ~17 for --cpus-per-task=18]: " N_ENVS_INPUT
N_ENVS="${N_ENVS_INPUT:-8}"

# ── Optional: resume path (if not passed as --resume flag) ───────────────────
if [[ -z "$RESUME_PATH" ]]; then
    read -rp "Resume from checkpoint .zip (leave blank to start fresh): " RESUME_INPUT
    RESUME_PATH="${RESUME_INPUT:-}"
fi

# ── Optional: extra train_rl.py flags ────────────────────────────────────────
if [[ -z "$EXTRA_ARGS" ]]; then
    read -rp "Extra train_rl.py flags (leave blank for defaults): " EXTRA_INPUT
    EXTRA_ARGS="${EXTRA_INPUT:-}"
fi

# ── Export vars into the environment (same pattern as sbatch_nn.sh) ──────────
export N_ENVS="$N_ENVS"
[[ -n "$RESUME_PATH" ]] && export RESUME="$RESUME_PATH"
[[ -n "$EXTRA_ARGS"  ]] && export EXTRA_ARGS="$EXTRA_ARGS"

# ── Submit ────────────────────────────────────────────────────────────────────
echo ""
echo "Submitting with:"
echo "  --partition=${PARTITION}"
[[ ${#SBATCH_EXTRA[@]} -gt 0 ]] && echo "  extra sbatch flags: ${SBATCH_EXTRA[*]}"

sbatch --partition="$PARTITION" "${SBATCH_EXTRA[@]}" "$SLURM_SCRIPT"
