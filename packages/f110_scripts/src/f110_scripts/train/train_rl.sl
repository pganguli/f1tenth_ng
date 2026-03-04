#!/bin/bash
# train_rl.sl — Slurm batch script for RL cloud-scheduler policy training.
#
# Key environment variables (override via `sbatch --export=VAR=val,... train_rl.sl`
# or set in sbatch_rl.sh before calling sbatch):
#
#   N_ENVS        Number of parallel SubprocVecEnv workers (default: 8).
#                 Set this to roughly half the allocated CPUs.
#   TIMESTEPS     Total training timesteps (default: 5_000_000).
#   RESUME        Path to a .zip checkpoint to resume from (default: "").
#   MAP           Space-separated map paths (default: Oschersleben).
#   WAYPOINTS     Space-separated waypoint TSV paths (default: Oschersleben).
#   SAVE_PATH     Output .zip path for the final policy (default below).
#   EXTRA_ARGS    Any additional flags forwarded verbatim to train_rl.py.

#SBATCH --job-name=f1tenth_rl
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --output=packages/f110_scripts/src/f110_scripts/train/slurm_logs/%x_%j.out
#SBATCH --error=packages/f110_scripts/src/f110_scripts/train/slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pganguli@unc.edu

# ── Defaults (all overridable via --export) ──────────────────────────────────
: "${N_ENVS:=8}"
: "${TIMESTEPS:=5000000}"
: "${RESUME:=}"
: "${MAP:=data/maps/F1/Oschersleben/Oschersleben_map}"
: "${WAYPOINTS:=data/maps/F1/Oschersleben/Oschersleben_centerline.tsv}"
: "${SAVE_PATH:=data/models/cloud_scheduler.zip}"
: "${EXTRA_ARGS:=}"

echo "=== f1tenth RL training ==="
echo "  Node        : $(hostname)"
echo "  GPU         : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "  CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "  N_ENVS      : ${N_ENVS}"
echo "  TIMESTEPS   : ${TIMESTEPS}"
echo "  RESUME      : ${RESUME:-<none>}"
echo "  MAP(s)      : ${MAP}"
echo "  SAVE_PATH   : ${SAVE_PATH}"
echo "==========================="

# ── Environment setup ────────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module purge
    module load cuda  # adjust version as needed: module avail cuda
fi

cd "$SLURM_SUBMIT_DIR" || exit 1
# shellcheck source=/dev/null
source .venv/bin/activate

# Ensure log and output directories exist
mkdir -p packages/f110_scripts/src/f110_scripts/train/slurm_logs
mkdir -p "$(dirname "$SAVE_PATH")"

# ── Build command ─────────────────────────────────────────────────────────────
# MAP and WAYPOINTS may contain multiple space-separated paths; shell-split them
# into arrays so they become separate --map / --waypoints tokens.
read -ra MAP_ARGS      <<< "$MAP"
read -ra WAYPOINT_ARGS <<< "$WAYPOINTS"

CMD=(
    python packages/f110_scripts/src/f110_scripts/train/train_rl.py
    --map      "${MAP_ARGS[@]}"
    --waypoints "${WAYPOINT_ARGS[@]}"
    --n-envs   "$N_ENVS"
    --device   auto
    --timesteps "$TIMESTEPS"
    --save-path "$SAVE_PATH"
    --checkpoint-freq 200000
    --eval-freq 500000
    --eval-episodes 5
    --progress-bar
)

if [[ -n "$RESUME" && -f "$RESUME" ]]; then
    CMD+=(--resume "$RESUME")
fi

# Append any caller-supplied extra flags
if [[ -n "$EXTRA_ARGS" ]]; then
    read -ra EXTRA_ARRAY <<< "$EXTRA_ARGS"
    CMD+=("${EXTRA_ARRAY[@]}")
fi

echo "Running: ${CMD[*]}"
srun "${CMD[@]}"
