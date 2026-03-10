#!/bin/bash
# train_rl.sl — Slurm batch script for RL cloud-scheduler policy training.
#
# Key environment variables (override via `sbatch --export=VAR=val,... train_rl.sl`
# or set in sbatch_rl.sh before calling sbatch):
#
#   N_ENVS            Number of parallel SubprocVecEnv workers (default: 4).
#                     Set this to roughly half the allocated CPUs.
#   TIMESTEPS         Total training timesteps (default: 5_000_000).
#   RESUME            Path to a .zip checkpoint to resume from (default: "").
#   MAP               Space-separated map paths (default: Oschersleben).
#   WAYPOINTS         Space-separated waypoint TSV paths (default: Oschersleben).
#   EVAL_MAP          Space-separated map path(s) for EvalCallback (held-out). If
#                     unset, falls back to the first MAP entry (train-map eval).
#   EVAL_WAYPOINTS    Space-separated waypoint TSV path(s) matching EVAL_MAP.
#   REWARD            Reward function name: cte | cte_sensitivity_reg |
#                     cte_sensitivity_annealed  (default: cte).
#   CALL_WEIGHTS      Space-separated sensitivity weights for
#                     cte_sensitivity_annealed [left_wall track_width heading].
#                     Example: "0.36876279 0.36876279 0.26247441"
#   TARGET_CALL_RATES Space-separated sensitivity weights for
#                     cte_sensitivity_reg [left_wall track_width heading].
#                     Example: "0.36876279 0.36876279 0.26247441"
#   EXTRA_ARGS        Any additional flags forwarded verbatim to train_rl.py.

#SBATCH --job-name=f1tenth_rl
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=16:00:00
#SBATCH --output=packages/f110_scripts/src/f110_scripts/train/slurm_logs/%x_%j.out
#SBATCH --error=packages/f110_scripts/src/f110_scripts/train/slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pganguli@unc.edu

# ── Defaults (all overridable via --export) ──────────────────────────────────
: "${N_ENVS:=6}"
: "${TIMESTEPS:=20000000}"
: "${RESUME:=}"
: "${MAP:=data/maps/F1/Oschersleben/Oschersleben_map}"
: "${WAYPOINTS:=data/maps/F1/Oschersleben/Oschersleben_centerline.tsv}"
: "${EVAL_MAP:=}"
: "${EVAL_WAYPOINTS:=}"
: "${REWARD:=cte}"
: "${CALL_WEIGHTS:=}"
: "${TARGET_CALL_RATES:=}"
: "${EXTRA_ARGS:=}"

echo "=== f1tenth RL training ==="
echo "  Node        : $(hostname)"
echo "  GPU         : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "  CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "  N_ENVS      : ${N_ENVS}"
echo "  TIMESTEPS   : ${TIMESTEPS}"
echo "  RESUME      : ${RESUME:-<none>}"
echo "  MAP(s)      : ${MAP}"
echo "  EVAL_MAP(s) : ${EVAL_MAP:-<same as MAP[0]>}"
echo "  REWARD      : ${REWARD}"
echo "  CALL_WEIGHTS: ${CALL_WEIGHTS:-<default>}"
echo "  TGT_RATES   : ${TARGET_CALL_RATES:-<default>}"
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
    --reward   "$REWARD"
    --checkpoint-freq 500000
    --eval-freq 1000000
    --eval-episodes 5
    --progress-bar
)

if [[ -n "$CALL_WEIGHTS" ]]; then
    read -ra CW_ARRAY <<< "$CALL_WEIGHTS"
    CMD+=(--call-weights "${CW_ARRAY[@]}")
fi

if [[ -n "$TARGET_CALL_RATES" ]]; then
    read -ra TCR_ARRAY <<< "$TARGET_CALL_RATES"
    CMD+=(--target-call-rates "${TCR_ARRAY[@]}")
fi

if [[ -n "$RESUME" && -f "$RESUME" ]]; then
    CMD+=(--resume "$RESUME")
fi

if [[ -n "$EVAL_MAP" ]]; then
    read -ra EVAL_MAP_ARGS <<< "$EVAL_MAP"
    CMD+=(--eval-map "${EVAL_MAP_ARGS[@]}")
fi

if [[ -n "$EVAL_WAYPOINTS" ]]; then
    read -ra EVAL_WP_ARGS <<< "$EVAL_WAYPOINTS"
    CMD+=(--eval-waypoints "${EVAL_WP_ARGS[@]}")
fi

# Append any caller-supplied extra flags
if [[ -n "$EXTRA_ARGS" ]]; then
    read -ra EXTRA_ARRAY <<< "$EXTRA_ARGS"
    CMD+=("${EXTRA_ARRAY[@]}")
fi

echo "Running: ${CMD[*]}"
srun "${CMD[@]}"
