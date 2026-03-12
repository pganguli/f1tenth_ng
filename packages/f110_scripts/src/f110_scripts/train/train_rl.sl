#!/bin/bash
# train_rl.sl — Slurm batch script for RL cloud-scheduler policy training.
# Override any variable below via `sbatch --export=VAR=val train_rl.sl`
# or export it before calling sbatch (see sbatch_rl.sh).

#SBATCH --job-name=f1tenth_rl
#SBATCH --qos=hp_volta_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=4:00:00
#SBATCH --output=packages/f110_scripts/src/f110_scripts/train/slurm_logs/%x_%j.out
#SBATCH --error=packages/f110_scripts/src/f110_scripts/train/slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ── Defaults ──────────────────────────────────────────────────────────────────
: "${N_ENVS:=6}"
: "${TIMESTEPS:=5000000}"
: "${RESUME:=}"
: "${MAP:=data/maps/F1/Oschersleben/Oschersleben_map}"
: "${WAYPOINTS:=data/maps/F1/Oschersleben/Oschersleben_centerline.tsv}"
: "${EVAL_MAP:=}"
: "${EVAL_WAYPOINTS:=}"
: "${REWARD:=cte}"
: "${CALL_WEIGHTS:=}"
: "${TARGET_CALL_RATES:=}"
: "${CLOUD_LATENCY:=10}"
: "${ALPHA_LEFT:=0.996}"
: "${ALPHA_TRACK:=0.988}"
: "${ALPHA_HEADING:=0.974}"
: "${EXTRA_ARGS:=}"

echo "=== Task $SLURM_JOB_ID: REWARD=$REWARD CLOUD_LATENCY=$CLOUD_LATENCY N_ENVS=$N_ENVS TIMESTEPS=$TIMESTEPS ==="
echo "=== Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-<none>} ==="

# ── Environment setup ─────────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module purge
    module load cuda
fi

cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate

# ── Build command ─────────────────────────────────────────────────────────────
read -ra MAP_ARGS      <<< "$MAP"
read -ra WAYPOINT_ARGS <<< "$WAYPOINTS"

CMD=(
    python packages/f110_scripts/src/f110_scripts/train/train_rl.py
    --map       "${MAP_ARGS[@]}"
    --waypoints "${WAYPOINT_ARGS[@]}"
    --n-envs    "$N_ENVS"
    --device    auto
    --timesteps "$TIMESTEPS"
    --reward    "$REWARD"
    --cloud-latency  "$CLOUD_LATENCY"
    --alpha-left     "$ALPHA_LEFT"
    --alpha-track    "$ALPHA_TRACK"
    --alpha-heading  "$ALPHA_HEADING"
    --checkpoint-freq 500000
    --eval-freq       1000000
    --eval-episodes   5
    --progress-bar
)

[[ -n "$CALL_WEIGHTS"    ]] && { read -ra _A <<< "$CALL_WEIGHTS";    CMD+=(--call-weights    "${_A[@]}"); }
[[ -n "$TARGET_CALL_RATES" ]] && { read -ra _A <<< "$TARGET_CALL_RATES"; CMD+=(--target-call-rates "${_A[@]}"); }
[[ -n "$RESUME" && -f "$RESUME" ]] && CMD+=(--resume "$RESUME")
[[ -n "$EVAL_MAP"        ]] && { read -ra _A <<< "$EVAL_MAP";        CMD+=(--eval-map        "${_A[@]}"); }
[[ -n "$EVAL_WAYPOINTS"  ]] && { read -ra _A <<< "$EVAL_WAYPOINTS";  CMD+=(--eval-waypoints  "${_A[@]}"); }

# Append any caller-supplied extra flags
if [[ -n "$EXTRA_ARGS" ]]; then
    read -ra EXTRA_ARRAY <<< "$EXTRA_ARGS"
    CMD+=("${EXTRA_ARRAY[@]}")
fi

echo "Running: ${CMD[*]}"
srun "${CMD[@]}"
