#!/bin/bash
#SBATCH --job-name=f1tenth_dnn
#SBATCH --partition=a100-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=scripts/train/slurm_logs/%x_%A_%a.out
#SBATCH --error=scripts/train/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-5

# ── Config array: matches SLURM_ARRAY_TASK_ID 0-5 ──
CONFIGS=(
    "scripts/train/config_heading_small.yaml"
    "scripts/train/config_heading_medium.yaml"
    "scripts/train/config_heading_large.yaml"
    "scripts/train/config_wall_small.yaml"
    "scripts/train/config_wall_medium.yaml"
    "scripts/train/config_wall_large.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Task $SLURM_ARRAY_TASK_ID: $CONFIG ==="
echo "=== Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES ==="

# ── Environment setup ──
module purge
module load cuda  # adjust version to what's available: module avail cuda

# Activate venv (edit path if cloned elsewhere)
cd "$SLURM_SUBMIT_DIR" || exit 1
# shellcheck source=/dev/null
source .venv/bin/activate

# ── Run training ──
srun python scripts/train/train.py --config "$CONFIG"
