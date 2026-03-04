# f1tenth DNN training — Slurm array job

#SBATCH --job-name=f1tenth_dnn
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=4:00:00
#SBATCH --output=packages/f110_scripts/src/f110_scripts/train/slurm_logs/%x_%A_%a.out
#SBATCH --error=packages/f110_scripts/src/f110_scripts/train/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-20
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pganguli@unc.edu

# ── Config array: matches SLURM_ARRAY_TASK_ID 0-20 ──
# Indices 0-6:  heading (arches 1-7)
# Indices 7-13: left_wall (arches 1-7)
# Indices 14-20: track_width (arches 1-7)
CONFIGS=(
    # arch 1  (indices 0-2)
    "packages/f110_scripts/src/f110_scripts/train/config_heading_1.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_left_wall_1.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_track_width_1.yaml"
    # arch 2  (indices 3-5)
    "packages/f110_scripts/src/f110_scripts/train/config_heading_2.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_left_wall_2.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_track_width_2.yaml"
    # arch 3  (indices 6-8)
    "packages/f110_scripts/src/f110_scripts/train/config_heading_3.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_left_wall_3.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_track_width_3.yaml"
    # arch 4  (indices 9-11)
    "packages/f110_scripts/src/f110_scripts/train/config_heading_4.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_left_wall_4.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_track_width_4.yaml"
    # arch 5  (indices 12-14)
    "packages/f110_scripts/src/f110_scripts/train/config_heading_5.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_left_wall_5.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_track_width_5.yaml"
    # arch 6  (indices 15-17)
    "packages/f110_scripts/src/f110_scripts/train/config_heading_6.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_left_wall_6.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_track_width_6.yaml"
    # arch 7  (indices 18-20)
    "packages/f110_scripts/src/f110_scripts/train/config_heading_7.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_left_wall_7.yaml"
    "packages/f110_scripts/src/f110_scripts/train/config_track_width_7.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Task $SLURM_ARRAY_TASK_ID: $CONFIG ==="
echo "=== Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES ==="

# ── Environment setup ──
if command -v module &>/dev/null; then
    module purge
    module load cuda  # adjust version to what's available: module avail cuda
fi

# Activate venv (edit path if cloned elsewhere)
cd "$SLURM_SUBMIT_DIR" || exit 1
# shellcheck source=/dev/null
source .venv/bin/activate

# ── Run training ──
srun python packages/f110_scripts/src/f110_scripts/train/train_nn.py --config "$CONFIG"
