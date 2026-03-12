# Training Pipeline

End-to-end procedure to re-collect data, train neural-network (NN) predictors,
calibrate the process-noise hyperparameters, and train/test the RL cloud
scheduler.  All commands are run from the **repository root** with the virtual
environment active.

```bash
source .venv/bin/activate
```

---

## Step 1 — Re-collect training data

Back up any existing dataset folder, then regenerate per-map NPZ files.

```bash
mv data/datasets data/datasets_old          # keep old data in case you need it
mkdir -p data/datasets
bash packages/f110_scripts/src/f110_scripts/datagen/collect_training_data.sh
```

The script writes one `lidar_tracking_<Map>_<planner>_n<N>.npz` file per map
into `data/datasets/`.

---

## Step 2 — Combine datasets into a single NPZ

```bash
python packages/f110_scripts/src/f110_scripts/datagen/combine_datasets.py \
    data/datasets/lidar_tracking_*.npz \
    --output data/datasets/combined_all.npz
```

> **Note:** The glob intentionally excludes `combined_all.npz` itself so that
> earlier combined files are not double-counted.

---

## Step 3 — Train NN predictors (Slurm)

The Slurm array covers 21 configs (3 features × 7 architectures, indices 0–20):

| Indices  | Architectures |
|----------|---------------|
| 0–2      | arch 1 — heading / left\_wall / track\_width |
| 3–5      | arch 2 |
| 6–8      | arch 3 |
| 9–11     | arch 4 |
| 12–14    | arch 5 |
| 15–17    | arch 6 — heading / left\_wall / track\_width |
| 18–20    | arch 7 |

Submit arch 1 and arch 6 at high priority first, remaining arches after:

```bash
# High-priority: arch 1 (0-2) and arch 6 (15-17)
sbatch --array=0-2,15-17 --qos=hp_volta_gpu \
    packages/f110_scripts/src/f110_scripts/train/train_nn.sl

# Remaining arches (can run concurrently)
sbatch --array=3-14,18-20 \
    packages/f110_scripts/src/f110_scripts/train/train_nn.sl
```

Alternatively, use the interactive launcher which prompts for array selection:

```bash
bash packages/f110_scripts/src/f110_scripts/train/sbatch_nn.sh
```

Wait for all 21 jobs to finish before continuing.

---

## Step 4 — Evaluate NN models and estimate σ_proc

### 4a. Evaluate models (records MSE per feature/arch)

```bash
python packages/f110_scripts/src/f110_scripts/train/eval_nn.py
```

Note the best-arch MSE values for each feature:
- `MSE_left` — σ²_c for left\_wall\_dist
- `MSE_track` — σ²_c for track\_width
- `MSE_heading` — σ²_c for heading\_error

### 4b. Estimate process-noise standard deviations

```bash
python packages/f110_scripts/src/f110_scripts/tune/estimate_sigma_proc.py
```

The script prints three values, e.g.:

```
left_wall_dist  sigma_proc = 0.0312
track_width     sigma_proc = 0.0287
heading_error   sigma_proc = 0.0195
```

### 4c. Compute optimal alpha values

For each feature, the age-0 (single-step) alpha that minimises blending MSE is:

$$\alpha_i = \frac{\sigma_e^2}{\sigma_e^2 + \sigma_c^2}$$

where σ²_e is the edge-model variance (obtained from `eval_nn.py` for the
chosen edge arch) and σ²_c is the cloud-model MSE from Step 4a.

Example Python snippet:

```python
# Replace with your actual MSE values
mse_edge_left, mse_cloud_left     = 0.00120, 0.00082
mse_edge_track, mse_cloud_track   = 0.00095, 0.00061
mse_edge_heading, mse_cloud_head  = 0.00210, 0.00140

alpha_left    = mse_edge_left    / (mse_edge_left    + mse_cloud_left)
alpha_track   = mse_edge_track   / (mse_edge_track   + mse_cloud_track)
alpha_heading = mse_edge_heading / (mse_edge_heading + mse_cloud_head)

print(f"--alpha-left {alpha_left:.4f}  --alpha-track {alpha_track:.4f}  --alpha-heading {alpha_heading:.4f}")
```

---

## Step 5 — Update alpha and σ_proc values in scripts

Replace the placeholder values in the three shell scripts with the numbers
computed in Step 4.  Using `sed` (substitute your numbers):

```bash
NEW_LEFT=0.996
NEW_TRACK=0.988
NEW_HEAD=0.974

for SCRIPT in \
    packages/f110_scripts/src/f110_scripts/train/train_rl.sh \
    packages/f110_scripts/src/f110_scripts/train/test_rl.sh; do
    sed -i \
        -e "s/--alpha-left [0-9.]*/--alpha-left $NEW_LEFT/g" \
        -e "s/--alpha-track [0-9.]*/--alpha-track $NEW_TRACK/g" \
        -e "s/--alpha-heading [0-9.]*/--alpha-heading $NEW_HEAD/g" \
        "$SCRIPT"
done

# Also update the Slurm job defaults:
sed -i \
    -e "s/ALPHA_LEFT:=[0-9.]*/ALPHA_LEFT:=$NEW_LEFT/" \
    -e "s/ALPHA_TRACK:=[0-9.]*/ALPHA_TRACK:=$NEW_TRACK/" \
    -e "s/ALPHA_HEADING:=[0-9.]*/ALPHA_HEADING:=$NEW_HEAD/" \
    packages/f110_scripts/src/f110_scripts/train/train_rl.sl
```

If you also have updated `sigma_proc_*` values, pass them to the RL training
command via `--sigma-proc-left`, `--sigma-proc-track`, `--sigma-proc-heading`
(add these flags in `train_rl.sh`).

---

## Step 6 — Train RL agents

Trains 6 agents (3 reward functions × 2 cloud latencies) by submitting Slurm
jobs via `sbatch_rl.sh`:

```bash
bash packages/f110_scripts/src/f110_scripts/train/train_rl.sh
```

Trained models are saved under `data/models/PPO_4/` with filenames of the form:

```
ppo_<reward>_k1_aL<alpha_left>_aT<alpha_track>_aH<alpha_heading>_lat<latency>.zip
```

---

## Step 7 — Test RL agents

Runs all scheduler variants (round-robin, sensitivity-proportional, and the 6
RL policies) on four held-out maps (MexicoCity, Monza, Silverstone, Spa) at
both latency settings:

```bash
bash packages/f110_scripts/src/f110_scripts/train/test_rl.sh 2>&1 | tee test_results_PPO4.txt
```

> Before running, verify that the model paths inside `test_rl.sh` match the
> filenames produced by Step 6.  If you changed alpha or latency values in
> Step 5, the zip filenames will differ and `test_rl.sh` must be updated to
> match.

---

## Quick reference

| Step | Script / command | Output |
|------|-----------------|--------|
| 1 | `collect_training_data.sh` | `data/datasets/lidar_tracking_*.npz` |
| 2 | `combine_datasets.py` | `data/datasets/combined_all.npz` |
| 3 | `sbatch --array=… train_nn.sl` | `data/models/` NN checkpoints |
| 4a | `eval_nn.py` | MSE table |
| 4b | `estimate_sigma_proc.py` | σ_proc values |
| 5 | `sed` updates | Updated shell scripts |
| 6 | `train_rl.sh` | `data/models/PPO_4/*.zip` |
| 7 | `test_rl.sh` | Console / `test_results_PPO4.txt` |
