# F1TENTH Next Gen (f1tenth_ng)

This repository is a modernized version of the F1TENTH Gym environment and planning algorithms, updated to support **Gymnasium** and **Pyglet 2.x**.

## Repository Structure

- `packages/f110_gym/`: The core F1TENTH Gymnasium environment.
- `packages/f110_planning/`: A library of planning and tracking algorithms (Pure Pursuit, LQR, etc.).
- `packages/f110_scripts/`: Example scripts and simulation utilities.
- `data/`: Maps and waypoint files.

## Installation

**Important Note for Cloning:** This repository uses Git LFS for large files. You **must** install Git LFS to clone the repository seamlessly. Please see the [official installation instructions](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing) (e.g. `brew install git-lfs` on macOS, or via your package manager such as `apt` on Linux). Once installed, run `git lfs install` to set it up before cloning.

We recommend using a virtual environment and installing both packages in editable mode.

```bash
# Clone the repository
git clone <repo-url>
cd f1tenth_ng

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the gym, planning, and scripts packages
pip install -e packages/f110_gym
pip install -e "packages/f110_planning[test]"
pip install -e "packages/f110_scripts[test]"
```

## Usage Workflow

The repository supports a complete research workflow from data generation to model evaluation:

### 1. Data Generation

Generate datasets of LiDAR scans and ground truth labels (heading error, wall distances) using a waypoint follower with added noise.

```bash
python packages/f110_scripts/src/f110_scripts/datagen/waypoint_datagen.py --map data/maps/F1/Oschersleben/Oschersleben_map --max-steps 10000
```

### 2. Combine Datasets

Merge multiple `.npz` files into a single training dataset with optional deduplication.

```bash
python packages/f110_scripts/src/f110_scripts/datagen/combine_datasets.py data/datasets/file1.npz data/datasets/file2.npz --output data/datasets/combined.npz --dedup
```

### 3. Training

Train LiDAR-based neural networks (e.g., for heading error prediction or wall distance estimation) using PyTorch Lightning.

```bash
python packages/f110_scripts/src/f110_scripts/train/train_nn.py --config packages/f110_scripts/src/f110_scripts/train/config_heading_1.yaml
```

You can monitor the training progress using TensorBoard:

```bash
tensorboard --logdir data/models/lightning_logs
```

### 4. Simulation & Evaluation

Test your planners (classic or DNN-based) in the simulation environment.

**Tracking Planners (Pure Pursuit, LQR, Stanley):**

```bash
python packages/f110_scripts/src/f110_scripts/sim/tracking_planners.py --map data/maps/F1/Oschersleben/Oschersleben_map
```

**Reactive Planners (Gap Follower, Disparity Extender, LiDAR DNN):**

```bash
python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py --planner dnn --map data/maps/F1/Oschersleben/Oschersleben_map
```

## Quickstart: Waypoint Following

This example demonstrates how to use `f110_gym` with a planner from `f110_planning` to follow a pre-defined raceline.

```python
import gymnasium as gym
import numpy as np
import f110_gym
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import load_waypoints

# 1. Create the environment
env = gym.make('f110-v0', 
               map='data/maps/F1/Oschersleben/Oschersleben_map', 
               render_mode='human', 
               num_agents=1)

# 2. Load waypoints using the utility function
waypoints = load_waypoints('data/maps/F1/Oschersleben/Oschersleben_centerline.tsv')

# 3. Initialize the planner
planner = PurePursuitPlanner(waypoints=waypoints)

# 4. Reset and run the simulation loop
obs, info = env.reset(options={'poses': np.array([[0.0, 0.0, 2.85]])})
done = False

while not done:
    # Plan next action
    action = planner.plan(obs)
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(np.array([[action.steer, action.speed]]))
    done = terminated or truncated
    
    env.render()
```

## Reinforcement Learning: Cloud Scheduler

A new Gym environment (`f110-cloud-scheduler-v0`) lets an RL agent learn when to
call a cloud inference in the edge-cloud planner.  The action space is
``Discrete(2)`` (0 = no call, 1 = issue call) and observations include the usual
simulator state plus ``latest_cloud_action`` and
``cloud_request_pending``.  The default reward is the negative squared
cross-track error, but you can pass a custom ``reward_fn`` when creating the
environment.

Example training script using Stable Baselines3::

```python
import gymnasium as gym
from stable_baselines3 import PPO
from f110_planning.utils import load_waypoints

waypoints = load_waypoints("data/maps/F1/Oschersleben/Oschersleben_centerline.tsv")
env = gym.make(
    "f110_gym:f110-cloud-scheduler-v0",
    map="data/maps/F1/Oschersleben/Oschersleben_map",
    waypoints=waypoints,
    cloud_latency=10,
    render_mode=None,
)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
```

You can provide a custom reward function by passing ``reward_fn`` to
``gym.make``:

```python
from typing import Dict, Any

def my_reward(obs: Dict[str, Any], action: int) -> float:
    # negative RMSE + small penalty for taking cloud
    return -obs["crosstrack_rmse_m"] - 0.1 * action

env = gym.make(
    "f110_gym:f110-cloud-scheduler-v0",
    map=..., waypoints=waypoints,
    reward_fn=my_reward,
)
```

The new environment is fully tested and included in the automated test suite.

## Documentation

- [f110_gym README](packages/f110_gym/README.md)
- [f110_planning README](packages/f110_planning/README.md)
