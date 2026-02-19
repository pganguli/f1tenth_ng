# F1TENTH Next Gen (f1tenth_ng)

This repository is a modernized version of the F1TENTH Gym environment and planning algorithms, updated to support **Gymnasium** and **Pyglet 2.x**.

## Repository Structure

- `f110_gym/`: The core F1TENTH Gymnasium environment.
- `f110_planning/`: A library of planning and tracking algorithms (Pure Pursuit, LQR, etc.).
- `scripts/`: Example scripts and simulation utilities.
- `data/`: Maps and waypoint files.

## Installation

We recommend using a virtual environment and installing both packages in editable mode.

```bash
# Clone the repository
git clone https://github.com/pganguli/f1tenth_ng.git
cd f1tenth_ng

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the gym and planning packages
pip install -e ./f110_gym
pip install -e ./f110_planning
```

## Usage Workflow

The repository supports a complete research workflow from data generation to model evaluation:

### 1. Data Generation

Generate datasets of LiDAR scans and ground truth labels (heading error, wall distances) using a waypoint follower with added noise.

```bash
python scripts/datagen/waypoint_datagen.py --map data/maps/F1/Oschersleben/Oschersleben_map --max-steps 10000
```

### 2. Combine Datasets

Merge multiple `.npz` files into a single training dataset with optional deduplication.

```bash
python scripts/datagen/combine_datasets.py data/datasets/file1.npz data/datasets/file2.npz --output data/datasets/combined.npz --dedup
```

### 3. Training

Train LiDAR-based neural networks (e.g., for heading error prediction or wall distance estimation) using PyTorch Lightning.

```bash
python scripts/train/train.py --config scripts/train/config_heading.yaml
```

You can monitor the training progress using TensorBoard:

```bash
tensorboard --logdir scripts/train/lightning_logs
```

### 4. Simulation & Evaluation

Test your planners (classic or DNN-based) in the simulation environment.

**Tracking Planners (Pure Pursuit, LQR, Stanley):**

```bash
python scripts/sim/tracking_planners.py --map data/maps/F1/Oschersleben/Oschersleben_map
```

**Reactive Planners (Gap Follower, Disparity Extender, LiDAR DNN):**

```bash
python scripts/sim/reactive_planners.py --planner dnn --map data/maps/F1/Oschersleben/Oschersleben_map
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

## Documentation

- [f110_gym README](f110_gym/README.md)
- [f110_planning README](f110_planning/README.md)
