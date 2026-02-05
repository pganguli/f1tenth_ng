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
git clone https://github.com/f1tenth/f1tenth_ng.git
cd f1tenth_ng

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the gym and planning packages
pip install -e ./f110_gym
pip install -e ./f110_planning
```

## Quickstart: Waypoint Following

This example demonstrates how to use `f110_gym` with a planner from `f110_planning` to follow a pre-defined raceline.

```python
import gymnasium as gym
import numpy as np
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import load_waypoints

# 1. Create the environment
env = gym.make('f110_gym:f110-v0', 
               map='data/maps/Example/Example', 
               render_mode='human', 
               num_agents=1)

# 2. Load waypoints using the utility function
waypoints = load_waypoints('data/maps/Example/Example_raceline.csv')

# 3. Initialize the planner
planner = PurePursuitPlanner(waypoints=waypoints)

# 4. Reset and run the simulation loop
obs, info = env.reset(options={'poses': np.array([[0.7, 0.0, 1.37]])})
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

## Citing
If you find this repository useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
