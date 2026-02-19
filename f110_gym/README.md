# The F1TENTH Gym Environment (Gymnasium)

This repository contains the F1TENTH Gym environment, updated to support **Gymnasium** and **Pyglet 2.x**.

## Features

- **Gymnasium API**: Full compatibility with the modern `gymnasium` interface.
- **Environment Specs**:
  - **Action Space**: `Box(-1.0, 1.0, (num_agents, 2))` representing `[steering_angle, velocity]`.
  - **Observation Space**: A dictionary containing:
    - `scans`: LiDAR scans.
    - `poses_x`, `poses_y`, `poses_theta`: Agent poses.
    - `linear_vel_x`, `linear_vel_y`, `ang_vel_z`: Agent velocities.
- **Rendering**: Enhanced visualization support with Pyglet 2.x, including camera tracking and custom callbacks.

## Installation

We recommend installing the environment in editable mode inside a virtual environment.

```bash
# From the root of the repository
pip install -e ./f110_gym
```

## Usage Example

The environment follows the standard Gymnasium loop:

```python
import gymnasium as gym
import numpy as np

# Create the environment
env = gym.make('f110_gym:f110-v0', 
               map='data/maps/F1/Oschersleben/Oschersleben_map', 
               render_mode='human', 
               num_agents=1)

# Reset with initial poses
obs, info = env.reset(options={'poses': np.array([[0.0, 0.0, 2.85]])})

done = False
while not done:
    # Action: [[steer, speed]]
    action = np.array([[0.0, 2.0]]) 
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
```

## Citing

If you find this Gym environment useful, please consider citing:

```bibtex
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
