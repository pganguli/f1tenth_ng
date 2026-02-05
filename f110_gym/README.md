# The F1TENTH Gym Environment (Gymnasium)

This repository contains the F1TENTH Gym environment, now updated to support **Gymnasium** and **Pyglet 2.x**.

## Major Updates
- **Gymnasium Migration**: The environment now follows the Gymnasium API (v0.26+).
- **Pyglet 2.x Support**: Rendering has been updated to support modern Pyglet versions.
- **Improved Integration**: Standardized observation and action spaces.

## Quickstart

We recommend installing the environment inside a virtual environment.

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the environment in editable mode
pip install -e .
```

### 2. Usage Example

The environment follows the standard Gymnasium `reset` and `step` loop:

```python
import gymnasium as gym
import numpy as np

# Create the environment
env = gym.make('f110_gym:f110-v0', map='data/maps/Example/Example', render_mode='human', num_agents=1)

# Reset the environment
obs, info = env.reset(options={'poses': np.array([[0.0, 0.0, 0.0]])})

done = False
while not done:
    # Sample random action: [[steer, speed]]
    action = np.array([[0.0, 1.0]]) 
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
```

## Citing
If you find this Gym environment useful, please consider citing:

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
