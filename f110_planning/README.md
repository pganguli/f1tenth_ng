# F1TENTH Planning Algorithms

This repository contains planning and tracking algorithms for the F1TENTH Gym environment. It has been updated to support **Gymnasium** and **Pyglet 2.x**.

## Features
- **Tracking Algorithms**: Pure Pursuit, LQR, etc.
- **Reactive Algorithms**: Follow the Gap, etc.
- **Utilities**: Waypoint loading, coordinate transformations, and geometry utilities.

## Installation

We recommend installing the package in editable mode inside your virtual environment.

```bash
# Clone the repository (if not already done)
git clone https://github.com/f1tenth/f1tenth_ng.git
cd f1tenth_ng/f110_planning

# Install the package
pip install -e .
```

## Quickstart

### Loading Waypoints
You can use the built-in utility to load waypoints from a CSV file (e.g., a raceline):

```python
from f110_planning.utils import load_waypoints

# Load waypoints (reordered to [x, y, v, th])
waypoints = load_waypoints('data/maps/Example/Example_raceline.csv')
```

### Using a Planner (Pure Pursuit)
Here is a minimal example of using the `PurePursuitPlanner`:

```python
from f110_planning.tracking import PurePursuitPlanner

# Initialize the planner
planner = PurePursuitPlanner(waypoints=waypoints)

# Plan an action based on observation
# obs is the observation dictionary from f110_gym
action = planner.plan(obs)

# action.speed and action.steer are the resulting actuation values
```

## Related Repositories
- [f110_gym](https://github.com/f1tenth/f1tenth_gym): The F1TENTH Gymnasium environment.

## Citing
If you find this planning library useful, please consider citing:

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
