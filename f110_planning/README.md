# F1TENTH Planning Algorithms

This repository contains planning and tracking algorithms for the F1TENTH Gym environment. It has been updated to support **Gymnasium** and **Pyglet 2.x**.

## Core API

### Planners

All planners inherit from `BasePlanner` and implement a `.plan(obs)` method.

```python
from f110_planning.tracking import PurePursuitPlanner

planner = PurePursuitPlanner(waypoints=waypoints)
action = planner.plan(obs) # returns f110_planning.base.Action
```

- **Tracking**: `PurePursuitPlanner`, `LQRPlanner`, `StanleyPlanner`. Requires waypoints.
- **Reactive**: `GapFollowerPlanner`, `DisparityExtenderPlanner`, `BubblePlanner`, `DynamicWaypointPlanner`, `LidarDNNPlanner`. Map-agnostic obstacle avoidance.
- **Misc**: `HybridPlanner` (Manual override), `ManualPlanner`, `RandomPlanner`.

### Actuation

Planners return an `Action` object:

```python
@dataclass
class Action:
    steer: float # Steering angle in radians
    speed: float # Velocity in m/s
```

## Features

- **Tracking Algorithms**:
  - `PurePursuitPlanner`: Classic geometric path tracking.
  - `LQRPlanner`: Linear Quadratic Regulator based tracking.
  - `StanleyPlanner`: Stanley controller for precise path following.
- **Reactive Algorithms**:
  - `GapFollowerPlanner`: Robust Follow the Gap algorithm (FGM).
  - `DisparityExtenderPlanner`: Disparity extension with obstacle inflation.
  - `BubblePlanner`: Local repulsion from nearest obstacles.
  - `DynamicWaypointPlanner`: Adaptive reactive waypoint generation.
  - `LidarDNNPlanner`: End-to-end steering prediction using trained models.
- **Utilities**:
  - `waypoint_utils`: Loading, interpolation, and nearest-point search.
  - `geometry_utils`: Coordinate transforms and Numba-optimized distance checks.
  - `sim_utils`: Standardized environment setup and CLI argument parsing.

## Installation

We recommend installing the package in editable mode inside your virtual environment.

```bash
# Clone the repository (if not already done)
git clone https://github.com/pganguli/f1tenth_ng.git
cd f1tenth_ng/f110_planning

# Install the package
pip install -e .
```

## Quickstart

### Loading Waypoints

You can use the built-in utility to load waypoints from a CSV or TSV file:

```python
from f110_planning.utils import load_waypoints

# Load waypoints
waypoints = load_waypoints('data/maps/F1/Oschersleben/Oschersleben_centerline.tsv')
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
