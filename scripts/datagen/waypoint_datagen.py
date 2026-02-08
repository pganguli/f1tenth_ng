#!/usr/bin/env python3
from argparse import Namespace

import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import load_waypoints
from f110_planning.utils import get_side_distances


import numpy as np
import csv



MAX_STEPS = 500
NUM_LIDAR_POINTS = 1080
CSV_PATH = "lidar_wall_data.csv"

def main():
    conf = Namespace(
        map_path="data/maps/Example/Example",
        map_ext=".png",
        sx=0.7,
        sy=0.0,
        stheta=1.37079632679,
    )

    waypoints = load_waypoints("data/maps/Example/Example_raceline.csv")

    planner = PurePursuitPlanner(waypoints=waypoints)

    env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode="human_fast",
        render_fps=60,
    )
    obs, info = env.reset(
    options={"poses": np.array([[conf.sx, conf.sy, conf.stheta]])})

    laptime = 0.0

    header = (
        [f"lidar_{i}" for i in range(NUM_LIDAR_POINTS)]
        + ["left_wall_dist", "right_wall_dist", "yaw"]
    )

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        step = 0
        done = False
        while step < MAX_STEPS and not done:
            action = planner.plan(obs)
            speed, steer = action.speed, action.steer
            obs, step_reward, terminated, truncated, info = env.step(
                np.array([[steer, speed]])
            )

            scan = obs["scans"][0]
            left_dist, right_dist = get_side_distances(scan)
            row = list(scan) + [left_dist, right_dist]
            writer.writerow(row)
            done = terminated or truncated
            laptime += step_reward

            step += 1

    env.close()
    print(f"Saved {step} samples to {CSV_PATH}")


if __name__ == "__main__":
    main()
