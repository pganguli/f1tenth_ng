#!/usr/bin/env python3
from argparse import Namespace
import os
import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import load_waypoints, get_side_distances, get_heading_error


import numpy as np
import csv


MAX_STEPS = 500
NUM_LIDAR_POINTS = 1080
CSV_PATH = "lidar_wall_data.csv"

def run_data_collection(max_steps, csv_path, map_path, waypoint_path):
    # Configuration
    conf = Namespace(
        map_path=map_path,
        map_ext=".png",
        sx=0.7,
        sy=0.0,
        stheta=1.37079632679,
    )

    waypoints = load_waypoints(waypoint_path)
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
        options={"poses": np.array([[conf.sx, conf.sy, conf.stheta]])}
    )

    laptime = 0.0

    header = (
        [f"lidar_{i}" for i in range(NUM_LIDAR_POINTS)]
        + ["left_wall_dist", "right_wall_dist", "heading_error"]
    )

    if not os.path.exists(csv_path):
        output_dir = "./dnn_data"
        os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        step = 0
        done = False

        while step < max_steps and not done:
            action = planner.plan(obs, ego_idx=0)
            speed, steer = action.speed, action.steer

            obs, step_reward, terminated, truncated, info = env.step(
                np.array([[steer, speed]])
            )

            scan = obs["scans"][0]
            left_dist, right_dist = get_side_distances(scan)
            car_position = np.array([obs["poses_x"][0], obs["poses_y"][0]])
            heading_error = get_heading_error(
                waypoints,
                car_position,
                obs["poses_theta"][0]
            )

            row = list(scan) + [left_dist, right_dist, heading_error]
            writer.writerow(row)

            done = terminated or truncated
            laptime += step_reward
            step += 1

    env.close()
    print(f"Saved {step} samples to {csv_path}")

def main():
    run_data_collection(
        max_steps=10000,
        csv_path="./dnn_data/training_data.csv",
        map_path="data/maps/Example/Example",
        waypoint_path="data/maps/Example/Example_raceline.csv",
    )
    run_data_collection(
        max_steps=500,
        csv_path="./dnn_data/testing_data.csv",
        map_path="data/maps/Example/Example",
        waypoint_path="data/maps/Example/Example_raceline.csv",
    )


if __name__ == "__main__":
    main()
