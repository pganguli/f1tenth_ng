#!/usr/bin/env python3

import time
from argparse import Namespace

import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.tracking import DNNPlanner
from f110_planning.utils import load_waypoints, get_side_distances, get_heading_error
from keras.models import load_model

multiple_models = {
    "w": list(),
    "p": list(),
    "rot": list()
}

def loadModel(nummodel):
    for i in range(1, nummodel+1):
        multiple_models["w"].append(load_model(f"./models/lidar_model1_{i}.keras"))
        multiple_models["p"].append(load_model(f"./models/lidar_model2_{i}.keras"))
        multiple_models["rot"].append(load_model(f"./models/lidar_model3_{i}.keras"))

def getMultipleModelOutput(obs, a, b, c):
    model1 = multiple_models["w"][a - 1]
    model2 = multiple_models["p"][b - 1]
    model3 = multiple_models["rot"][c - 1]
    lidarDistances = np.array(obs).reshape((1, len(obs)))
    val1 = model1.predict(lidarDistances, verbose=None)[0]
    val2 = model2.predict(lidarDistances, verbose=None)[0]
    val3 = model3.predict(lidarDistances, verbose=None)[0]
    val = np.concatenate((val1, val2, val3), axis=0)
    return val


def main():
    conf = Namespace(
        map_path="data/maps/Example/Example",
        map_ext=".png",
        sx=0.7,
        sy=0.0,
        stheta=1.37079632679,
    )

    waypoints_orig = load_waypoints("data/maps/Example/Example_raceline.csv")

    planner = DNNPlanner()
    num=2
    loadModel(num)

    env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode="human_fast",
        render_fps=60,
        max_laps=None,  # Run forever, don't terminate on lap completion
    )

    from f110_planning.render_callbacks import (
        camera_tracking,
        create_trace_renderer,
        create_waypoint_renderer,
        render_lidar,
        render_side_distances,
        create_heading_error_renderer,
    )

    env.unwrapped.add_render_callback(camera_tracking)
    env.unwrapped.add_render_callback(render_lidar)
    env.unwrapped.add_render_callback(render_side_distances)
    env.unwrapped.add_render_callback(
        create_trace_renderer(color=(255, 255, 0), max_points=10000)
    )
    
    heading_error_renderer = create_heading_error_renderer(waypoints_orig, agent_idx=0)
    env.unwrapped.add_render_callback(heading_error_renderer)
    
    if waypoints_orig.size > 0:
        render_waypoints = create_waypoint_renderer(
            waypoints_orig, color=(255, 255, 255, 64)
        )
        env.unwrapped.add_render_callback(render_waypoints)

    obs, info = env.reset(
        options={"poses": np.array([[conf.sx, conf.sy, conf.stheta]])}
    )
    env.render()

    laptime = 0.0
    start = time.time()

    done = False
    while not done:
        scan = obs['scans'][0]
        left_dist, right_dist = get_side_distances(scan)
        car_position = np.array([obs["poses_x"][0], obs["poses_y"][0]])
        heading_error = get_heading_error(
                waypoints_orig,
                car_position,
                obs["poses_theta"][0]
            )
        currentLidarModel = [left_dist, right_dist,heading_error]
        action = planner.plan(obs['scans'][0], currentLidarModel)
        speed, steer = action.speed, action.steer
        obs, step_reward, terminated, truncated, info = env.step(
            np.array([[steer, speed]])
        )
        done = terminated or truncated
        laptime += float(step_reward)
        env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()