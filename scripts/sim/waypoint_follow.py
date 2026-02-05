#!/usr/bin/env python3

import time
from argparse import Namespace

import numpy as np
from f110_gym.envs.base_classes import Integrator

import gymnasium as gym
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import load_waypoints


def main():
    # Configuration for Example map
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

    from f110_planning.render_callbacks import camera_tracking, render_lidar, create_waypoint_renderer
    # Add callbacks
    env.unwrapped.add_render_callback(camera_tracking)
    env.unwrapped.add_render_callback(render_lidar)
    if waypoints.size > 0:
        render_waypoints = create_waypoint_renderer(waypoints)
        env.unwrapped.add_render_callback(render_waypoints)

    obs, info = env.reset(
        options={"poses": np.array([[conf.sx, conf.sy, conf.stheta]])}
    )
    env.render()

    laptime = 0.0
    start = time.time()

    done = False
    while not done:
        action = planner.plan(obs)
        speed, steer = action.speed, action.steer
        obs, step_reward, terminated, truncated, info = env.step(np.array([[steer, speed]]))
        done = terminated or truncated
        laptime += step_reward
        env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
