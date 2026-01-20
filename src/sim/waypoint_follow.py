import time
from argparse import Namespace

import numpy as np
import yaml
from f110_gym.envs.base_classes import Integrator

import gym

from planners import PurePursuitPlanner


def main():
    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.82461887897713965,
        "vgain": 1.375,
    }  # 0.90338203837889}

    with open("data/config.yaml") as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(
        conf, (0.17145 + 0.15875)
    )  # FlippyPlanner(speed=0.2, flip_every=1, steer=10)

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta]])
    )
    env.render()

    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = planner.plan(
            obs["poses_x"][0],
            obs["poses_y"][0],
            obs["poses_theta"][0],
            work["tlad"],
            work["vgain"],
        )
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode="human")

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
