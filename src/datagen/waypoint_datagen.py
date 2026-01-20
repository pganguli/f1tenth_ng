from argparse import Namespace

import numpy as np
import yaml
from f110_gym.envs.base_classes import Integrator

import gym

from planners import PurePursuitPlanner


def main(num_steps, save_path):
    """
    Generate a dataset of LiDAR scans and full vehicle state.

    Args:
        num_steps (int): number of simulation steps to run
        save_path (str): file path to save dataset
    """
    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.82461887897713965,
        "vgain": 1.375,
    }

    with open("data/config.yaml") as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, (0.17145 + 0.15875))

    def render_callback(env_renderer):
        # update camera to follow car
        e = env_renderer
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

    # Create environment
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

    # ---------------- Dataset storage ----------------
    num_lidar_beams = 1080
    dataset = {
        "lidar": np.zeros((num_steps, num_lidar_beams), dtype=np.float32),
        "left_wall_dist": np.zeros((num_steps,), dtype=np.float32),
        "right_wall_dist": np.zeros((num_steps,), dtype=np.float32),
        "track_width": np.zeros((num_steps,), dtype=np.float32),
        "x": np.zeros((num_steps,), dtype=np.float32),
        "y": np.zeros((num_steps,), dtype=np.float32),
        "theta": np.zeros((num_steps,), dtype=np.float32),
        "speed": np.zeros((num_steps,), dtype=np.float32),
        "steer": np.zeros((num_steps,), dtype=np.float32),
        "yaw_rate": np.zeros((num_steps,), dtype=np.float32),
        "accel": np.zeros((num_steps,), dtype=np.float32),
    }

    # ---------------- LiDAR geometry -----------------
    fov = 4.7  # radians
    angles = np.linspace(-fov / 2, fov / 2, num_lidar_beams)
    sin_angles = np.sin(angles)

    prev_speed = 0.0

    for k in range(num_steps):
        x_pos = obs["poses_x"][0]
        y_pos = obs["poses_y"][0]
        theta = obs["poses_theta"][0]

        speed_cmd, steer_cmd = planner.plan(
            x_pos, y_pos, theta, work["tlad"], work["vgain"]
        )

        obs, step_reward, done, info = env.step(np.array([[steer_cmd, speed_cmd]]))
        env.render(mode="human")

        # --- LiDAR ---
        lidar_scan = obs["scans"][0]  # shape (1080,)
        y_coords = lidar_scan * sin_angles
        left_dist = np.max(y_coords)
        right_dist = -np.min(y_coords)
        track_width = left_dist + right_dist

        # --- Vehicle state ---
        vx = obs["linear_vels_x"][0] if "linear_vels_x" in obs else speed_cmd
        yaw_rate = obs["ang_vels_z"][0] if "ang_vels_z" in obs else 0.0
        accel = (vx - prev_speed) / env.timestep if k > 0 else 0.0
        prev_speed = vx

        # --- Store in dataset ---
        dataset["lidar"][k] = lidar_scan
        dataset["left_wall_dist"][k] = left_dist
        dataset["right_wall_dist"][k] = right_dist
        dataset["track_width"][k] = track_width
        dataset["x"][k] = x_pos
        dataset["y"][k] = y_pos
        dataset["theta"][k] = theta
        dataset["speed"][k] = vx
        dataset["steer"][k] = steer_cmd
        dataset["yaw_rate"][k] = yaw_rate
        dataset["accel"][k] = accel

        if done:
            print(f"Episode finished at step {k}.")
            # Trim arrays if simulation ended early
            for key in dataset:
                dataset[key] = dataset[key][: k + 1]
            break

    np.savez_compressed(save_path, **dataset)
    print(f"Dataset saved to {save_path}")
    print(f"Samples collected: {dataset['lidar'].shape[0]}")

    env.close()


if __name__ == "__main__":
    main(num_steps=5000, save_path="data/dataset.npz")
