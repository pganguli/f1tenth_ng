import numpy as np
from f110_gym.envs.base_classes import Integrator
from pyglet import canvas as pyg_canvas
from pyglet import window as pyg_window

from gym import make as env_make


def env_init(init_x=0.0, init_y=0.0, init_theta=1.57):
    env = env_make(
        id="f110_gym:f110-v0",
        params={
            "mu": 1.0489,  # surface friction coefficient [-]
            "C_Sf": 4.718,  # Cornering stiffness coefficient, front [1/rad]
            "C_Sr": 5.4562,  # Cornering stiffness coefficient, rear [1/rad]
            "lf": 0.15875,  # Distance from center of gravity to front axle [m]
            "lr": 0.17145,  # Distance from center of gravity to rear axle [m]
            "h": 0.074,  # Height of center of gravity [m]
            "m": 3.74,  # Total mass of the vehicle [kg]
            "I": 0.04712,  # Moment of inertial of the entire vehicle about the z axis [kgm^2]
            "s_min": -0.4189,  # Minimum steering angle constraint [rad]
            "s_max": 0.4189,  # Maximum steering angle constraint [rad]
            "sv_min": -3.2,  # Minimum steering velocity constraint [rad/s]
            "sv_max": 3.2,  # Maximum steering velocity constraint [rad/s]
            "v_switch": 7.319,  # Switching velocity (velocity at which the acceleration is no longer able to create wheel spin) [m/s]
            "a_max": 9.51,  # Maximum longitudinal acceleration [m/s^2]
            "v_min": -5.0,  # Minimum longitudinal velocity [m/s]
            "v_max": 20.0,  # Maximum longitudinal velocity [m/s]
            "width": 0.31,  # width of the vehicle [m]
            "length": 0.58,  # length of the vehicle [m]
        },
        map="data/map",
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    obs, step_reward, done, info = env.reset(
        poses=np.array([[init_x, init_y, init_theta]])
    )
    return env, obs, step_reward, done, info


def kbd_init():
    display = pyg_canvas.get_display()
    keys = pyg_window.key.KeyStateHandler()
    windows = display.get_windows()
    if not windows:
        raise RuntimeError("No pyglet window found")
    windows[0].push_handlers(keys)
    return keys


def main():
    env, obs, step_reward, done, info = env_init()
    env.render()
    keys = kbd_init()

    # planner = planner()

    lap_time = 0.0
    while not done:
        # action = planner.plan(obs)
        ego_steer = 0
        ego_speed = 0
        if keys[pyg_window.key.W]:
            ego_speed = 1
        if keys[pyg_window.key.A]:
            ego_steer = 1
        if keys[pyg_window.key.D]:
            ego_steer = -1
        action = np.array(
            [
                [
                    np.clip(ego_steer, env.params["s_min"], env.params["s_max"]),
                    np.clip(ego_speed, env.params["v_min"], env.params["v_max"]),
                ]
            ]
        )
        obs, step_reward, done, info = env.step(action)
        print(
            f"{np.histogram(obs['scans'], bins=5)[0]=}",
            f"{obs['poses_x']=}",
            f"{obs['poses_y']=}",
            f"{obs['poses_theta']=}",
            f"{obs['linear_vels_x']=}",
            f"{obs['linear_vels_y']=}",
            f"{obs['ang_vels_z']=}",
            sep="\t",
        )
        lap_time += step_reward
        env.render()


if __name__ == "__main__":
    main()
