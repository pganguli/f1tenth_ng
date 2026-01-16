import gym
import yaml
import time
import numpy as np
import pyglet

from argparse import Namespace
from f110_gym.envs.base_classes import Integrator


# =========================
# Load config
# =========================
with open('config_example_map.yaml') as f:
    conf = Namespace(**yaml.load(f, Loader=yaml.FullLoader))


# =========================
# Environment
# =========================
env = gym.make(
    'f110_gym:f110-v0',
    map=conf.map_path,
    map_ext=conf.map_ext,
    num_agents=1,
    timestep=0.01,
    integrator=Integrator.RK4
)

obs, _, done, _ = env.reset(
    np.array([[conf.sx, conf.sy, conf.stheta]])
)

env.render(mode='human')


# =========================
# Keyboard setup (robust)
# =========================
def get_pyglet_window():
    display = pyglet.canvas.get_display()
    windows = display.get_windows()
    if not windows:
        raise RuntimeError("No pyglet window found")
    return windows[0]

window = get_pyglet_window()
keys = pyglet.window.key.KeyStateHandler()
window.push_handlers(keys)


# =========================
# Control parameters
# =========================
speed = 0.0
steer = 0.0

MAX_SPEED = 5.0
MAX_STEER = 0.6
ACCEL_STEP = 0.2
STEER_STEP = 0.05
STEER_RETURN = 0.9   # auto-centering factor
SPEED_DECAY = 0.99   # friction


# =========================
# Main loop
# =========================
start = time.time()

while not done:
    # --- Inputs ---
    if keys[pyglet.window.key.W]:
        speed += ACCEL_STEP
    if keys[pyglet.window.key.S]:
        speed -= ACCEL_STEP
    if keys[pyglet.window.key.A]:
        steer += STEER_STEP
    if keys[pyglet.window.key.D]:
        steer -= STEER_STEP
    if keys[pyglet.window.key.ESCAPE]:
        break

    # --- Passive dynamics ---
    steer *= STEER_RETURN
    speed *= SPEED_DECAY

    # --- Clamp ---
    speed = np.clip(speed, -MAX_SPEED, MAX_SPEED)
    steer = np.clip(steer, -MAX_STEER, MAX_STEER)

    # --- Step ---
    obs, reward, done, info = env.step(np.array([[steer, speed]]))

    # LiDAR data
    lidar_scan = obs['scans']
    num_beams = np.size(lidar_scan)
    fov = 4.7  # radians
    angles = np.linspace(-fov/2, fov/2, num_beams)  # -FOV/2 = leftmost, +FOV/2 = rightmost

    # Convert polar coordinates to y (lateral) distances
    y_coords = lidar_scan * np.sin(angles)

    left_wall = np.max(y_coords)   # furthest left point
    right_wall = np.min(y_coords)  # furthest right point (negative)
    track_width = left_wall - right_wall  # total width

    print(f"{left_wall=}")
    print(f"{right_wall=}")
    print(f"{track_width=}")

    env.render(mode='human')


print("Simulation ended.")
print("Real elapsed time:", time.time() - start)
