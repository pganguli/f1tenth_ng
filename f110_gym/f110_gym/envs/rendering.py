"""
Rendering engine for f1tenth gym env based on pyglet and OpenGL
Author: Hongrui Zheng
Updated for pyglet 2.x compatibility
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyglet

if os.environ.get("DISPLAY") is None:
    pyglet.options["headless"] = True

# pylint: disable=wrong-import-position
import yaml
from PIL import Image

# In pyglet 2.x, legacy GL functions need to be imported from gl_compat or use ctypes
# We use pyglet's built-in projection handling instead
try:
    from pyglet.gl import Config

    _HAS_GL = True
except (ImportError, Exception):  # pylint: disable=broad-exception-caught
    _HAS_GL = False

    class Config:
        """Stub for pyglet.gl.Config if OpenGL is unavailable."""

        # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            pass

# helpers
from .collision_models import get_vertices
# pylint: enable=wrong-import-position

# zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31


class CarShape:
    """
    Custom shape class for rendering cars as quads using pyglet 2.x shapes API.
    """

    def __init__(
        self,
        vertices: list[float],
        color: tuple[int, int, int],
        batch: pyglet.graphics.Batch,
    ) -> None:
        """
        Create a car shape from 4 corner vertices.

        Args:
            vertices: List of 8 floats [x1, y1, x2, y2, x3, y3, x4, y4]
            color: RGB tuple (r, g, b)
            batch: pyglet Batch to add the shape to
        """
        self._batch = batch
        self._color = color
        self._vertices_data = vertices

        # Create two triangles to form a quad (pyglet 2.x doesn't have GL_QUADS directly)
        # Triangle 1: vertices 0, 1, 2
        # Triangle 2: vertices 0, 2, 3
        self._triangles: list[pyglet.shapes.Triangle] = []
        self._update_triangles()

    def _update_triangles(self) -> None:
        """Update the triangle shapes based on current vertices."""
        # Delete old triangles
        for tri in self._triangles:
            tri.delete()
        self._triangles = []

        v = self._vertices_data
        if len(v) >= 8:
            # Triangle 1: v0, v1, v2
            tri1 = pyglet.shapes.Triangle(
                v[0],
                v[1],  # vertex 0
                v[2],
                v[3],  # vertex 1
                v[4],
                v[5],  # vertex 2
                color=self._color,
                batch=self._batch,
            )
            # Triangle 2: v0, v2, v3
            tri2 = pyglet.shapes.Triangle(
                v[0],
                v[1],  # vertex 0
                v[4],
                v[5],  # vertex 2
                v[6],
                v[7],  # vertex 3
                color=self._color,
                batch=self._batch,
            )
            self._triangles = [tri1, tri2]

    @property
    def vertices(self) -> list[float]:
        """Get the current vertices of the car shape."""
        return self._vertices_data

    @vertices.setter
    def vertices(self, value: list[float]) -> None:
        self._vertices_data = value
        self._update_triangles()

    def delete(self) -> None:
        """Clean up and delete all sub-shapes associated with this car."""
        for tri in self._triangles:
            tri.delete()
        self._triangles = []


@dataclass
class CameraViewport:
    """Ortho bounds for the camera."""

    left: float
    right: float
    bottom: float
    top: float


@dataclass
class RendererCamera:
    """Camera and view state for the renderer."""

    viewport: CameraViewport
    zoom_level: float
    zoomed_width: int
    zoomed_height: int
    x: float
    y: float
    rotation: float = 0.0


@dataclass
class RendererMap:
    """Map-related data and shapes."""

    points: np.ndarray | None
    shapes: list[pyglet.shapes.Circle]


@dataclass
class RendererSim:
    """Simulation-related data (agent poses, scans, etc.)."""

    poses: np.ndarray | None
    ego_idx: int
    cars: list[CarShape]
    scans: list[np.ndarray] | None


@dataclass
class RendererUI:
    """UI elements displayed over the simulation."""

    score_label: pyglet.text.Label
    fps_display: pyglet.window.FPSDisplay


class EnvRenderer(pyglet.window.Window if _HAS_GL else object):
    """
    F1TENTH Environment Renderer.

    A window class inheriting from pyglet.window.Window that handles the camera,
    coordinate projections, and object rendering for the simulation.

    Features:
    - Support for Pyglet 2.x graphics API.
    - Ego-centric camera following with rotation capability.
    - Automatic coordinate scaling (50.0 pixels per meter).
    - Map and LIDAR scan visualization.
    """

    def __init__(self, width: int, height: int, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the renderer.

        Args:
            width (int): Window width in pixels.
            height (int): Window height in pixels.
            *args: Additional arguments passed to the pyglet Window.
            **kwargs: Additional keyword arguments passed to the pyglet Window.
        """
        if not _HAS_GL:
            raise ImportError(
                "Rendering is unavailable. This usually happens in headless "
                "environments where EGL/GL libraries are missing. Please check "
                "if libEGL and libGL are installed."
            )
        conf = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
        super().__init__(
            width, height, config=conf, resizable=True, vsync=False, *args, **kwargs
        )

        # Set background color
        pyglet.gl.glClearColor(9 / 255, 32 / 255, 87 / 255, 1.0)

        # pylint: disable=unreachable
        self.camera = RendererCamera(
            viewport=CameraViewport(
                left=-width / 2,
                right=width / 2,
                bottom=-height / 2,
                top=height / 2,
            ),
            zoom_level=1.2,
            zoomed_width=width,
            zoomed_height=height,
            x=0.0,
            y=0.0,
        )

        # batches for graphics and UI
        self.batch = pyglet.graphics.Batch()
        self.ui_batch = pyglet.graphics.Batch()

        # map and simulation state
        self.map_state = RendererMap(points=None, shapes=[])
        self.sim_state = RendererSim(poses=None, ego_idx=0, cars=[], scans=None)

        # projection matrix
        self.projection = pyglet.math.Mat4()

        # UI elements
        score_label = pyglet.text.Label(
            f"Lap Time: {0.0:.2f}, Ego Lap Count: {0.0:.0f}",
            font_size=24,
            x=width - 20,
            y=height - 20,
            anchor_x="right",
            anchor_y="top",
            color=(255, 255, 255, 255),
        )

        fps_display = pyglet.window.FPSDisplay(self)
        fps_display.label.x = 20
        fps_display.label.y = height - 20
        fps_display.label.anchor_y = "top"
        fps_display.label.font_size = 16
        fps_display.label.color = (200, 200, 200, 255)

        self.ui = RendererUI(score_label=score_label, fps_display=fps_display)

    @property
    def cars(self) -> list[CarShape]:
        """Backwards compatibility for cars list."""
        return self.sim_state.cars

    @property
    def scans(self) -> list[np.ndarray] | None:
        """Backwards compatibility for scans list."""
        return self.sim_state.scans

    @property
    def poses(self) -> np.ndarray | None:
        """Backwards compatibility for poses."""
        return self.sim_state.poses

    @property
    def left(self) -> float:
        """Backwards compatibility for viewport left bound."""
        return self.camera.viewport.left

    @left.setter
    def left(self, value: float) -> None:
        self.camera.viewport.left = value

    @property
    def right(self) -> float:
        """Backwards compatibility for viewport right bound."""
        return self.camera.viewport.right

    @right.setter
    def right(self, value: float) -> None:
        self.camera.viewport.right = value

    @property
    def top(self) -> float:
        """Backwards compatibility for viewport top bound."""
        return self.camera.viewport.top

    @top.setter
    def top(self, value: float) -> None:
        self.camera.viewport.top = value

    @property
    def bottom(self) -> float:
        """Backwards compatibility for viewport bottom bound."""
        return self.camera.viewport.bottom

    @bottom.setter
    def bottom(self, value: float) -> None:
        self.camera.viewport.bottom = value

    def update_map(self, map_path: str, map_ext: str) -> None:
        """
        Update the map being drawn by the renderer. Converts image to a list of 3D points
        representing each obstacle pixel in the map.

        Args:
            map_path (str): absolute path to the map without extensions
            map_ext (str): extension for the map image file

        Returns:
            None
        """

        # load map metadata
        map_metadata = self._load_map_metadata(map_path)

        # load and process map image to get obstacle points
        map_points = self._get_map_points(map_path, map_ext, map_metadata)

        # Clear old map point shapes
        for shape in self.map_state.shapes:
            shape.delete()
        self.map_state.shapes = []

        # Create circle shapes for map points (pyglet 2.x compatible)
        # Using small circles to represent points
        point_color = (183, 193, 222)
        for i in range(map_points.shape[0]):
            point = pyglet.shapes.Circle(
                x=map_points[i, 0],
                y=map_points[i, 1],
                radius=1.0,  # Small radius for point-like appearance
                color=point_color,
                batch=self.batch,
            )
            self.map_state.shapes.append(point)

        self.map_state.points = map_points

    def on_resize(self, width: int, height: int) -> None:
        """
        Callback function on window resize, overrides inherited method, and updates camera values
        on top of the inherited on_resize() method.

        Potential improvements on current behavior: zoom/pan resets on window resize.

        Args:
            width (int): new width of window
            height (int): new height of window

        Returns:
            None
        """

        # call overrided function
        super().on_resize(width, height)

        # update camera value
        (width, height) = self.get_size()
        cam = self.camera
        cam.viewport.left = -cam.zoom_level * width / 2
        cam.viewport.right = cam.zoom_level * width / 2
        cam.viewport.bottom = -cam.zoom_level * height / 2
        cam.viewport.top = cam.zoom_level * height / 2
        cam.zoomed_width = int(cam.zoom_level * width)
        cam.zoomed_height = int(cam.zoom_level * height)

        # Update UI element positions
        ui = self.ui
        ui.score_label.x = width - 20
        ui.score_label.y = height - 20

        ui.fps_display.label.x = 20
        ui.fps_display.label.y = height - 20

    def on_mouse_drag(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        x: int,  # pylint: disable=unused-argument
        y: int,  # pylint: disable=unused-argument
        dx: int,
        dy: int,
        buttons: int,  # pylint: disable=unused-argument
        modifiers: int,  # pylint: disable=unused-argument
    ) -> None:
        """
        Callback function on mouse drag, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (int): Relative X position from the previous mouse position.
            dy (int): Relative Y position from the previous mouse position.
            buttons (int): Bitwise combination of the mouse buttons currently pressed.
            modifiers (int): Bitwise combination of any keyboard modifiers currently active.

        Returns:
            None
        """

        # pan camera
        cam = self.camera
        if cam.rotation == 0.0:
            cam.x -= dx * cam.zoom_level
            cam.y -= dy * cam.zoom_level
            cam.viewport.left -= dx * cam.zoom_level
            cam.viewport.right -= dx * cam.zoom_level
            cam.viewport.bottom -= dy * cam.zoom_level
            cam.viewport.top -= dy * cam.zoom_level
        else:
            # Handle rotation in panning
            s = np.sin(-cam.rotation)
            c = np.cos(-cam.rotation)
            cam.x -= (dx * c - dy * s) * cam.zoom_level
            cam.y -= (dx * s + dy * c) * cam.zoom_level

    def on_mouse_scroll(
        self,
        x: int,
        y: int,
        scroll_x: float,  # pylint: disable=unused-argument
        scroll_y: float,
    ) -> None:
        """
        Callback function on mouse scroll, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (float): Amount of movement on the horizontal axis.
            dy (float): Amount of movement on the vertical axis.

        Returns:
            None
        """

        # Get scale factor
        f = ZOOM_IN_FACTOR if scroll_y > 0 else ZOOM_OUT_FACTOR if scroll_y < 0 else 1

        # If zoom_level is in the proper range
        cam = self.camera
        if 0.01 < cam.zoom_level * f < 10:
            cam.zoom_level *= f

            (width, height) = self.get_size()

            mouse_x = x / width
            mouse_y = y / height

            mouse_x_in_world = cam.viewport.left + mouse_x * cam.zoomed_width
            mouse_y_in_world = cam.viewport.bottom + mouse_y * cam.zoomed_height

            cam.zoomed_width = int(cam.zoomed_width * f)
            cam.zoomed_height = int(cam.zoomed_height * f)

            cam.viewport.left = mouse_x_in_world - mouse_x * cam.zoomed_width
            cam.viewport.right = mouse_x_in_world + (1 - mouse_x) * cam.zoomed_width
            cam.viewport.bottom = mouse_y_in_world - mouse_y * cam.zoomed_height
            cam.viewport.top = mouse_y_in_world + (1 - mouse_y) * cam.zoomed_height

    def on_close(self) -> None:
        """
        Callback function when the 'x' is clicked on the window, overrides inherited method.
        Also throws exception to end the python program when in a loop.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError: with a message that indicates the rendering window was closed
        """

        super().on_close()
        raise RuntimeError("Rendering window was closed.")

    def on_draw(self) -> None:
        """
        Function when the pyglet is drawing. The function draws the batch created that includes
        the map points, the agent polygons, the information text, and the fps display.

        Args:
            None

        Returns:
            None
        """

        # if map and poses doesn't exist, raise exception
        if self.map_state.points is None:
            raise RuntimeError("Map not set for renderer.")
        if self.sim_state.poses is None:
            raise RuntimeError("Agent poses not updated for renderer.")

        # Clear window
        self.clear()

        # Set up the projection for 2D rendering with camera
        cam = self.camera
        ortho = pyglet.math.Mat4.orthogonal_projection(
            cam.viewport.left,
            cam.viewport.right,
            cam.viewport.bottom,
            cam.viewport.top,
            -1000,
            1000,
        )

        if cam.rotation != 0.0:
            # Create translation matrices to rotate around camera center
            # However, the camera viewport is already defined around the center
            # so we just need to translate the world by -camera.x, -camera.y
            trans_to_origin = pyglet.math.Mat4.from_translation(
                pyglet.math.Vec3(-cam.x, -cam.y, 0)
            )
            rot_mat = pyglet.math.Mat4.from_rotation(cam.rotation, (0, 0, 1))

            # Combine: Translate -> Rotate -> Ortho projection
            self.projection = ortho @ rot_mat @ trans_to_origin
        else:
            # Still need to translate to center on camera x,y even if no rotation
            trans_to_origin = pyglet.math.Mat4.from_translation(
                pyglet.math.Vec3(-cam.x, -cam.y, 0)
            )
            self.projection = ortho @ trans_to_origin

        # Draw all batches
        with self.window_block():
            self.batch.draw()

        # Set up screen space projection for UI elements
        self.projection = pyglet.math.Mat4.orthogonal_projection(
            0, self.width, 0, self.height, -1, 1
        )

        # Draw UI batch
        self.ui_batch.draw()

        # Draw UI elements (score label and fps) in screen space
        ui = self.ui
        ui.score_label.draw()
        ui.fps_display.draw()

    def window_block(self) -> "_ProjectionContext":
        """Context manager for setting window projection"""
        return _ProjectionContext(self)

    def update_obs(self, obs: dict[str, Any]) -> None:
        """
        Updates the renderer with the latest observation from the gym environment,
        including the agent poses, and the information text.

        Args:
            obs (dict): observation dict from the gym env

        Returns:
            None
        """

        sim = self.sim_state
        sim.scans = obs.get("scans")
        sim.ego_idx = obs["ego_idx"]
        poses_x = obs["poses_x"]
        poses_y = obs["poses_y"]
        poses_theta = obs["poses_theta"]

        num_agents = len(poses_x)
        if sim.poses is None:
            sim.cars = []
            for i in range(num_agents):
                if i == sim.ego_idx:
                    vertices_np = get_vertices(
                        np.array([0.0, 0.0, 0.0]), CAR_LENGTH, CAR_WIDTH
                    )
                    vertices = vertices_np.flatten().tolist()
                    car = CarShape(
                        vertices=vertices,
                        color=(172, 97, 185),  # Ego car color
                        batch=self.batch,
                    )
                    sim.cars.append(car)
                else:
                    vertices_np = get_vertices(
                        np.array([0.0, 0.0, 0.0]), CAR_LENGTH, CAR_WIDTH
                    )
                    vertices = vertices_np.flatten().tolist()
                    car = CarShape(
                        vertices=vertices,
                        color=(99, 52, 94),  # Other car color
                        batch=self.batch,
                    )
                    sim.cars.append(car)

        poses = np.stack((poses_x, poses_y, poses_theta)).T
        for j in range(poses.shape[0]):
            vertices_np = 50.0 * get_vertices(poses[j, :], CAR_LENGTH, CAR_WIDTH)
            vertices = vertices_np.flatten().tolist()
            sim.cars[j].vertices = vertices
        sim.poses = poses

        ui = self.ui
        ui.score_label.text = (
            f"Lap Time: {obs['lap_times'][0]:.2f}, "
            f"Ego Lap Count: {obs['lap_counts'][obs['ego_idx']]:.0f}"
        )

    def _load_map_metadata(self, map_path: str) -> tuple[float, float, float]:
        """Helper to load map metadata from yaml file."""
        try:
            with open(map_path + ".yaml", "r", encoding="utf-8") as yaml_stream:
                metadata = yaml.safe_load(yaml_stream)
                resolution = metadata["resolution"]
                origin = metadata["origin"]
                return resolution, origin[0], origin[1]
        except (yaml.YAMLError, IOError, KeyError) as ex:
            print(f"Error loading map metadata: {ex}")
            raise

    def _get_map_points(
        self,
        map_path: str,
        map_ext: str,
        metadata: tuple[float, float, float],
    ) -> np.ndarray:
        """Helper to process map image and return obstacle coordinates."""
        map_img = np.array(
            Image.open(map_path + map_ext).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        ).astype(np.float64)
        map_height, map_width = map_img.shape

        # convert map pixels to coordinates
        resolution, origin_x, origin_y = metadata
        map_x, map_y = np.meshgrid(np.arange(map_width), np.arange(map_height))
        map_x = (map_x * resolution + origin_x).flatten()
        map_y = (map_y * resolution + origin_y).flatten()
        map_coords = np.vstack((map_x, map_y, np.zeros(map_y.shape)))

        # mask and only leave the obstacle points
        return 50.0 * map_coords[:, (map_img == 0.0).flatten()].T


class _ProjectionContext:
    """Context manager for temporarily setting window projection."""

    def __init__(self, window: EnvRenderer) -> None:
        self.window = window

    def __enter__(self) -> "_ProjectionContext":
        # Store old view and set new projection
        self.window.view = pyglet.math.Mat4()
        return self

    def __exit__(self, *args: Any) -> None:
        pass
