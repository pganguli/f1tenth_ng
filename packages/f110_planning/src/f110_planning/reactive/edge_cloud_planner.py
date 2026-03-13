"""
Edge-Cloud hybrid DNN planner for F1TENTH.

Wraps two :class:`LidarDNNPlanner` instances – a lightweight **edge** model
called every time-step and a heavier **cloud** model whose result arrives with
a configurable latency of *N* simulation steps.  The two predicted features
(left-wall distance, track width, heading error) are blended per-feature
before the final action is computed.
"""

from collections import deque
from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner, CloudScheduler
from ..schedulers import FixedIntervalScheduler
from ..utils.reactive_utils import get_reactive_action
from .lidar_dnn_planner import LidarDNNPlanner


class EdgeCloudPlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """
    Hybrid edge-cloud reactive planner.

    Parameters
    ----------
    cloud_latency : int
        Round-trip latency in simulation steps for cloud inference.
        A cloud request issued at step *t* yields a result that becomes
        available at step *t + cloud_latency*.
    scheduler : Optional[CloudScheduler]
        scheduler object that decides whether to issue a cloud request
        at each step. Defaults to ``FixedIntervalScheduler(interval=1)`` (calls cloud
        every step).
    alpha_left : float
        Blending weight for left-wall distance (0 = edge only, 1 = cloud only).
    alpha_track : float
        Blending weight for track width (0 = edge only, 1 = cloud only).
    alpha_heading : float
        Blending weight for heading error (0 = edge only, 1 = cloud only).
    sigma_proc_left, sigma_proc_track, sigma_proc_heading : float | None
        Process-noise standard deviations for each feature.  Accepted for
        interface compatibility with :class:`SelectiveEdgeCloudPlanner` but
        not used in this simpler, non-age-weighted planner.

    Note
    ----
    The remaining keyword arguments configure the two underlying
    :class:`LidarDNNPlanner` instances.  ``edge_*`` prefixed arguments are
    forwarded to the edge planner and ``cloud_*`` prefixed arguments to the
    cloud planner.  ``lookahead_distance``, ``max_speed``, and
    ``lateral_gain`` are shared by both unless overridden per-planner.

    Three separate single-output model files are required per planner:
    ``*_left_wall_model_path`` (left wall distance),
    ``*_track_width_model_path`` (track width = left + right), and
    ``*_heading_model_path`` (heading error).  Right wall distance is derived
    at inference time as ``track_width - left_dist``.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        # ---- edge-cloud knobs ----
        cloud_latency: int = 10,
        alpha_left: float = 0.5,
        alpha_track: float = 0.5,
        alpha_heading: float = 0.5,
        sigma_proc_left: Optional[float] = None,   # accepted for interface compat
        sigma_proc_track: Optional[float] = None,
        sigma_proc_heading: Optional[float] = None,
        scheduler: Optional[CloudScheduler] = None,
        # ---- shared defaults ----
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        lateral_gain: float = 1.0,
        # ---- edge model paths ----
        edge_left_wall_model_path: Optional[str] = None,
        edge_track_width_model_path: Optional[str] = None,
        edge_heading_model_path: Optional[str] = None,
        # ---- cloud model paths ----
        cloud_left_wall_model_path: Optional[str] = None,
        cloud_track_width_model_path: Optional[str] = None,
        cloud_heading_model_path: Optional[str] = None,
    ) -> None:
        # CloudScheduler / FixedIntervalScheduler imported at module level

        self.cloud_latency = cloud_latency
        self.alpha_left = alpha_left
        self.alpha_track = alpha_track
        self.alpha_heading = alpha_heading
        self.scheduler = (
            scheduler if scheduler is not None else FixedIntervalScheduler(interval=1)
        )

        self.edge_planner = LidarDNNPlanner(
            left_model_path=edge_left_wall_model_path,
            track_width_model_path=edge_track_width_model_path,
            heading_model_path=edge_heading_model_path,
            lookahead_distance=lookahead_distance,
            max_speed=max_speed,
            lateral_gain=lateral_gain,
        )
        self.cloud_planner = LidarDNNPlanner(
            left_model_path=cloud_left_wall_model_path,
            track_width_model_path=cloud_track_width_model_path,
            heading_model_path=cloud_heading_model_path,
            lookahead_distance=lookahead_distance,
            max_speed=max_speed,
            lateral_gain=lateral_gain,
        )

        # Expose last_target_point from the edge planner so that render
        # callbacks (e.g. create_dynamic_waypoint_renderer) work.
        self.last_target_point = self.edge_planner.last_target_point

        # Internal state
        self._step: int = 0
        self._cloud_requests: deque[tuple[int, dict[str, Any]]] = deque()
        self._latest_cloud_action: Action | None = None
        # Latest cloud-predicted features (left_dist, track_width, heading_error)
        self._latest_cloud_features: tuple[float, float, float] | None = None
        # record whether a cloud call was requested on the last invocation
        self.last_cloud_call: bool = False

    @property
    def last_call_mask(self) -> list[bool]:
        """Expose a per-DNN mask compatible with CloudCallCountMetric.

        EdgeCloudPlanner calls all three DNN features together (left-wall,
        track-width, heading) as a single cloud request, so the mask is
        uniform: all True when a call was made, all False otherwise.
        """
        return [self.last_cloud_call] * 3

    # ------------------------------------------------------------------
    # BasePlanner interface
    # ------------------------------------------------------------------
    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Compute the fused edge-cloud action for the current observation.

        Call this once per simulation step.  Internally it:

        1. Uses the scheduler to decide whether to issue a cloud request.
        2. Checks whether any in-flight cloud response has arrived.
        3. Runs the edge planner on the *current* observation.
        4. Returns either the pure edge action (if no cloud result yet) or
           the feature-level blend of cloud and edge predictions.
        """
        step = self._step

        # 1. Use scheduler to decide whether to issue a cloud request
        self.last_cloud_call = bool(
            self.scheduler.should_call_cloud(step, obs, self._latest_cloud_action)
        )
        if self.last_cloud_call:
            obs_snapshot = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in obs.items()
            }
            self._cloud_requests.append((step + self.cloud_latency, obs_snapshot))

        # 2. Receive any cloud result whose arrival time has been reached
        while self._cloud_requests and self._cloud_requests[0][0] <= step:
            _, stale_obs = self._cloud_requests.popleft()
            self._latest_cloud_action = self.cloud_planner.plan(stale_obs, ego_idx=ego_idx)
            self._latest_cloud_features = (
                self.cloud_planner.last_left_dist,
                self.cloud_planner.last_track_width,
                self.cloud_planner.last_heading_error,
            )

        # 3. Edge prediction (always latest obs)
        edge_action = self.edge_planner.plan(obs, ego_idx=ego_idx)
        self.last_target_point = self.edge_planner.last_target_point

        # 4. Blend at feature level when a cloud prediction is available
        if self._latest_cloud_features is not None:
            c_left, c_track, c_head = self._latest_cloud_features
            left    = self.alpha_left    * c_left  + (1.0 - self.alpha_left)    * self.edge_planner.last_left_dist
            track   = self.alpha_track   * c_track + (1.0 - self.alpha_track)   * self.edge_planner.last_track_width
            heading = self.alpha_heading * c_head  + (1.0 - self.alpha_heading) * self.edge_planner.last_heading_error
            right   = max(track - left, 0.0)

            car_theta    = obs["poses_theta"][ego_idx]
            car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
            current_speed = obs["linear_vels_x"][ego_idx]

            action = get_reactive_action(
                self.edge_planner,
                left_dist=left,
                right_dist=right,
                heading_error=heading,
                car_position=car_position,
                car_theta=car_theta,
                current_speed=current_speed,
            )
            self.last_target_point = self.edge_planner.last_target_point
        else:
            action = edge_action

        self._step += 1
        return action

    def reset(self) -> None:
        """Reset internal step counter and in-flight cloud requests."""
        self._step = 0
        self._cloud_requests.clear()
        self._latest_cloud_action = None
        self._latest_cloud_features = None
