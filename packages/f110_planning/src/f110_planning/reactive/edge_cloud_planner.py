"""
Edge-Cloud hybrid DNN planner for F1TENTH.

Wraps two :class:`LidarDNNPlanner` instances – a lightweight **edge** model
called every time-step and a heavier **cloud** model whose result arrives with
a configurable latency of *N* simulation steps.  The two predicted features
(left-wall distance, track width, heading error) are blended before the final
action is computed.
"""

from collections import deque
from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner, CloudScheduler
from ..schedulers import FixedIntervalScheduler
from ..utils import get_reactive_action
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
    alpha_left, alpha_track, alpha_heading : float
        Blending weights for the predicted reactive features.  For backwards
        compatibility, ``alpha_steer`` and ``alpha_speed`` are still accepted
        and used as fallbacks when these are not provided.
    deviation_steer_weight, deviation_speed_weight : float
        Relative weights used when converting edge-vs-held-cloud action
        disagreement into a scalar scheduler deviation signal.

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

    # Empirical MSE constants for the canonical per-feature model pairs:
    # edge: arch1 (left), arch2 (track_width), arch2 (heading_error)
    # cloud: arch5 (left), arch7 (track_width), arch6 (heading_error)
    _SIGMA2_EDGE: tuple[float, float, float] = (0.028020, 0.036518, 0.019371)
    _SIGMA2_CLOUD: tuple[float, float, float] = (0.000518, 0.001539, 0.001140)

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        # ---- edge-cloud knobs ----
        cloud_latency: int = 10,
        alpha_steer: float = 0.7,
        alpha_speed: float = 0.7,
        alpha_left: float | None = None,
        alpha_track: float | None = None,
        alpha_heading: float | None = None,
        sigma_proc_left: float | None = None,
        sigma_proc_track: float | None = None,
        sigma_proc_heading: float | None = None,
        age_decay_lambda: float = 0.0,
        deviation_steer_weight: float = 1.0,
        deviation_speed_weight: float = 1.0,
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
        self.alpha_steer = alpha_steer
        self.alpha_speed = alpha_speed
        self.alpha_left = alpha_left if alpha_left is not None else alpha_speed
        self.alpha_track = alpha_track if alpha_track is not None else alpha_speed
        self.alpha_heading = alpha_heading if alpha_heading is not None else alpha_steer
        self._static_alphas = [self.alpha_left, self.alpha_track, self.alpha_heading]
        self._sigma_proc = [sigma_proc_left, sigma_proc_track, sigma_proc_heading]
        self._age_decay_lambda = age_decay_lambda
        self.deviation_steer_weight = deviation_steer_weight
        self.deviation_speed_weight = deviation_speed_weight
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
        self._cloud_requests: deque[
            tuple[int, dict[str, Any], tuple[float, float, float]]
        ] = deque()
        self._latest_cloud_action: Action | None = None
        self._latest_cloud_features: tuple[float, float, float] | None = None
        self._cloud_last_updated: int = -1
        # record whether a cloud call was requested on the last invocation
        self.last_cloud_call: bool = False
        self.last_cloud_age: list[int] = [999, 999, 999]
        self._prev_scheduler_edge_features: tuple[float, float, float] | None = None
        self._prev_returned_action: Action | None = None

    # ------------------------------------------------------------------
    # BasePlanner interface
    # ------------------------------------------------------------------
    @staticmethod
    def _action_tuple(action: Action | None) -> tuple[float, float] | None:
        if action is None:
            return None
        return (float(action.steer), float(action.speed))

    def _resolve_action(
        self,
        obs: dict[str, Any],
        ego_idx: int,
        edge_action: Action,
    ) -> Action:
        """Return the final action after blending the latest cloud features, if any."""
        if self._latest_cloud_features is None:
            return edge_action

        c_left, c_track, c_head = self._latest_cloud_features
        age = self.last_cloud_age[0]
        alpha_left, alpha_track, alpha_heading = self._resolved_alphas(age)
        left = alpha_left * c_left + (1.0 - alpha_left) * self.edge_planner.last_left_dist
        track = alpha_track * c_track + (1.0 - alpha_track) * self.edge_planner.last_track_width
        heading = (
            alpha_heading * c_head
            + (1.0 - alpha_heading) * self.edge_planner.last_heading_error
        )
        right = max(track - left, 0.0)

        car_theta = obs["poses_theta"][ego_idx]
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
        return action

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Compute the fused edge-cloud action for the current observation.

        Call this once per simulation step.  Internally it:

        1. Checks whether any in-flight cloud response has arrived.
        2. Runs the edge planner on the *current* observation.
        3. Uses the scheduler to decide whether to issue a cloud request.
        4. Returns either the pure edge action (if no cloud result yet) or
           the feature-level blend of cloud and edge predictions.
        """
        step = self._step

        # 1. Edge action (always latest obs)
        edge_action = self.edge_planner.plan(obs, ego_idx=ego_idx)

        # Keep render-callback pointer in sync
        self.last_target_point = self.edge_planner.last_target_point

        # 2. Receive any cloud result whose arrival time has been reached
        cloud_received = False
        while self._cloud_requests and self._cloud_requests[0][0] <= step:
            cloud_received = True
            _, stale_obs, enqueued_edge = self._cloud_requests.popleft()
            self._latest_cloud_action = self.cloud_planner.plan(
                stale_obs, ego_idx=ego_idx
            )
            self._latest_cloud_features = tuple(
                float(cloud_now + (edge_now - edge_then))
                for cloud_now, edge_now, edge_then in zip(
                    (
                        self.cloud_planner.last_left_dist,
                        self.cloud_planner.last_track_width,
                        self.cloud_planner.last_heading_error,
                    ),
                    (
                        self.edge_planner.last_left_dist,
                        self.edge_planner.last_track_width,
                        self.edge_planner.last_heading_error,
                    ),
                    enqueued_edge,
                )
            )
            self._cloud_last_updated = step

        self._update_cloud_age(step)

        edge_features = (
            float(self.edge_planner.last_left_dist),
            float(self.edge_planner.last_track_width),
            float(self.edge_planner.last_heading_error),
        )
        base_context = {
            "step": int(step),
            "ego_speed": float(obs["linear_vels_x"][ego_idx]),
            "edge_features": edge_features,
            "prev_edge_features": self._prev_scheduler_edge_features,
            "edge_action": self._action_tuple(edge_action),
            "prev_command": self._action_tuple(self._prev_returned_action),
            "cloud_age": int(self.last_cloud_age[0]),
            "cloud_in_flight": bool(self._cloud_requests),
            "cloud_queue_depth": int(len(self._cloud_requests)),
            "cloud_last_updated_step": int(self._cloud_last_updated),
            "cloud_received": bool(cloud_received),
        }
        if self._latest_cloud_action is not None:
            base_context["deviation"] = self._action_distance(
                edge_action,
                self._latest_cloud_action,
            )

        if self.cloud_latency <= 0:
            # Preserve the existing zero-latency fast path semantics: the
            # scheduler sees the pre-immediate-cloud command context.
            context = dict(base_context)
            context["current_command"] = self._action_tuple(edge_action)
            self.last_cloud_call = bool(
                self.scheduler.should_call_cloud(
                    step,
                    obs,
                    self._latest_cloud_action,
                    context=context,
                )
            )
            if self.last_cloud_call:
                obs_snapshot = {
                    k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in obs.items()
                }
                self._latest_cloud_action = self.cloud_planner.plan(
                    obs_snapshot, ego_idx=ego_idx
                )
                self._latest_cloud_features = (
                    self.cloud_planner.last_left_dist,
                    self.cloud_planner.last_track_width,
                    self.cloud_planner.last_heading_error,
                )
                self._cloud_last_updated = step
                self._update_cloud_age(step)
            action = self._resolve_action(obs, ego_idx, edge_action)
        else:
            action = self._resolve_action(obs, ego_idx, edge_action)
            context = dict(base_context)
            context["current_command"] = self._action_tuple(action)
            self.last_cloud_call = bool(
                self.scheduler.should_call_cloud(
                    step,
                    obs,
                    self._latest_cloud_action,
                    context=context,
                )
            )
            if self.last_cloud_call:
                obs_snapshot = {
                    k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in obs.items()
                }
                self._cloud_requests.append(
                    (
                        step + self.cloud_latency,
                        obs_snapshot,
                        edge_features,
                    )
                )

        self._prev_scheduler_edge_features = edge_features
        self._prev_returned_action = action
        self._step += 1
        return action

    def _action_distance(self, edge: Action, cloud: Action) -> float:
        steer_term = self.deviation_steer_weight * abs(edge.steer - cloud.steer)
        speed_term = self.deviation_speed_weight * abs(edge.speed - cloud.speed)
        return float(steer_term + speed_term)

    def _resolved_alphas(self, age: int) -> tuple[float, float, float]:
        """Return static or anchored sigma-age-decayed per-feature cloud weights."""
        values: list[float] = []
        for idx, sigma_proc in enumerate(self._sigma_proc):
            if sigma_proc is None or self._age_decay_lambda <= 0.0:
                values.append(self._static_alphas[idx])
            else:
                base = self._SIGMA2_EDGE[idx] + self._SIGMA2_CLOUD[idx]
                denom = base + self._age_decay_lambda * (sigma_proc * sigma_proc) * age
                decay = base / denom if denom > 0.0 else 1.0
                values.append(max(0.0, min(1.0, self._static_alphas[idx] * decay)))
        return (values[0], values[1], values[2])

    def _update_cloud_age(self, step: int) -> None:
        """Refresh the public age counter for the last received cloud result."""
        age = step - self._cloud_last_updated if self._cloud_last_updated >= 0 else 999
        self.last_cloud_age = [age, age, age]

    @staticmethod
    def _alpha_age(age: int, sigma_e2: float, sigma_c2: float, sigma_proc2: float) -> float:
        """Compute the selective-planner age-dependent cloud blend weight."""
        denom = sigma_e2 + sigma_c2 + sigma_proc2 * age
        return sigma_e2 / denom if denom > 0.0 else 0.5

    def reset(self) -> None:
        """Reset internal step counter and in-flight cloud requests."""
        self._step = 0
        self._cloud_requests.clear()
        self._latest_cloud_action = None
        self._latest_cloud_features = None
        self._cloud_last_updated = -1
        self.last_cloud_call = False
        self.last_cloud_age = [999, 999, 999]
        self._prev_scheduler_edge_features = None
        self._prev_returned_action = None
