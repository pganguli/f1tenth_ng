"""
Edge-Cloud hybrid DNN planner for F1TENTH.

Wraps two :class:`LidarDNNPlanner` instances – a lightweight **edge** model
called every time-step and a heavier **cloud** model whose result arrives with
a configurable latency of *N* simulation steps.  The two actions are blended
with independent weights for steering and speed.
"""

from collections import deque
from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner
from .lidar_dnn_planner import LidarDNNPlanner


class EdgeCloudPlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """
    Hybrid edge-cloud reactive planner.

    Parameters
    ----------
    cloud_latency : int
        Round-trip latency in simulation steps for cloud inference. A cloud request issued at step *t* yields a result that becomes available at step *t + cloud_latency*.
    scheduler : Optional[CloudScheduler]
        Scheduler object that decides whether to issue a cloud request at each step. Defaults to AlwaysCallScheduler (calls cloud every step).
    alpha_steer : float
        Blending weight for steering (0 = edge only, 1 = cloud only).
    alpha_speed : float
        Blending weight for speed (0 = edge only, 1 = cloud only).

    The remaining keyword arguments configure the two underlying
    :class:`LidarDNNPlanner` instances.  ``edge_*`` prefixed arguments are
    forwarded to the edge planner and ``cloud_*`` prefixed arguments to the
    cloud planner.  ``lookahead_distance``, ``max_speed``, and
    ``lateral_gain`` are shared by both unless overridden per-planner.
    """

    def __init__(
        self,
        # ---- edge-cloud knobs ----
        cloud_latency: int = 30,
        alpha_steer: float = 0.7,
        alpha_speed: float = 0.7,
        scheduler: Optional["CloudScheduler"] = None,
        # ---- shared defaults ----
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        lateral_gain: float = 1.0,
        # ---- edge model paths / arch ----
        edge_wall_model_path: Optional[str] = None,
        edge_heading_model_path: Optional[str] = None,
        edge_arch_id: int = 8,
        edge_heading_arch_id: Optional[int] = None,
        # ---- cloud model paths / arch ----
        cloud_wall_model_path: Optional[str] = None,
        cloud_heading_model_path: Optional[str] = None,
        cloud_arch_id: int = 10,
        cloud_heading_arch_id: Optional[int] = None,
    ) -> None:
        from ..base import AlwaysCallScheduler, CloudScheduler

        self.cloud_latency = cloud_latency
        self.alpha_steer = alpha_steer
        self.alpha_speed = alpha_speed
        self.scheduler = scheduler if scheduler is not None else AlwaysCallScheduler()

        self.edge_planner = LidarDNNPlanner(
            wall_model_path=edge_wall_model_path,
            heading_model_path=edge_heading_model_path,
            arch_id=edge_arch_id,
            heading_arch_id=edge_heading_arch_id,
            lookahead_distance=lookahead_distance,
            max_speed=max_speed,
            lateral_gain=lateral_gain,
        )
        self.cloud_planner = LidarDNNPlanner(
            wall_model_path=cloud_wall_model_path,
            heading_model_path=cloud_heading_model_path,
            arch_id=cloud_arch_id,
            heading_arch_id=cloud_heading_arch_id,
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
           the weighted blend of cloud and edge.
        """
        step = self._step

        # 1. Use scheduler to decide whether to issue a cloud request
        if self.scheduler.should_call_cloud(step, obs, self._latest_cloud_action):
            obs_snapshot = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in obs.items()
            }
            self._cloud_requests.append((step + self.cloud_latency, obs_snapshot))

        # 2. Receive any cloud result whose arrival time has been reached
        while self._cloud_requests and self._cloud_requests[0][0] <= step:
            _, stale_obs = self._cloud_requests.popleft()
            self._latest_cloud_action = self.cloud_planner.plan(
                stale_obs, ego_idx=ego_idx
            )

        # 3. Edge action (always latest obs)
        edge_action = self.edge_planner.plan(obs, ego_idx=ego_idx)

        # Keep render-callback pointer in sync
        self.last_target_point = self.edge_planner.last_target_point

        # 4. Blend
        if self._latest_cloud_action is not None:
            action = self._blend(edge_action, self._latest_cloud_action)
        else:
            action = edge_action

        self._step += 1
        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _blend(self, edge: Action, cloud: Action) -> Action:
        steer = self.alpha_steer * cloud.steer + (1.0 - self.alpha_steer) * edge.steer
        speed = self.alpha_speed * cloud.speed + (1.0 - self.alpha_speed) * edge.speed
        return Action(steer=steer, speed=speed)

    def reset(self) -> None:
        """Reset internal step counter and in-flight cloud requests."""
        self._step = 0
        self._cloud_requests.clear()
        self._latest_cloud_action = None
