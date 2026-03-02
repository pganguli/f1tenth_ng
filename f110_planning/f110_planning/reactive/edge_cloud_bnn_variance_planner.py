"""
Edge-cloud planner that triggers cloud calls when edge BNN variance is high.
"""

from collections import deque
from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner, UncertaintyThresholdScheduler
from .lidar_dnn_planner import LidarDNNPlanner


class EdgeCloudBNNVariancePlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """
    Edge-cloud hybrid planner with uncertainty-triggered cloud requests.

    The edge planner runs each step. A cloud request is sent when edge BNN
    uncertainty crosses a threshold, and the cloud response is fused after
    a configurable latency.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        cloud_latency: int = 30,
        alpha_steer: float = 0.7,
        alpha_speed: float = 0.7,
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        lateral_gain: float = 1.0,
        # Uncertainty-trigger policy
        uncertainty_threshold: float = 0.03,
        min_interval: int = 1,
        warmup_steps: int = 1,
        fallback_interval: int = 10,
        # Edge model
        edge_wall_model_path: Optional[str] = None,
        edge_heading_model_path: Optional[str] = None,
        edge_arch_id: int = 11,
        edge_heading_arch_id: Optional[int] = None,
        edge_mc_samples: int = 3,
        edge_bayes_inference_mode: str = "mc",
        # Cloud model
        cloud_wall_model_path: Optional[str] = None,
        cloud_heading_model_path: Optional[str] = None,
        cloud_arch_id: int = 13,
        cloud_heading_arch_id: Optional[int] = None,
        cloud_mc_samples: int = 3,
        cloud_bayes_inference_mode: str = "mc",
        # Shared uncertainty control knobs
        uncertainty_speed_gain: float = 0.6,
        uncertainty_lookahead_gain: float = 0.8,
        uncertainty_rise_delta: float = 0.002,
    ) -> None:
        self.cloud_latency = max(0, int(cloud_latency))
        self.alpha_steer = float(alpha_steer)
        self.alpha_speed = float(alpha_speed)

        self.scheduler = UncertaintyThresholdScheduler(
            threshold=uncertainty_threshold,
            min_interval=min_interval,
            warmup_steps=warmup_steps,
            fallback_interval=fallback_interval,
        )

        self.edge_planner = LidarDNNPlanner(
            wall_model_path=edge_wall_model_path,
            heading_model_path=edge_heading_model_path,
            arch_id=edge_arch_id,
            heading_arch_id=edge_heading_arch_id,
            lookahead_distance=lookahead_distance,
            max_speed=max_speed,
            lateral_gain=lateral_gain,
            mc_samples=edge_mc_samples,
            bayes_inference_mode=edge_bayes_inference_mode,
            uncertainty_speed_gain=uncertainty_speed_gain,
            uncertainty_lookahead_gain=uncertainty_lookahead_gain,
            uncertainty_rise_delta=uncertainty_rise_delta,
        )
        self.cloud_planner = LidarDNNPlanner(
            wall_model_path=cloud_wall_model_path,
            heading_model_path=cloud_heading_model_path,
            arch_id=cloud_arch_id,
            heading_arch_id=cloud_heading_arch_id,
            lookahead_distance=lookahead_distance,
            max_speed=max_speed,
            lateral_gain=lateral_gain,
            mc_samples=cloud_mc_samples,
            bayes_inference_mode=cloud_bayes_inference_mode,
            uncertainty_speed_gain=uncertainty_speed_gain,
            uncertainty_lookahead_gain=uncertainty_lookahead_gain,
            uncertainty_rise_delta=uncertainty_rise_delta,
        )

        self.last_target_point = self.edge_planner.last_target_point

        self._step: int = 0
        self._cloud_requests: deque[tuple[int, dict[str, Any]]] = deque()
        self._latest_cloud_action: Action | None = None

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        step = self._step

        while self._cloud_requests and self._cloud_requests[0][0] <= step:
            _, stale_obs = self._cloud_requests.popleft()
            self._latest_cloud_action = self.cloud_planner.plan(
                stale_obs, ego_idx=ego_idx
            )

        edge_action = self.edge_planner.plan(obs, ego_idx=ego_idx)

        context = {"edge_uncertainty": self.edge_planner.last_uncertainty}
        if self.scheduler.should_call_cloud(
            step, obs, self._latest_cloud_action, context=context
        ):
            obs_snapshot = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in obs.items()
            }
            self._cloud_requests.append((step + self.cloud_latency, obs_snapshot))

        self.last_target_point = self.edge_planner.last_target_point

        if self._latest_cloud_action is not None:
            action = self._blend(edge_action, self._latest_cloud_action)
        else:
            action = edge_action

        self._step += 1
        return action

    def _blend(self, edge: Action, cloud: Action) -> Action:
        steer = self.alpha_steer * cloud.steer + (1.0 - self.alpha_steer) * edge.steer
        speed = self.alpha_speed * cloud.speed + (1.0 - self.alpha_speed) * edge.speed
        return Action(steer=steer, speed=speed)

    def reset(self) -> None:
        self._step = 0
        self._cloud_requests.clear()
        self._latest_cloud_action = None
