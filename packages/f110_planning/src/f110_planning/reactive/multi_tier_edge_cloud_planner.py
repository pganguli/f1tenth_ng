"""
Multi-tier edge-cloud reactive planner for F1TENTH.

Runs an edge planner every step and can request either a medium or large cloud
planner depending on a tier-aware scheduler. Feature predictions are blended
before computing the final control action.
"""

from collections import deque
from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner, CloudTier, TieredCloudScheduler
from ..utils import get_reactive_action
from .lidar_dnn_planner import LidarDNNPlanner


class MultiTierEdgeCloudPlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """Hybrid planner with edge, medium-cloud, and large-cloud DNN models."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        scheduler: TieredCloudScheduler,
        medium_cloud_latency: int = 6,
        large_cloud_latency: int = 10,
        alpha_steer_medium: float = 0.5,
        alpha_speed_medium: float = 0.1,
        alpha_steer_large: float = 0.7,
        alpha_speed_large: float = 0.2,
        alpha_left_medium: float | None = None,
        alpha_track_medium: float | None = None,
        alpha_heading_medium: float | None = None,
        alpha_left_large: float | None = None,
        alpha_track_large: float | None = None,
        alpha_heading_large: float | None = None,
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        lateral_gain: float = 1.0,
        edge_left_wall_model_path: Optional[str] = None,
        edge_track_width_model_path: Optional[str] = None,
        edge_heading_model_path: Optional[str] = None,
        medium_left_wall_model_path: Optional[str] = None,
        medium_track_width_model_path: Optional[str] = None,
        medium_heading_model_path: Optional[str] = None,
        large_left_wall_model_path: Optional[str] = None,
        large_track_width_model_path: Optional[str] = None,
        large_heading_model_path: Optional[str] = None,
        oracle_disagreement_routing: bool = False,
        tier_steer_weight: float = 1.0,
        tier_speed_weight: float = 0.0,
    ) -> None:
        self.scheduler = scheduler
        self.medium_cloud_latency = medium_cloud_latency
        self.large_cloud_latency = large_cloud_latency
        self.alpha_steer_medium = alpha_steer_medium
        self.alpha_speed_medium = alpha_speed_medium
        self.alpha_steer_large = alpha_steer_large
        self.alpha_speed_large = alpha_speed_large
        self.alpha_left_medium = (
            alpha_left_medium if alpha_left_medium is not None else alpha_speed_medium
        )
        self.alpha_track_medium = (
            alpha_track_medium if alpha_track_medium is not None else alpha_speed_medium
        )
        self.alpha_heading_medium = (
            alpha_heading_medium
            if alpha_heading_medium is not None
            else alpha_steer_medium
        )
        self.alpha_left_large = (
            alpha_left_large if alpha_left_large is not None else alpha_speed_large
        )
        self.alpha_track_large = (
            alpha_track_large if alpha_track_large is not None else alpha_speed_large
        )
        self.alpha_heading_large = (
            alpha_heading_large
            if alpha_heading_large is not None
            else alpha_steer_large
        )
        self.oracle_disagreement_routing = oracle_disagreement_routing
        self.tier_steer_weight = tier_steer_weight
        self.tier_speed_weight = tier_speed_weight

        planner_kwargs = {
            "lookahead_distance": lookahead_distance,
            "max_speed": max_speed,
            "lateral_gain": lateral_gain,
        }
        self.edge_planner = LidarDNNPlanner(
            left_model_path=edge_left_wall_model_path,
            track_width_model_path=edge_track_width_model_path,
            heading_model_path=edge_heading_model_path,
            **planner_kwargs,
        )
        self.medium_cloud_planner = LidarDNNPlanner(
            left_model_path=medium_left_wall_model_path,
            track_width_model_path=medium_track_width_model_path,
            heading_model_path=medium_heading_model_path,
            **planner_kwargs,
        )
        self.large_cloud_planner = LidarDNNPlanner(
            left_model_path=large_left_wall_model_path,
            track_width_model_path=large_track_width_model_path,
            heading_model_path=large_heading_model_path,
            **planner_kwargs,
        )

        self.last_target_point = self.edge_planner.last_target_point
        self._step = 0
        self._requests: dict[CloudTier, deque[tuple[int, dict[str, Any]]]] = {
            CloudTier.MEDIUM: deque(),
            CloudTier.LARGE: deque(),
        }
        self._latest_cloud_actions: dict[CloudTier, Action | None] = {
            CloudTier.MEDIUM: None,
            CloudTier.LARGE: None,
        }
        self._latest_cloud_features: dict[CloudTier, tuple[float, float, float] | None] = {
            CloudTier.MEDIUM: None,
            CloudTier.LARGE: None,
        }
        self._active_cloud_tier: CloudTier | None = None
        self.last_cloud_call = False
        self.last_cloud_tier_called: CloudTier | None = None

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """Compute the fused multi-tier edge/cloud action for the current observation."""
        step = self._step
        self._receive_due_cloud_results(step, ego_idx)

        edge_action = self.edge_planner.plan(obs, ego_idx=ego_idx)

        context = {
            "ego_speed": float(obs["linear_vels_x"][ego_idx]),
        }
        if "crosstrack_dist" in obs:
            try:
                context["deviation"] = abs(float(obs["crosstrack_dist"][ego_idx]))
            except (TypeError, ValueError, IndexError):
                pass

        if self.oracle_disagreement_routing:
            probe_action = self.medium_cloud_planner.plan(obs, ego_idx=ego_idx)
            context["deviation"] = self._action_distance(edge_action, probe_action)

        requested_tier = self.scheduler.choose_cloud_tier(
            step=step,
            obs=obs,
            latest_cloud_actions=self._latest_cloud_actions,
            context=context,
        )
        self.last_cloud_call = requested_tier is not None
        self.last_cloud_tier_called = requested_tier

        if requested_tier is not None:
            obs_snapshot = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in obs.items()
            }
            arrival_step = step + self._latency_for_tier(requested_tier)
            self._requests[requested_tier].append((arrival_step, obs_snapshot))

        self.last_target_point = self.edge_planner.last_target_point
        action = self._fused_action(obs, ego_idx, edge_action)
        self._step += 1
        return action

    def _receive_due_cloud_results(self, step: int, ego_idx: int) -> None:
        for tier in (CloudTier.MEDIUM, CloudTier.LARGE):
            planner = self._planner_for_tier(tier)
            queue = self._requests[tier]
            while queue and queue[0][0] <= step:
                _, stale_obs = queue.popleft()
                self._latest_cloud_actions[tier] = planner.plan(stale_obs, ego_idx=ego_idx)
                self._latest_cloud_features[tier] = (
                    planner.last_left_dist,
                    planner.last_track_width,
                    planner.last_heading_error,
                )
                self._active_cloud_tier = tier

    def _fused_action(self, obs: dict[str, Any], ego_idx: int, edge_action: Action) -> Action:
        if self._active_cloud_tier == CloudTier.LARGE:
            large_features = self._latest_cloud_features[CloudTier.LARGE]
            if large_features is not None:
                return self._blend_features(obs, ego_idx, large_features, CloudTier.LARGE)

        if self._active_cloud_tier == CloudTier.MEDIUM:
            medium_features = self._latest_cloud_features[CloudTier.MEDIUM]
            if medium_features is not None:
                return self._blend_features(obs, ego_idx, medium_features, CloudTier.MEDIUM)

        if self._latest_cloud_features[CloudTier.LARGE] is not None:
            return self._blend_features(
                obs,
                ego_idx,
                self._latest_cloud_features[CloudTier.LARGE],
                CloudTier.LARGE,
            )

        if self._latest_cloud_features[CloudTier.MEDIUM] is not None:
            return self._blend_features(
                obs,
                ego_idx,
                self._latest_cloud_features[CloudTier.MEDIUM],
                CloudTier.MEDIUM,
            )

        return edge_action

    def _blend_features(
        self,
        obs: dict[str, Any],
        ego_idx: int,
        cloud_features: tuple[float, float, float],
        tier: CloudTier,
    ) -> Action:
        if tier == CloudTier.MEDIUM:
            alpha_left = self.alpha_left_medium
            alpha_track = self.alpha_track_medium
            alpha_heading = self.alpha_heading_medium
        else:
            alpha_left = self.alpha_left_large
            alpha_track = self.alpha_track_large
            alpha_heading = self.alpha_heading_large

        c_left, c_track, c_heading = cloud_features
        left = alpha_left * c_left + (1.0 - alpha_left) * self.edge_planner.last_left_dist
        track = (
            alpha_track * c_track
            + (1.0 - alpha_track) * self.edge_planner.last_track_width
        )
        heading = (
            alpha_heading * c_heading
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

    def _latency_for_tier(self, tier: CloudTier) -> int:
        return self.medium_cloud_latency if tier == CloudTier.MEDIUM else self.large_cloud_latency

    def _planner_for_tier(self, tier: CloudTier) -> LidarDNNPlanner:
        return self.medium_cloud_planner if tier == CloudTier.MEDIUM else self.large_cloud_planner

    def _action_distance(self, edge: Action, cloud: Action) -> float:
        steer_term = self.tier_steer_weight * abs(float(edge.steer) - float(cloud.steer))
        speed_term = self.tier_speed_weight * abs(float(edge.speed) - float(cloud.speed))
        return float(steer_term + speed_term)

    def reset(self) -> None:
        """Reset internal state and in-flight requests."""
        self._step = 0
        for queue in self._requests.values():
            queue.clear()
        self._latest_cloud_actions = {
            CloudTier.MEDIUM: None,
            CloudTier.LARGE: None,
        }
        self._latest_cloud_features = {
            CloudTier.MEDIUM: None,
            CloudTier.LARGE: None,
        }
        self._active_cloud_tier = None
        self.last_cloud_call = False
        self.last_cloud_tier_called = None
