"""
Selective Edge-Cloud hybrid DNN planner for F1TENTH.

Unlike :class:`~f110_planning.reactive.EdgeCloudPlanner` which schedules a
single binary call/no-call decision, this planner exposes **per-DNN** cloud
calling.  At each step the caller provides a ``call_mask`` specifying which of
the m=3 cloud DNNs (left-wall, track-width, heading-error) to call.

Only the *selected* DNNs incur a cloud round-trip; the others use their last
received cloud result ("held").  Before any cloud result arrives for a given
DNN, the corresponding edge output is used as a seamless fallback so control is
always well-defined.

The resolved (left, track, heading) triple is used to compute a "cloud action"
(steer + speed) via the same reactive controller as the edge planner.  The two
actions are then blended with :attr:`alpha_steer` / :attr:`alpha_speed`.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner
from ..utils import F110_WHEELBASE, get_reactive_actuation
from .lidar_dnn_planner import LidarDNNPlanner


class SelectiveEdgeCloudPlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """Hybrid edge-cloud planner with independent per-DNN call scheduling.

    Parameters
    ----------
    cloud_latency : int
        Round-trip latency in simulation steps for any cloud DNN.
    alpha_steer : float
        Blending weight for steering (0 = edge only, 1 = cloud only).
    alpha_speed : float
        Blending weight for speed (0 = edge only, 1 = cloud only).
    top_k : int
        Maximum number of DNNs that may be called per step (informational;
        the mask is enforced by the caller / environment).
    lookahead_distance, max_speed, lateral_gain : float
        Shared reactive-controller parameters forwarded to both planners.
    edge_*_model_path : str | None
        Paths to TorchScript ``.pt`` models for the edge planner.
    cloud_*_model_path : str | None
        Paths to TorchScript ``.pt`` models for the cloud planner.
    """

    # DNN slot indices — used as list indices throughout
    LEFT = 0
    TRACK = 1
    HEADING = 2
    NUM_DNNS = 3

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        cloud_latency: int = 10,
        alpha_steer: float = 0.7,
        alpha_speed: float = 0.7,
        top_k: int = 1,
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        lateral_gain: float = 1.0,
        edge_left_wall_model_path: Optional[str] = None,
        edge_track_width_model_path: Optional[str] = None,
        edge_heading_model_path: Optional[str] = None,
        cloud_left_wall_model_path: Optional[str] = None,
        cloud_track_width_model_path: Optional[str] = None,
        cloud_heading_model_path: Optional[str] = None,
    ) -> None:
        self.cloud_latency = cloud_latency
        self.alpha_steer = alpha_steer
        self.alpha_speed = alpha_speed
        self.top_k = top_k
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.lateral_gain = lateral_gain
        self.wheelbase = F110_WHEELBASE

        self.edge_planner = LidarDNNPlanner(
            left_model_path=edge_left_wall_model_path,
            track_width_model_path=edge_track_width_model_path,
            heading_model_path=edge_heading_model_path,
            lookahead_distance=lookahead_distance,
            max_speed=max_speed,
            lateral_gain=lateral_gain,
        )
        # Cloud planner — individual model references used directly for
        # per-DNN inference; the LidarDNNPlanner wrapper provides _load_model.
        self._cloud_loader = LidarDNNPlanner(
            left_model_path=cloud_left_wall_model_path,
            track_width_model_path=cloud_track_width_model_path,
            heading_model_path=cloud_heading_model_path,
            lookahead_distance=lookahead_distance,
            max_speed=max_speed,
            lateral_gain=lateral_gain,
        )
        # Convenient direct references to each cloud model
        self._cloud_models = [
            self._cloud_loader.left_model,
            self._cloud_loader.track_width_model,
            self._cloud_loader.heading_model,
        ]

        # per-DNN in-flight queues: each entry is (arrival_step, scan_1d)
        self._queues: list[deque[tuple[int, np.ndarray]]] = [
            deque() for _ in range(self.NUM_DNNS)
        ]
        # per-DNN held cloud results; None = no result yet → edge fallback
        self._cloud_cache: list[Optional[float]] = [None] * self.NUM_DNNS
        # step at which each slot's cloud cache was last updated (-1 = never)
        self._cloud_last_updated: list[int] = [-1] * self.NUM_DNNS

        # -------------------------------------------------------------------
        # Public attributes readable by the env for observation building
        # -------------------------------------------------------------------
        # Latest edge DNN outputs (populated every step)
        self.last_edge_left: float = 0.0
        self.last_edge_track: float = 0.0
        self.last_edge_heading: float = 0.0
        # Resolved cloud values (cloud cache if set, else current edge output)
        self.last_cloud_left: float = 0.0
        self.last_cloud_track: float = 0.0
        self.last_cloud_heading: float = 0.0
        # Most recent blended action
        self.last_action: Optional[Action] = None
        # Which DNNs were requested this step
        self.last_call_mask: list[bool] = [False] * self.NUM_DNNS
        # Steps since each slot's cloud cache was last updated (0 = just refreshed)
        # Large value (999) before the first cloud result ever arrives.
        self.last_cloud_age: list[int] = [999] * self.NUM_DNNS
        # For render callbacks that expect last_target_point
        self.last_target_point = None

        self._step: int = 0

    # ------------------------------------------------------------------
    # BasePlanner interface
    # ------------------------------------------------------------------
    def plan(  # pylint: disable=too-many-locals
        self,
        obs: dict[str, Any],
        call_mask: Optional[list[bool]] = None,
        ego_idx: int = 0,
    ) -> Action:
        """Compute blended edge-cloud action with selective DNN calling.

        Parameters
        ----------
        obs : dict
            Current simulator observation.
        call_mask : list[bool] of length NUM_DNNS, optional
            ``True`` at index *i* requests a cloud inference for DNN *i* this
            step.  If ``None``, no cloud calls are made (edge-only mode),
            which satisfies the :class:`~f110_planning.base.BasePlanner`
            interface signature.
        ego_idx : int
            Agent index within the observation arrays.

        Returns
        -------
        Action
            Blended (steer, speed) command.
        """
        if call_mask is None:
            call_mask = [False] * self.NUM_DNNS
        self.last_call_mask = list(call_mask)

        step = self._step
        scan: np.ndarray = obs["scans"][ego_idx]

        # 1. Run edge planner — caches last_left_dist / last_track_width / last_heading_error
        edge_action = self.edge_planner.plan(obs, ego_idx)
        self.last_edge_left = self.edge_planner.last_left_dist
        self.last_edge_track = self.edge_planner.last_track_width
        self.last_edge_heading = self.edge_planner.last_heading_error
        self.last_target_point = self.edge_planner.last_target_point

        # 2. Enqueue cloud requests for selected DNNs
        scan_snapshot = scan.copy()
        for i, call in enumerate(call_mask):
            if call:
                self._queues[i].append((step + self.cloud_latency, scan_snapshot))

        # 3. Receive any arrived cloud results
        for i in range(self.NUM_DNNS):
            while self._queues[i] and self._queues[i][0][0] <= step:
                _, stale_scan = self._queues[i].popleft()
                result = self._cloud_loader.predict(self._cloud_models[i], stale_scan)
                if result is not None:
                    self._cloud_cache[i] = float(result)
                    self._cloud_last_updated[i] = step

        # Update public cloud-age attribute
        for i in range(self.NUM_DNNS):
            self.last_cloud_age[i] = (
                step - self._cloud_last_updated[i]
                if self._cloud_last_updated[i] >= 0
                else 999
            )

        # 4. Resolve each slot:
        #    • Called DNNs   — use the most recently received cloud result if
        #      one exists (latency may mean it is 1–cloud_latency steps old),
        #      otherwise fall back to the current edge output.
        #    • Uncalled DNNs — ALWAYS use the current fresh edge output.
        #      Using a held cloud value here can be arbitrarily stale (e.g.
        #      cloud_cache[HEADING] not refreshed for 50+ steps while the
        #      agent called only the left-wall DNN on a straight).  Feeding
        #      a stale heading-error into get_reactive_actuation is the
        #      primary cause of crashes at extreme bends.
        resolved_left = (
            self._cloud_cache[self.LEFT]
            if (call_mask[self.LEFT] and self._cloud_cache[self.LEFT] is not None)
            else self.last_edge_left
        )
        resolved_track = (
            self._cloud_cache[self.TRACK]
            if (call_mask[self.TRACK] and self._cloud_cache[self.TRACK] is not None)
            else self.last_edge_track
        )
        resolved_heading = (
            self._cloud_cache[self.HEADING]
            if (call_mask[self.HEADING] and self._cloud_cache[self.HEADING] is not None)
            else self.last_edge_heading
        )

        self.last_cloud_left = resolved_left
        self.last_cloud_track = resolved_track
        self.last_cloud_heading = resolved_heading

        # 5. Compute cloud action from resolved features
        car_theta: float = float(obs["poses_theta"][ego_idx])
        car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
        current_speed: float = float(obs["linear_vels_x"][ego_idx])

        resolved_right = max(resolved_track - resolved_left, 0.0)
        _, cloud_steer, cloud_speed = get_reactive_actuation(
            left_dist=resolved_left,
            right_dist=resolved_right,
            heading_error=resolved_heading,
            car_position=car_position,
            car_theta=car_theta,
            current_speed=current_speed,
            lookahead_gain=self.lookahead_distance,
            max_speed=self.max_speed,
            wheelbase=self.wheelbase,
            lateral_gain=self.lateral_gain,
        )

        # 6. Blend
        blended_steer = (
            self.alpha_steer * cloud_steer + (1.0 - self.alpha_steer) * edge_action.steer
        )
        blended_speed = (
            self.alpha_speed * cloud_speed + (1.0 - self.alpha_speed) * edge_action.speed
        )

        action = Action(steer=blended_steer, speed=blended_speed)
        self.last_action = action
        self._step += 1
        return action

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset step counter, in-flight queues, and cloud caches."""
        self._step = 0
        self._queues = [deque() for _ in range(self.NUM_DNNS)]
        self._cloud_cache = [None] * self.NUM_DNNS
        self._cloud_last_updated = [-1] * self.NUM_DNNS
        self.last_action = None
        self.last_call_mask = [False] * self.NUM_DNNS
        self.last_cloud_age = [999] * self.NUM_DNNS
        self.last_edge_left = 0.0
        self.last_edge_track = 0.0
        self.last_edge_heading = 0.0
        self.last_cloud_left = 0.0
        self.last_cloud_track = 0.0
        self.last_cloud_heading = 0.0
        self.last_target_point = None
