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

The (left, track, heading) triple is blended feature-by-feature using
MSE-derived per-feature weights before being passed into the reactive
controller once.  Weights are age-dependent when process-noise parameters
are supplied.
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
    alpha_left : float
        Static blending weight for the left-wall feature (0 = edge, 1 = cloud).
        Default is the closed-form MSE optimum σ²_e / (σ²_e + σ²_c).
    alpha_track : float
        Static blending weight for the track-width feature.
    alpha_heading : float
        Static blending weight for the heading-error feature.
    sigma_proc_left, sigma_proc_track, sigma_proc_heading : float | None
        Per-feature process-noise standard deviations (σ_proc, not σ²).
        When provided the blending weight ages as
        ``α_i(t) = σ²_e / (σ²_e + σ²_c + σ²_proc × age)``.
        ``None`` (default) keeps the weight fixed at the supplied static value.
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

    # Empirical MSE constants from DNN evaluation (used for age-dependent alpha computation).
    # Order: [LEFT, TRACK, HEADING]
    # Edge: arch1 (left_wall_dist), arch2 (track_width), arch2 (heading_error)
    # Cloud: arch5 (left_wall_dist), arch7 (track_width), arch6 (heading_error)
    _SIGMA2_EDGE: tuple[float, ...] = (0.028020, 0.036518, 0.019371)
    _SIGMA2_CLOUD: tuple[float, ...] = (0.000518, 0.001539, 0.001140)

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        cloud_latency: int = 10,
        alpha_left: float = 0.996,
        alpha_track: float = 0.988,
        alpha_heading: float = 0.974,
        sigma_proc_left: Optional[float] = None,
        sigma_proc_track: Optional[float] = None,
        sigma_proc_heading: Optional[float] = None,
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
        self._static_alphas: list[float] = [alpha_left, alpha_track, alpha_heading]
        self._sigma_proc: list[Optional[float]] = [
            sigma_proc_left, sigma_proc_track, sigma_proc_heading
        ]
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

        # per-DNN in-flight queues:
        # each entry is (arrival_step, scan_1d, edge_left, edge_track, edge_heading)
        self._queues: list[deque[tuple[int, np.ndarray, float, float, float]]] = [
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
        self.edge_planner.plan(obs, ego_idx)
        self.last_edge_left = self.edge_planner.last_left_dist
        self.last_edge_track = self.edge_planner.last_track_width
        self.last_edge_heading = self.edge_planner.last_heading_error
        self.last_target_point = self.edge_planner.last_target_point

        # 2. Enqueue cloud requests for selected DNNs (snapshot edge outputs for
        #    edge-delta latency correction on arrival)
        scan_snapshot = scan.copy()
        for i, call in enumerate(call_mask):
            if call:
                self._queues[i].append((
                    step + self.cloud_latency,
                    scan_snapshot,
                    self.last_edge_left,
                    self.last_edge_track,
                    self.last_edge_heading,
                ))

        # 3. Receive any arrived cloud results; apply edge-delta correction
        _edge_now = (self.last_edge_left, self.last_edge_track, self.last_edge_heading)
        for i in range(self.NUM_DNNS):
            while self._queues[i] and self._queues[i][0][0] <= step:
                _, stale_scan, enq_left, enq_track, enq_heading = self._queues[i].popleft()
                result = self._cloud_loader.predict(self._cloud_models[i], stale_scan)
                if result is not None:
                    enq_edge = (enq_left, enq_track, enq_heading)[i]
                    self._cloud_cache[i] = float(result) + (_edge_now[i] - enq_edge)
                    self._cloud_last_updated[i] = step

        # Update public cloud-age attribute
        for i in range(self.NUM_DNNS):
            self.last_cloud_age[i] = (
                step - self._cloud_last_updated[i]
                if self._cloud_last_updated[i] >= 0
                else 999
            )

        # 4. Feature-level blend: for each slot compute an age-dependent (or
        #    static) alpha and blend the raw feature values.  Cache always
        #    contributes regardless of call_mask; the call_mask only controls
        #    which cloud DNNs receive new requests this step.
        edge_vals = (self.last_edge_left, self.last_edge_track, self.last_edge_heading)
        blended: list[float] = []
        for i in range(self.NUM_DNNS):
            if self._cloud_cache[i] is None:
                # No cloud result has ever arrived for this slot → use pure edge.
                blended.append(edge_vals[i])
            else:
                age = self.last_cloud_age[i]
                sp = self._sigma_proc[i]
                if sp is not None:
                    a = self._alpha_age(
                        age,
                        self._SIGMA2_EDGE[i],
                        self._SIGMA2_CLOUD[i],
                        sp * sp,
                    )
                else:
                    a = self._static_alphas[i]
                blended.append(a * self._cloud_cache[i] + (1.0 - a) * edge_vals[i])

        blended_left, blended_track, blended_heading = blended
        self.last_cloud_left = blended_left
        self.last_cloud_track = blended_track
        self.last_cloud_heading = blended_heading

        # 5. Compute action from blended feature triple (single controller call).
        car_theta: float = float(obs["poses_theta"][ego_idx])
        car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
        current_speed: float = float(obs["linear_vels_x"][ego_idx])

        blended_right = max(blended_track - blended_left, 0.0)
        _, blended_steer, blended_speed = get_reactive_actuation(
            left_dist=blended_left,
            right_dist=blended_right,
            heading_error=blended_heading,
            car_position=car_position,
            car_theta=car_theta,
            current_speed=current_speed,
            lookahead_gain=self.lookahead_distance,
            max_speed=self.max_speed,
            wheelbase=self.wheelbase,
            lateral_gain=self.lateral_gain,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _alpha_age(age: int, sigma_e2: float, sigma_c2: float, sigma_proc2: float) -> float:
        """Compute the age-dependent blending weight.

        ``α = σ²_e / (σ²_e + σ²_c + σ²_proc × age)``

        At age=0 this reduces to the closed-form static MSE optimum.  As age
        grows the cloud contribution is down-weighted by accumulated process
        noise.  Returns 0.5 if the denominator is zero (degenerate case).
        """
        denom = sigma_e2 + sigma_c2 + sigma_proc2 * age
        return sigma_e2 / denom if denom > 0.0 else 0.5
