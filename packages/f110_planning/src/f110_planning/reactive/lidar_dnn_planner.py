"""
PyTorch-based DNN planner for F1TENTH.
Uses trained models to predict wall distances and heading errors for navigation.
"""

import io
import json
from typing import Any, Optional

import numpy as np
import torch

from ..base import Action, BasePlanner
from ..utils import F110_WHEELBASE, get_reactive_action


class LidarDNNPlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """
    Reactive planner that uses PyTorch models to predict control features from LiDAR.

    This planner mimics the behavior of the DynamicWaypointPlanner, but instead
    of using geometric calculations on the map/scan, it uses neural networks
    to predict wall distances and orientation errors directly from raw sensor data.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        left_model_path: Optional[str] = None,
        track_width_model_path: Optional[str] = None,
        heading_model_path: Optional[str] = None,
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        lateral_gain: float = 1.0,
    ) -> None:
        """
        Initializes the DNN planner and loads the specified models.

        Each model path should point to a self-sufficient ``.pt`` file produced
        by :func:`~f110_planning.utils.nn_models.save_as_torchscript`.  The
        architecture and all hyperparameters are read directly from the file —
        no ``arch_id`` or external config is required.

        Three separate single-output models are used at inference time:

        * ``left_model_path``         — predicts left wall distance.
        * ``track_width_model_path``  — predicts total track width
          (left + right wall distance).  Right wall distance is derived
          as ``track_width - left_dist``.
        * ``heading_model_path``      — predicts path heading error.

        Args:
            left_model_path: Path to ``.pt`` model for left wall distance.
            track_width_model_path: Path to ``.pt`` model for track width.
            heading_model_path: Path to ``.pt`` model for path heading error.
            lookahead_distance: Gain for the adaptive lookahead calculation.
            max_speed: Velocity limit on straight sections.
            lateral_gain: Scaling for the lateral centering response.
        """
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.lateral_gain = lateral_gain
        self.wheelbase = F110_WHEELBASE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.last_target_point = None

        # cached outputs from the most recent plan() call; readable by external
        # components (e.g. SelectiveEdgeCloudPlanner) without re-running the models
        self.last_left_dist: float = 0.0
        self.last_track_width: float = 0.0
        self.last_heading_error: float = 0.0

        self.left_model = self._load_model(left_model_path)
        self.track_width_model = self._load_model(track_width_model_path)
        self.heading_model = self._load_model(heading_model_path)

    def _load_model(
        self,
        path: Optional[str],
    ) -> Optional[torch.nn.Module]:
        """Load a self-sufficient ``.pt`` TorchScript model file.

        The file must have been produced by
        :func:`~f110_planning.utils.nn_models.save_as_torchscript`.  All
        architecture and configuration information is read from the embedded
        metadata — no ``arch_id`` argument is needed.

        For non-quantized models the returned :class:`torch.ScriptModule` is
        used directly (no Python class definitions required at inference time).

        For quantized models the embedded FP32 ``state_dict`` is loaded into a
        freshly instantiated :func:`~f110_planning.utils.nn_models.get_architecture`
        module, after which ``quantize_()`` is re-applied.  This faithfully
        reproduces the INT8 dynamic-activation / INT8-weight quantisation used
        during training.
        """
        if not path:
            return None

        extra: dict[str, bytes] = {"metadata.json": b"", "state_dict.pt": b""}
        scripted = torch.jit.load(path, _extra_files=extra, map_location=self.device)
        metadata: dict[str, Any] = json.loads(extra["metadata.json"].decode())

        if metadata.get("quantized", False):
            # Re-instantiate the FP32 module, load embedded FP32 weights, then
            # re-apply quantization so the in-memory representation matches what
            # was used at training time.
            from ..utils.nn_models import (  # pylint: disable=import-outside-toplevel
                get_architecture,
            )
            from torchao.quantization import (  # pylint: disable=import-outside-toplevel
                Int8DynamicActivationInt8WeightConfig,
                quantize_,
            )

            model = get_architecture(metadata["arch_id"])
            model.eval()
            quantize_(model, Int8DynamicActivationInt8WeightConfig())
            buf = io.BytesIO(extra["state_dict.pt"])
            state_dict = torch.load(buf, map_location=self.device, weights_only=False)
            model.load_state_dict(state_dict)
            model.to(self.device)
            return model

        scripted.eval()
        return scripted

    def predict(self, model: Optional[torch.nn.Module], scan: np.ndarray) -> Any:
        """
        Performs a forward pass through the provided model using normalized scan data.

        Returns a scalar float, or ``None`` if *model* is ``None``.
        """
        if model is None:
            return None
        with torch.no_grad():
            x = torch.from_numpy(scan).float().unsqueeze(0).unsqueeze(0).to(self.device)
            # Normalize to 0-1 range based on training assumptions
            x = torch.clip(x / 10.0, 0, 1)
            out = model(x)
            return out.item()

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:  # pylint: disable=too-many-locals
        scan = obs["scans"][ego_idx]
        car_theta = obs["poses_theta"][ego_idx]
        car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
        current_speed = obs["linear_vels_x"][ego_idx]

        # 1. Predict geometric features using DNNs
        left_dist = self.predict(self.left_model, scan) or 0.0
        track_width = self.predict(self.track_width_model, scan) or 0.0
        right_dist = max(track_width - left_dist, 0.0)

        heading_error = self.predict(self.heading_model, scan) or 0.0

        # cache for external readers (e.g. SelectiveEdgeCloudPlanner)
        self.last_left_dist = left_dist
        self.last_track_width = track_width
        self.last_heading_error = heading_error

        # pylint: disable=duplicate-code
        # Compute dynamic waypoint and actuation using shared logic helper
        return get_reactive_action(
            self,
            left_dist=left_dist,
            right_dist=right_dist,
            heading_error=heading_error,
            car_position=car_position,
            car_theta=car_theta,
            current_speed=current_speed,
        )
