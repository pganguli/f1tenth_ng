"""
PyTorch-based DNN planner for F1TENTH.
Uses trained models to predict wall distances and heading errors for navigation.
"""

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
        right_model_path: Optional[str] = None,
        heading_model_path: Optional[str] = None,
        wall_model_path: Optional[str] = None,
        arch_id: int = 5,
        heading_arch_id: Optional[int] = None,
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        lateral_gain: float = 1.0,
    ) -> None:
        """
        Initializes the DNN planner and loads the specified models.

        Args:
            left_model_path: Path to separate model for left wall distance.
            right_model_path: Path to separate model for right wall distance.
            heading_model_path: Path to model for path heading error.
            wall_model_path: Path to dual-head model for both wall distances.
            arch_id: Architecture index for the backbone and wall heads.
            heading_arch_id: Architecture index specifically for the heading model.
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

        self.wall_model = self._load_model(wall_model_path, arch_id, task="wall")
        self.left_model = self._load_model(left_model_path, arch_id, task="heading")
        self.right_model = self._load_model(right_model_path, arch_id, task="heading")

        h_arch = heading_arch_id if heading_arch_id is not None else arch_id
        self.heading_model = self._load_model(
            heading_model_path, h_arch, task="heading"
        )

    def _load_model(
        self,
        path: Optional[str],
        arch_id: int,
        task: str = "heading",
    ) -> Optional[torch.nn.Module]:
        """
        Internal helper to instantiate and load weights for a single model.

        Supports standard state_dict files and torchao-quantized state_dict files.
        Automatically detects whether the checkpoint was saved with INT8
        quantization and prepares the architecture accordingly.
        """
        if not path:
            return None
        from ..utils.nn_models import (  # pylint: disable=import-outside-toplevel
            get_architecture,
        )

        model = get_architecture(arch_id, task=task)
        state_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Auto-detect torchao INT8-quantized checkpoints
        is_quantized = any(
            "AffineQuantizedTensor" in type(v).__name__
            or "LinearActivationQuantizedTensor" in type(v).__name__
            for v in state_dict.values()
        )
        if is_quantized:
            model.eval()
            from torchao.quantization import (  # pylint: disable=import-outside-toplevel
                Int8DynamicActivationInt8WeightConfig,
                quantize_,
            )

            quantize_(model, Int8DynamicActivationInt8WeightConfig())

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, model: Optional[torch.nn.Module], scan: np.ndarray) -> Any:
        """
        Performs a forward pass through the provided model using normalized scan data.
        """
        if model is None:
            return None
        with torch.no_grad():
            x = torch.from_numpy(scan).float().unsqueeze(0).unsqueeze(0).to(self.device)
            # Normalize to 0-1 range based on training assumptions
            x = torch.clip(x / 10.0, 0, 1)
            out = model(x)
            if out.shape[1] > 1:
                return out.cpu().numpy().flatten()
            return out.item()

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:  # pylint: disable=too-many-locals
        scan = obs["scans"][ego_idx]
        car_theta = obs["poses_theta"][ego_idx]
        car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
        current_speed = obs["linear_vels_x"][ego_idx]

        # 1. Predict geometric features using DNNs
        if self.wall_model is not None:
            wall_dists = self.predict(self.wall_model, scan)
            if wall_dists is not None and len(wall_dists) >= 2:
                left_dist, right_dist = wall_dists[0], wall_dists[1]
            else:
                left_dist, right_dist = 0.0, 0.0
        else:
            left_dist = self.predict(self.left_model, scan) or 0.0
            right_dist = self.predict(self.right_model, scan) or 0.0

        heading_error = self.predict(self.heading_model, scan) or 0.0

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
