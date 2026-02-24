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
        max_speed: float = 6.5,
        lateral_gain: float = 1.0,
        mc_samples: int = 20,
        bayes_inference_mode: str = "deterministic",
        uncertainty_speed_gain: float = 0.6,
        uncertainty_lookahead_gain: float = 0.8,
        min_speed_scale: float = 0.5,
        max_lookahead_scale: float = 1.8,
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
            mc_samples: Number of Monte Carlo forward passes for Bayesian models.
            bayes_inference_mode: "deterministic" (mu weights only) or "mc".
            uncertainty_speed_gain: How strongly uncertainty reduces max speed.
            uncertainty_lookahead_gain: How strongly uncertainty increases lookahead.
            min_speed_scale: Lower bound on uncertainty-scaled speed factor.
            max_lookahead_scale: Upper bound on uncertainty-scaled lookahead factor.
        """
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.lateral_gain = lateral_gain
        self.base_lookahead_distance = lookahead_distance
        self.base_max_speed = max_speed
        self.wheelbase = F110_WHEELBASE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mc_samples = max(1, int(mc_samples))
        self.bayes_inference_mode = (
            bayes_inference_mode if bayes_inference_mode in {"deterministic", "mc"} else "deterministic"
        )
        self.uncertainty_speed_gain = max(0.0, float(uncertainty_speed_gain))
        self.uncertainty_lookahead_gain = max(0.0, float(uncertainty_lookahead_gain))
        self.min_speed_scale = float(np.clip(min_speed_scale, 0.1, 1.0))
        self.max_lookahead_scale = max(1.0, float(max_lookahead_scale))

        self.last_target_point = None
        self.last_left_std = 0.0
        self.last_right_std = 0.0
        self.last_heading_std = 0.0
        self.last_uncertainty = 0.0
        self.last_speed_scale = 1.0
        self.last_lookahead_scale = 1.0

        self.wall_model = self._load_model(wall_model_path, arch_id, task="wall")
        self.left_model = self._load_model(left_model_path, arch_id, task="heading")
        self.right_model = self._load_model(right_model_path, arch_id, task="heading")

        h_arch = heading_arch_id if heading_arch_id is not None else arch_id
        self.heading_model = self._load_model(
            heading_model_path, h_arch, task="heading"
        )
        self.wall_is_bayesian = (
            self._is_bayesian_model(self.wall_model)
            if self.wall_model is not None
            else False
        )
        self.left_is_bayesian = (
            self._is_bayesian_model(self.left_model)
            if self.left_model is not None
            else False
        )
        self.right_is_bayesian = (
            self._is_bayesian_model(self.right_model)
            if self.right_model is not None
            else False
        )
        self.heading_is_bayesian = (
            self._is_bayesian_model(self.heading_model)
            if self.heading_model is not None
            else False
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

    def _is_bayesian_model(self, model: torch.nn.Module) -> bool:
        """
        Checks whether the model contains BayesianLinear layers.
        """
        from ..utils.nn_models import BayesianLinear  # pylint: disable=import-outside-toplevel

        return any(isinstance(m, BayesianLinear) for m in model.modules())

    def _enable_bayesian_sampling(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """
        Enables train mode for Bayesian layers only.
        """
        from ..utils.nn_models import BayesianLinear  # pylint: disable=import-outside-toplevel

        toggled: list[torch.nn.Module] = []
        for module in model.modules():
            if isinstance(module, BayesianLinear) and not module.training:
                module.train()
                toggled.append(module)
        return toggled

    @staticmethod
    def _restore_eval(modules: list[torch.nn.Module]) -> None:
        """
        Restores eval mode for modules that were temporarily toggled.
        """
        for module in modules:
            module.eval()

    def predict(self, model: Optional[torch.nn.Module], scan: np.ndarray) -> Any:
        """
        Returns predictive mean and uncertainty-aware standard deviation.
        """
        if model is None:
            return None

        with torch.no_grad():
            x = torch.from_numpy(scan).float().unsqueeze(0).unsqueeze(0).to(self.device)
            # Normalize to 0-1 range based on training assumptions
            x = torch.clip(x / 10.0, 0, 1)

            if self._is_bayesian_model(model) and self.bayes_inference_mode == "mc":
                toggled = self._enable_bayesian_sampling(model)
                try:
                    preds = [model(x) for _ in range(self.mc_samples)]
                finally:
                    self._restore_eval(toggled)

                pred_stack = torch.stack(preds, dim=0)  # [mc, batch=1, out]
                mean = pred_stack.mean(dim=0).squeeze(0)  # [out]
                std = pred_stack.std(dim=0, unbiased=False).squeeze(0)  # [out]
            else:
                out = model(x).squeeze(0)
                mean = out
                std = torch.zeros_like(mean)

            if mean.ndim > 0 and mean.numel() > 1:
                return mean.cpu().numpy().flatten(), std.cpu().numpy().flatten()
            return mean.item(), std.item()

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:  # pylint: disable=too-many-locals
        scan = obs["scans"][ego_idx]
        car_theta = obs["poses_theta"][ego_idx]
        car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
        current_speed = obs["linear_vels_x"][ego_idx]

        # 1. Predict geometric features using DNNs
        if self.wall_model is not None:
            wall_pred = self.predict(self.wall_model, scan)
            wall_dists = wall_pred[0] if wall_pred is not None else None
            wall_std = wall_pred[1] if wall_pred is not None else None
            if wall_dists is not None and len(wall_dists) >= 2:
                left_dist, right_dist = wall_dists[0], wall_dists[1]
                left_std = wall_std[0] if wall_std is not None else 0.0
                right_std = wall_std[1] if wall_std is not None else 0.0
            else:
                left_dist, right_dist = 0.0, 0.0
                left_std, right_std = 0.0, 0.0
        else:
            left_pred = self.predict(self.left_model, scan)
            right_pred = self.predict(self.right_model, scan)
            if left_pred is None:
                left_dist, left_std = 0.0, 0.0
            else:
                left_dist, left_std = left_pred
            if right_pred is None:
                right_dist, right_std = 0.0, 0.0
            else:
                right_dist, right_std = right_pred

        heading_pred = self.predict(self.heading_model, scan)
        if heading_pred is None:
            heading_error, heading_std = 0.0, 0.0
        else:
            heading_error, heading_std = heading_pred

        # 2. Conservative adaptation under high predictive uncertainty
        std_components: list[float] = []
        if self.wall_model is not None and self.wall_is_bayesian:
            std_components.extend([abs(float(left_std)), abs(float(right_std))])
        else:
            if self.left_is_bayesian:
                std_components.append(abs(float(left_std)))
            if self.right_is_bayesian:
                std_components.append(abs(float(right_std)))
        if self.heading_is_bayesian:
            std_components.append(abs(float(heading_std)))

        uncertainty = float(np.mean(std_components)) if std_components else 0.0
        speed_scale = float(
            np.clip(
                1.0 - self.uncertainty_speed_gain * uncertainty,
                self.min_speed_scale,
                1.0,
            )
        )
        lookahead_scale = float(
            np.clip(
                1.0 + self.uncertainty_lookahead_gain * uncertainty,
                1.0,
                self.max_lookahead_scale,
            )
        )

        self.last_left_std = float(left_std)
        self.last_right_std = float(right_std)
        self.last_heading_std = float(heading_std)
        self.last_uncertainty = uncertainty
        self.last_speed_scale = speed_scale
        self.last_lookahead_scale = lookahead_scale

        self.max_speed = self.base_max_speed * speed_scale
        self.lookahead_distance = self.base_lookahead_distance * lookahead_scale

        # pylint: disable=duplicate-code
        # Compute dynamic waypoint and actuation using shared logic helper
        try:
            return get_reactive_action(
                self,
                left_dist=left_dist,
                right_dist=right_dist,
                heading_error=heading_error,
                car_position=car_position,
                car_theta=car_theta,
                current_speed=current_speed,
            )
        finally:
            self.max_speed = self.base_max_speed
            self.lookahead_distance = self.base_lookahead_distance
