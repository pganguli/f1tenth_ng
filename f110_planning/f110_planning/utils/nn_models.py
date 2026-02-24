"""
Shared DNN architectures for F1TENTH.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class DualHeadWallModel(nn.Module):
    """
    Predicts both Left and Right wall distances from a single LiDAR scan.

    This model share a common feature-extraction backbone but uses independent
    fully-connected heads for the two wall distance predictions.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone

        def _make_head() -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Linear(feature_dim, hidden_dim),
                nn.ELU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.left_head = _make_head()
        self.right_head = _make_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input scan and returns [left_dist, right_dist].
        """
        features = self.backbone(x)
        left = self.left_head(features)
        right = self.right_head(features)
        return torch.cat([left, right], dim=1)


class BayesianLinear(nn.Module):
    """
    Variational Bayesian linear layer with diagonal Gaussian posterior.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 0.1,
        init_rho: float = -3.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.init_rho = init_rho

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, self.init_rho)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, self.init_rho)

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)

    @staticmethod
    def _sample(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(rho)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self._sample(self.weight_mu, self.weight_rho)
            bias = self._sample(self.bias_mu, self.bias_rho)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_loss(self) -> torch.Tensor:
        """
        KL[q(w|theta)||p(w)] for all weights and biases.
        """

        prior_var = self.prior_sigma**2

        def _kl(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
            var = self._sigma(rho).pow(2)
            term = (var + mu.pow(2)) / prior_var - 1.0 + torch.log(prior_var / var)
            return 0.5 * torch.sum(term)

        return _kl(self.weight_mu, self.weight_rho) + _kl(self.bias_mu, self.bias_rho)


class BayesianHeadingModel(nn.Module):
    """
    Heading predictor with deterministic CNN backbone and Bayesian FC head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        hidden_dim: int,
        prior_sigma: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        layers: list[nn.Module] = [
            BayesianLinear(feature_dim, hidden_dim, prior_sigma=prior_sigma),
            nn.ELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(BayesianLinear(hidden_dim, 1, prior_sigma=prior_sigma))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def kl_loss(self) -> torch.Tensor:
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_loss()
        return kl


class BayesianDualHeadWallModel(nn.Module):
    """
    Dual-head wall-distance predictor with Bayesian fully connected heads.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        hidden_dim: int,
        prior_sigma: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        def _make_head() -> nn.Sequential:
            layers: list[nn.Module] = [
                BayesianLinear(feature_dim, hidden_dim, prior_sigma=prior_sigma),
                nn.ELU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(BayesianLinear(hidden_dim, 1, prior_sigma=prior_sigma))
            return nn.Sequential(*layers)

        self.left_head = _make_head()
        self.right_head = _make_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        left = self.left_head(features)
        right = self.right_head(features)
        return torch.cat([left, right], dim=1)

    def kl_loss(self) -> torch.Tensor:
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_loss()
        return kl


def get_architecture(arch_id: int, task: str = "heading") -> nn.Module:
    """
    Factory function for standard F1TENTH neural network architectures.

    Args:
        arch_id: An index (1-13) identifying the specific layer configuration.
            Legacy heading architectures: 1-7.
            Multi-size architectures (heading & wall): 8 (small), 9 (medium), 10 (large).
            Bayesian architectures (heading & wall): 11 (small), 12 (medium), 13 (large).
        task: Either 'heading' (1 output) or 'wall' (2 outputs via DualHeadWallModel).

    Returns:
        An initialized PyTorch nn.Module.
    """

    def get_backbone(backbone_id: int) -> tuple[nn.Module, int]:
        # Base families of feature extractors
        families = {
            0: (  # Tiny: ~140 conv params, dim=528
                nn.Sequential(
                    nn.Conv1d(1, 4, kernel_size=3),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=4),
                    nn.Conv1d(4, 8, kernel_size=3),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=4),
                    nn.Flatten(),
                ),
                528,
            ),
            1: (  # Medium: ~440 conv params, dim=2144
                nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=3),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(8, 16, kernel_size=3),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=4),
                    nn.Flatten(),
                ),
                2144,
            ),
            2: (  # Large: ~1.5K conv params, dim=4288
                nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=3),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(16, 32, kernel_size=3),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=4),
                    nn.Flatten(),
                ),
                4288,
            ),
            3: (  # Deep: 3-layer CNN, ~20K conv params, dim=8512
                nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=3),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(16, 32, kernel_size=3),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(32, 64, kernel_size=3),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Flatten(),
                ),
                8512,
            ),
        }
        return families.get(backbone_id, families[1])

    def _insert_batchnorm(seq: nn.Sequential) -> nn.Sequential:
        """Insert BatchNorm1d after every Conv1d in a flat Sequential."""
        new_layers: list[nn.Module] = []
        for layer in seq:
            new_layers.append(layer)
            if isinstance(layer, nn.Conv1d):
                new_layers.append(nn.BatchNorm1d(layer.out_channels))
        return nn.Sequential(*new_layers)

    # --- Wall task: DualHeadWallModel with configurable backbone + heads ---
    if task == "wall":
        # Multi-size wall configs: (backbone_id, head_hidden, use_batchnorm, dropout)
        wall_configs: dict[int, tuple[int, int, bool, float]] = {
            8: (0, 16, False, 0.0),  # Small: tiny backbone, no BN, no dropout
            9: (1, 32, True, 0.0),   # Medium: BN in backbone
            10: (3, 64, True, 0.3),  # Large: BN in backbone + dropout in heads
        }
        if arch_id in wall_configs:
            bb_id, hidden, use_bn, dropout = wall_configs[arch_id]
            backbone, dim = get_backbone(bb_id)
            if use_bn:
                backbone = _insert_batchnorm(backbone)
            return DualHeadWallModel(backbone, dim, hidden_dim=hidden, dropout=dropout)

        bayes_wall_configs: dict[int, tuple[int, int, bool, float, float]] = {
            11: (0, 16, False, 0.0, 0.1),  # Small Bayesian wall model
            12: (1, 32, True, 0.0, 0.1),   # Medium Bayesian wall model
            13: (3, 64, True, 0.3, 0.1),   # Large Bayesian wall model
        }
        if arch_id in bayes_wall_configs:
            bb_id, hidden, use_bn, dropout, prior_sigma = bayes_wall_configs[arch_id]
            backbone, dim = get_backbone(bb_id)
            if use_bn:
                backbone = _insert_batchnorm(backbone)
            return BayesianDualHeadWallModel(
                backbone,
                dim,
                hidden_dim=hidden,
                prior_sigma=prior_sigma,
                dropout=dropout,
            )
        # Legacy wall dispatch (arch_ids 1-7)
        backbone_id = 1 if arch_id <= 4 else 2
        backbone, dim = get_backbone(backbone_id)
        return DualHeadWallModel(backbone, dim)

    # Legacy / Heading architectures (arch_id 1-7)
    factories = {
        1: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(134, 1),
        ),
        2: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 1),
        ),
        3: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(1072, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        ),
        4: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(2144, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        ),
        5: lambda: nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(2144, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        ),
        6: lambda: nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 32, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(4288, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        ),
        7: lambda: nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(4288, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ),
        # --- Multi-size heading architectures ---
        8: lambda: nn.Sequential(  # Small: ~8.6K params
            nn.Conv1d(1, 4, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(4, 8, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(528, 16),
            nn.ELU(),
            nn.Linear(16, 1),
        ),
        9: lambda: nn.Sequential(  # Medium: ~69K params, with BatchNorm
            nn.Conv1d(1, 8, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(2144, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        ),
        10: lambda: nn.Sequential(  # Large: ~1.1M params, with BatchNorm + Dropout
            nn.Conv1d(1, 16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(8512, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        ),
    }

    bayes_heading_configs: dict[int, tuple[int, int, bool, float, float]] = {
        11: (0, 16, False, 0.0, 0.1),  # Small Bayesian heading model
        12: (1, 32, True, 0.0, 0.1),   # Medium Bayesian heading model
        13: (3, 128, True, 0.3, 0.1),  # Large Bayesian heading model
    }
    if arch_id in bayes_heading_configs:
        bb_id, hidden, use_bn, dropout, prior_sigma = bayes_heading_configs[arch_id]
        backbone, dim = get_backbone(bb_id)
        if use_bn:
            backbone = _insert_batchnorm(backbone)
        return BayesianHeadingModel(
            backbone,
            dim,
            hidden_dim=hidden,
            prior_sigma=prior_sigma,
            dropout=dropout,
        )

    if arch_id not in factories:
        raise ValueError(f"Architecture ID {arch_id} not supported for {task}.")

    return factories[arch_id]()
