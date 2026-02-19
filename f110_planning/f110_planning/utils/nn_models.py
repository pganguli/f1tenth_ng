"""
Shared DNN architectures for F1TENTH.
"""

import torch
from torch import nn


class DualHeadWallModel(nn.Module):
    """
    Predicts both Left and Right wall distances from a single LiDAR scan.

    This model share a common feature-extraction backbone but uses independent
    fully-connected heads for the two wall distance predictions.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        self.left_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.right_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input scan and returns [left_dist, right_dist].
        """
        features = self.backbone(x)
        left = self.left_head(features)
        right = self.right_head(features)
        return torch.cat([left, right], dim=1)


def get_architecture(arch_id: int, task: str = "heading") -> nn.Module:
    """
    Factory function for standard F1TENTH neural network architectures.

    Args:
        arch_id: An index (1-7) identifying the specific layer configuration.
        task: Either 'heading' (1 output) or 'wall' (2 outputs via DualHeadWallModel).

    Returns:
        An initialized PyTorch nn.Module.
    """

    def get_backbone(backbone_id: int) -> tuple[nn.Module, int]:
        # Base families of feature extractors
        families = {
            1: (
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
            2: (
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
        }
        return families.get(backbone_id, families[1])

    if task == "wall":
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
    }

    if arch_id not in factories:
        raise ValueError(f"Architecture ID {arch_id} not supported for {task}.")

    return factories[arch_id]()
