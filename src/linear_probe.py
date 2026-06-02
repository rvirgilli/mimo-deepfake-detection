"""
Linear Probe classifier for evaluating frozen feature quality.

This is the simplest possible classifier to test if features are
inherently linearly separable (good features) or not (bad features).

Usage:
    frontend = get_frontend("mimo", ...)
    model = LinearProbeModel(frontend)
"""

import torch
import torch.nn as nn
from torch import Tensor

from .frontends.base import BaseFrontend


class LinearProbeModel(nn.Module):
    """
    Linear probe classifier for frozen feature evaluation.

    This model applies global average pooling over time, then a single
    linear layer for classification. No hidden layers, no attention,
    no complex processing - just the bare minimum to test feature quality.

    Args:
        frontend: Frontend feature extractor (must be frozen for proper evaluation)
        num_classes: Number of output classes (default: 2 for real/fake)
        pool_type: Pooling strategy ('mean', 'max', or 'both')
        dropout: Dropout rate before linear layer
    """

    def __init__(
        self,
        frontend: BaseFrontend,
        num_classes: int = 2,
        pool_type: str = "mean",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.frontend = frontend
        self.pool_type = pool_type

        # Determine input dimension based on pooling
        if pool_type == "both":
            in_dim = frontend.out_dim * 2
        else:
            in_dim = frontend.out_dim

        # Simple dropout + linear
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)

        # Store config for logging
        self.arch_params = {
            "pool_type": pool_type,
            "dropout": dropout,
            "in_dim": in_dim,
            "num_classes": num_classes,
            "frontend_dim": frontend.out_dim,
        }

        # Verify frontend is frozen (warning if not)
        trainable = sum(p.numel() for p in frontend.parameters() if p.requires_grad)
        if trainable > 0:
            import warnings
            warnings.warn(
                f"Frontend has {trainable:,} trainable parameters. "
                "For proper linear probe evaluation, frontend should be frozen."
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Raw audio waveform (batch, samples) or (batch, samples, 1)

        Returns:
            logits: (batch, num_classes)
        """
        # Extract features: (batch, time, dim)
        x = self.frontend.extract_feat(x.squeeze(-1))

        # Global pooling over time
        if self.pool_type == "mean":
            x = x.mean(dim=1)  # (batch, dim)
        elif self.pool_type == "max":
            x = x.max(dim=1).values  # (batch, dim)
        elif self.pool_type == "both":
            x_mean = x.mean(dim=1)
            x_max = x.max(dim=1).values
            x = torch.cat([x_mean, x_max], dim=1)  # (batch, dim*2)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Dropout + linear
        x = self.dropout(x)
        x = self.fc(x)

        return x


class MLPProbeModel(nn.Module):
    """
    MLP probe classifier - slightly more complex than linear probe.

    Useful for comparison: if MLP does much better than Linear,
    features have useful structure but aren't linearly separable.

    Args:
        frontend: Frontend feature extractor
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        pool_type: Pooling strategy
        dropout: Dropout rate
    """

    def __init__(
        self,
        frontend: BaseFrontend,
        hidden_dim: int = 256,
        num_classes: int = 2,
        pool_type: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.frontend = frontend
        self.pool_type = pool_type

        # Determine input dimension based on pooling
        if pool_type == "both":
            in_dim = frontend.out_dim * 2
        else:
            in_dim = frontend.out_dim

        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.arch_params = {
            "pool_type": pool_type,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "in_dim": in_dim,
            "num_classes": num_classes,
        }

    def forward(self, x: Tensor) -> Tensor:
        # Extract features
        x = self.frontend.extract_feat(x.squeeze(-1))

        # Global pooling
        if self.pool_type == "mean":
            x = x.mean(dim=1)
        elif self.pool_type == "max":
            x = x.max(dim=1).values
        elif self.pool_type == "both":
            x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)

        return self.mlp(x)
