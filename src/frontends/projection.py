"""
Projection modules for mapping frontend features to AASIST input dimension.

The projection layer is critical for MiMo because:
- MiMo has 454 discriminative dimensions (Cohen's d > 0.5) out of 1280
- A single linear layer (1280 → 128) may lose discriminative information
- An MLP can better preserve and combine discriminative features

Usage:
    # Linear projection (baseline)
    proj = LinearProjection(in_dim=1280, out_dim=128)

    # MLP projection (recommended for MiMo)
    proj = MLPProjection(
        in_dim=1280,
        out_dim=128,
        hidden_dims=[512, 256],
        activation='gelu',
        dropout=0.1
    )
"""

from typing import List, Optional

import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    """
    Simple linear projection (original AASIST behavior).

    Maps frontend features directly to AASIST input dimension.
    This is the baseline - works well for wav2vec2 but may lose
    information for MiMo's higher-dimensional features.

    Args:
        in_dim: Input dimension (e.g., 1024 for wav2vec2, 1280 for MiMo)
        out_dim: Output dimension (default: 128 for AASIST)
    """

    def __init__(self, in_dim: int = 1280, out_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_dim)
        Returns:
            (batch, seq_len, out_dim)
        """
        return self.linear(x)


class MLPProjection(nn.Module):
    """
    Multi-layer perceptron projection for better feature preservation.

    Designed to better preserve discriminative information when projecting
    from high-dimensional frontend features (e.g., MiMo's 1280-dim with
    454 discriminative dimensions) to AASIST's 128-dim input.

    Args:
        in_dim: Input dimension (e.g., 1280 for MiMo)
        out_dim: Output dimension (default: 128 for AASIST)
        hidden_dims: List of hidden layer dimensions (default: [512, 256])
        activation: Activation function ('gelu', 'relu', 'selu', 'swish')
        dropout: Dropout rate (default: 0.1)
        use_batchnorm: Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        in_dim: int = 1280,
        out_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'gelu',
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims

        # Build MLP layers
        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final projection layer (no activation/dropout)
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)
        self.use_batchnorm = use_batchnorm

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_dim)
        Returns:
            (batch, seq_len, out_dim)
        """
        B, T, D = x.shape

        # Reshape for BatchNorm1d: (B*T, D)
        x = x.reshape(B * T, D)

        # Apply MLP
        x = self.mlp(x)

        # Reshape back: (B, T, out_dim)
        x = x.reshape(B, T, self.out_dim)

        return x


def get_projection(
    projection_type: str,
    in_dim: int,
    out_dim: int = 128,
    **kwargs
) -> nn.Module:
    """
    Factory function to create projection module.

    Args:
        projection_type: 'linear' or 'mlp'
        in_dim: Input dimension from frontend
        out_dim: Output dimension for AASIST (default: 128)
        **kwargs: Additional arguments for MLPProjection

    Returns:
        Projection module

    Example:
        # Linear projection
        proj = get_projection('linear', in_dim=1280)

        # MLP projection with custom config
        proj = get_projection(
            'mlp',
            in_dim=1280,
            hidden_dims=[512, 256],
            activation='gelu',
            dropout=0.1
        )
    """
    if projection_type == 'linear':
        return LinearProjection(in_dim=in_dim, out_dim=out_dim)
    elif projection_type == 'mlp':
        return MLPProjection(in_dim=in_dim, out_dim=out_dim, **kwargs)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}. Choose 'linear' or 'mlp'")
