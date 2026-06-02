"""
Base frontend interface for audio feature extraction.

This module defines the abstract interface that all frontend feature extractors
(wav2vec2, MiMo, etc.) must implement for compatibility with the AASIST backend.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BaseFrontend(nn.Module, ABC):
    """
    Abstract base class for audio frontend feature extractors.

    All frontends must implement:
    - out_dim: Output feature dimension
    - sample_rate: Expected input sample rate
    - extract_feat(): Feature extraction method

    This abstraction allows swapping between wav2vec2 and MiMo frontends
    while keeping the AASIST backend unchanged.
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Output feature dimension (e.g., 1024 for wav2vec2, 1280 for MiMo)"""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Expected input sample rate (e.g., 16000 for wav2vec2, 24000 for MiMo)"""
        pass

    @abstractmethod
    def extract_feat(self, input_data: Tensor) -> Tensor:
        """
        Extract features from raw audio waveform.

        Args:
            input_data: Raw audio waveform tensor
                Shape: (batch, samples) or (batch, samples, 1)

        Returns:
            features: Extracted feature tensor
                Shape: (batch, seq_len, out_dim)
        """
        pass

    def freeze(self) -> None:
        """Freeze all parameters (for using pretrained frontend)"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters (for fine-tuning)"""
        for param in self.parameters():
            param.requires_grad = True

    @property
    def num_params(self) -> int:
        """Total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        """Number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
