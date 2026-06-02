"""
EnCodec Frontend for audio feature extraction.

This module wraps Meta's EnCodec neural audio codec to extract
continuous encoder features (before RVQ quantization) for use
in audio deepfake detection.

EnCodec is a reconstruction-based model, unlike wav2vec2's contrastive
pre-training. This allows us to test whether the reconstruction objective
(not just MiMo specifically) underperforms for discrimination tasks.

References:
- Paper: "High Fidelity Neural Audio Compression" (Défossez et al., 2022)
- HuggingFace: https://huggingface.co/facebook/encodec_24khz
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseFrontend


class EnCodecFrontend(BaseFrontend):
    """
    EnCodec frontend feature extractor.

    Extracts continuous encoder features BEFORE RVQ quantization.
    This preserves maximum information for downstream classification.

    Args:
        model_name: HuggingFace model name (default: facebook/encodec_24khz)
        freeze: Whether to freeze encoder weights (default: True)

    Attributes:
        out_dim: 128 for encodec_24khz (encoder hidden size)
        sample_rate: 24000 Hz for encodec_24khz
        frame_rate: ~75 Hz (24000 / 320 stride)
    """

    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        freeze: bool = True,
    ):
        super().__init__()

        try:
            from transformers import EncodecModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for EnCodecFrontend. "
                "Install with: pip install transformers"
            )

        # Load model and processor
        self.model = EncodecModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Get model config
        self._sample_rate = self.processor.sampling_rate  # 24000 for 24khz model
        self._out_dim = self.model.config.hidden_size  # 128 for 24khz model

        # EnCodec uses bandwidth to control quality, we use highest
        self.bandwidth = 24.0  # kbps, highest quality

        if freeze:
            self.freeze()

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def extract_feat(self, input_data: Tensor) -> Tensor:
        """
        Extract continuous encoder features before RVQ.

        Args:
            input_data: Raw audio waveform at 24kHz
                Shape: (batch, samples) or (batch, samples, 1)

        Returns:
            features: Continuous encoder embeddings
                Shape: (batch, time, 128)
        """
        # Handle input shape
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        # Ensure model is on correct device
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device)

        # EnCodec expects (batch, channels, samples) - mono audio
        if input_data.ndim == 2:
            input_data = input_data.unsqueeze(1)  # (batch, 1, samples)

        # Extract encoder embeddings directly (before quantization)
        # The encoder outputs (batch, hidden_size, time)
        with torch.set_grad_enabled(self.training and not self._is_frozen()):
            embeddings = self.model.encoder(input_data)

        # Transpose to (batch, time, hidden_size) for consistency with other frontends
        embeddings = embeddings.transpose(1, 2)

        return embeddings

    def _is_frozen(self) -> bool:
        """Check if model is frozen."""
        return not any(p.requires_grad for p in self.model.parameters())


class EnCodecRVQFrontend(BaseFrontend):
    """
    EnCodec frontend using post-quantization RVQ embeddings.

    This extracts the quantized embeddings (sum of RVQ codebook vectors)
    rather than continuous pre-quantization features.

    Use this to compare whether quantization hurts performance.
    """

    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        freeze: bool = True,
        bandwidth: float = 24.0,
    ):
        super().__init__()

        from transformers import EncodecModel, AutoProcessor

        self.model = EncodecModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        self._sample_rate = self.processor.sampling_rate
        # RVQ embeddings have same dim as encoder output
        self._out_dim = self.model.config.hidden_size
        self.bandwidth = bandwidth

        if freeze:
            self.freeze()

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def extract_feat(self, input_data: Tensor) -> Tensor:
        """
        Extract RVQ embeddings (post-quantization).

        Args:
            input_data: Raw audio at 24kHz, shape (batch, samples)

        Returns:
            features: RVQ embeddings, shape (batch, time, 128)
        """
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device)

        if input_data.ndim == 2:
            input_data = input_data.unsqueeze(1)

        with torch.no_grad():
            # Get encoder output
            embeddings = self.model.encoder(input_data)

            # Quantize to get codes
            codes = self.model.quantizer.encode(embeddings, bandwidth=self.bandwidth)

            # Decode codes back to embeddings (sum of codebook vectors)
            quantized = self.model.quantizer.decode(codes)

        # (batch, hidden, time) -> (batch, time, hidden)
        return quantized.transpose(1, 2)
