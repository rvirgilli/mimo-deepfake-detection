"""
MiMo Feature Extraction Strategies.

This module provides different strategies for extracting features from the
MiMo encoder for deepfake detection:

- continuous: Pre-quantization hidden states (default, 1280-dim)
- rvq_sum: Sum of all RVQ layer embeddings (1280-dim)
- rvq_concat: Concatenated RVQ embeddings from all layers (25,600-dim)
- rvq_fine: Fine-detail RVQ layers only (configurable dim)
- dual_stream: Continuous + RVQ sum concatenated (2,560-dim)
- weighted: Learnable per-layer weights before summing (1280-dim)

Each strategy captures different aspects of the audio representation:
- Layers 0-1 (1024 codebook): Coarse semantic features
- Layers 2-19 (128 codebook): Fine acoustic details / artifacts
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeatureStrategy(nn.Module, ABC):
    """
    Base class for MiMo feature extraction strategies.

    Subclasses must implement:
    - out_dim property: Output feature dimension
    - extract(): Feature extraction from tokenizer (25Hz)
    - extract_50hz(): Feature extraction at native 50Hz (optional override)
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Output dimension of extracted features."""
        pass

    @abstractmethod
    def extract(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        """
        Extract features using this strategy at 25Hz.

        Args:
            tokenizer: MiMoAudioTokenizer instance
            mels: Mel spectrogram (batch, n_mels, seq_len)
            mels_lens: Actual mel lengths (batch,)
            output_length: Output sequence length at 50Hz (batch,)

        Returns:
            features: Extracted features (batch, seq_len_25hz, out_dim)
        """
        pass

    def extract_50hz(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        """
        Extract features at native 50Hz resolution.

        Default implementation extracts at 25Hz and upsamples.
        Subclasses can override for native 50Hz extraction.

        Args:
            tokenizer: MiMoAudioTokenizer instance with get_features_50hz() patched
            mels: Mel spectrogram (batch, n_mels, seq_len)
            mels_lens: Actual mel lengths (batch,)
            output_length: Output sequence length at 50Hz (batch,)

        Returns:
            features: Extracted features (batch, seq_len_50hz, out_dim)
        """
        # Default: extract at 25Hz and upsample
        features = self.extract(tokenizer, mels, mels_lens, output_length)
        target_length = output_length.max().item()
        return self._upsample_to_length(features, target_length)

    def _upsample_to_length(self, features: Tensor, target_length: int) -> Tensor:
        """Upsample features to target temporal length."""
        # features: (batch, seq_len, dim) -> transpose -> interpolate -> transpose
        features = features.transpose(1, 2)
        features = F.interpolate(
            features, size=target_length, mode='linear', align_corners=False
        )
        return features.transpose(1, 2)


class ContinuousStrategy(FeatureStrategy):
    """
    Pre-quantization continuous features (default).

    Extracts the hidden states from the encoder before quantization.
    This preserves all continuous information before discretization.

    Output: (batch, seq_len, 1280)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._out_dim = 1280

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def extract(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        hidden_states, _ = tokenizer.encoder.get_features(mels, output_length)
        return hidden_states

    def extract_50hz(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        """Native 50Hz extraction - skip final downsampling."""
        hidden_states, _ = tokenizer.encoder.get_features_50hz(mels, output_length)
        return hidden_states


class RVQSumStrategy(FeatureStrategy):
    """
    Sum of RVQ embeddings across all quantizer layers.

    This is the standard VQ reconstruction approach - sums embeddings
    from all 20 quantizer layers to reconstruct the representation.

    Output: (batch, seq_len, 1280)
    """

    def __init__(self, n_q: int = 20, **kwargs):
        super().__init__()
        self._out_dim = 1280
        self.n_q = n_q

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def extract(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        # Get discrete tokens: (batch, seq_len, n_q)
        codes, _ = tokenizer.encoder.encode(
            mels, mels_lens, output_length=output_length, n_q=self.n_q
        )

        batch_size, seq_len, n_q = codes.shape
        device = codes.device

        # Sum embeddings from all layers
        summed = torch.zeros(batch_size, seq_len, self._out_dim, device=device)

        for i in range(n_q):
            codebook = tokenizer.encoder.quantizer.vq.layers[i].codebook  # (codebook_size, 1280)
            emb = F.embedding(codes[:, :, i], codebook)  # (batch, seq_len, 1280)
            summed = summed + emb

        return summed


class RVQConcatStrategy(FeatureStrategy):
    """
    Concatenated RVQ embeddings from all quantizer layers.

    Concatenates embeddings from all 20 layers for a rich but
    high-dimensional representation.

    Output: (batch, seq_len, 20 * 1280 = 25,600)
    """

    def __init__(self, n_q: int = 20, **kwargs):
        super().__init__()
        self.n_q = n_q
        self._out_dim = n_q * 1280

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def extract(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        # Get discrete tokens
        codes, _ = tokenizer.encoder.encode(
            mels, mels_lens, output_length=output_length, n_q=self.n_q
        )

        batch_size, seq_len, n_q = codes.shape

        # Lookup and concatenate embeddings from all layers
        embeddings = []
        for i in range(n_q):
            codebook = tokenizer.encoder.quantizer.vq.layers[i].codebook
            emb = F.embedding(codes[:, :, i], codebook)  # (batch, seq_len, 1280)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=-1)  # (batch, seq_len, n_q * 1280)


class RVQFineStrategy(FeatureStrategy):
    """
    Fine-detail RVQ layers only, skipping coarse semantic layers.

    Hypothesis: Layers 0-1 capture coarse semantic content, while
    layers 2+ capture fine acoustic details where deepfake artifacts
    may be more evident.

    Default: Layers 2-8 (7 layers * 1280 = 8,960-dim)
    """

    def __init__(
        self,
        start_layer: int = 2,
        end_layer: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.n_layers = end_layer - start_layer + 1
        self._out_dim = self.n_layers * 1280

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def extract(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        # Need all layers up to end_layer for encoding
        n_q_needed = self.end_layer + 1

        codes, _ = tokenizer.encoder.encode(
            mels, mels_lens, output_length=output_length, n_q=n_q_needed
        )

        batch_size, seq_len, _ = codes.shape

        # Only use layers [start_layer, end_layer]
        embeddings = []
        for i in range(self.start_layer, self.end_layer + 1):
            codebook = tokenizer.encoder.quantizer.vq.layers[i].codebook
            emb = F.embedding(codes[:, :, i], codebook)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=-1)


class DualStreamStrategy(FeatureStrategy):
    """
    Parallel continuous + RVQ sum streams.

    Combines pre-quantization continuous features with post-quantization
    RVQ reconstruction. Hypothesis: continuous stream preserves nuances
    lost in discretization, while RVQ stream provides robust features.

    Output: (batch, seq_len, 2 * 1280 = 2,560)
    """

    def __init__(self, n_q: int = 20, **kwargs):
        super().__init__()
        self.n_q = n_q
        self._out_dim = 2 * 1280

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def extract(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        # Continuous branch
        continuous, _ = tokenizer.encoder.get_features(mels, output_length)

        # RVQ branch - sum of embeddings
        codes, _ = tokenizer.encoder.encode(
            mels, mels_lens, output_length=output_length, n_q=self.n_q
        )

        batch_size, seq_len, n_q = codes.shape
        device = codes.device

        rvq_sum = torch.zeros(batch_size, seq_len, 1280, device=device)
        for i in range(n_q):
            codebook = tokenizer.encoder.quantizer.vq.layers[i].codebook
            emb = F.embedding(codes[:, :, i], codebook)
            rvq_sum = rvq_sum + emb

        return torch.cat([continuous, rvq_sum], dim=-1)

    def extract_50hz(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        """
        Hybrid 50Hz: native continuous + upsampled RVQ.

        Continuous branch gets native 50Hz features (no interpolation artifacts).
        RVQ branch gets 25Hz features upsampled to 50Hz (RVQ inherently at 25Hz).
        """
        # Continuous branch: native 50Hz
        continuous_50hz, _ = tokenizer.encoder.get_features_50hz(mels, output_length)
        target_length = continuous_50hz.size(1)

        # RVQ branch: 25Hz → upsample to 50Hz
        codes, _ = tokenizer.encoder.encode(
            mels, mels_lens, output_length=output_length, n_q=self.n_q
        )

        batch_size, seq_len_25hz, n_q = codes.shape
        device = codes.device

        rvq_sum_25hz = torch.zeros(batch_size, seq_len_25hz, 1280, device=device)
        for i in range(n_q):
            codebook = tokenizer.encoder.quantizer.vq.layers[i].codebook
            emb = F.embedding(codes[:, :, i], codebook)
            rvq_sum_25hz = rvq_sum_25hz + emb

        # Upsample RVQ to 50Hz
        rvq_sum_50hz = self._upsample_to_length(rvq_sum_25hz, target_length)

        return torch.cat([continuous_50hz, rvq_sum_50hz], dim=-1)


class WeightedRVQStrategy(FeatureStrategy):
    """
    Learnable per-layer weights before summing RVQ embeddings.

    Allows the model to learn which quantizer layers are most
    discriminative for deepfake detection.

    Output: (batch, seq_len, 1280)
    """

    def __init__(self, n_q: int = 20, **kwargs):
        super().__init__()
        self.n_q = n_q
        self._out_dim = 1280

        # Learnable layer weights (initialized to uniform)
        self.layer_weights = nn.Parameter(torch.ones(n_q))

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def extract(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        codes, _ = tokenizer.encoder.encode(
            mels, mels_lens, output_length=output_length, n_q=self.n_q
        )

        batch_size, seq_len, n_q = codes.shape
        device = codes.device

        # Softmax to normalize weights
        weights = F.softmax(self.layer_weights, dim=0)

        weighted_sum = torch.zeros(batch_size, seq_len, self._out_dim, device=device)
        for i in range(n_q):
            codebook = tokenizer.encoder.quantizer.vq.layers[i].codebook
            emb = F.embedding(codes[:, :, i], codebook)
            weighted_sum = weighted_sum + weights[i] * emb

        return weighted_sum

    def get_layer_importance(self) -> Tensor:
        """Return normalized layer weights for analysis."""
        return F.softmax(self.layer_weights, dim=0).detach()


class LayerSelectStrategy(FeatureStrategy):
    """
    Layer-selected continuous features from transformer layers.

    Research shows using only a subset of transformer layers (e.g., 12 of 32)
    dramatically improves deepfake detection (0.22% EER vs ~10% with all layers).

    Key insight: Middle layers capture discriminative artifacts while early
    layers are too raw and later layers too semantic/high-level.

    Args:
        num_layers: Number of transformer layers to use (default: 12)
        start_layer: First layer to use (default: 0, use 2 for layers 2-13)

    Output: (batch, seq_len, 1280)
    """

    def __init__(self, num_layers: int = 12, start_layer: int = 0, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.start_layer = start_layer
        self._out_dim = 1280

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def extract(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        """Extract features at 25Hz using layer selection."""
        # Ensure layer selection patch is available
        if not hasattr(tokenizer.encoder, 'get_features_layer_select'):
            from .mimo_native50hz import patch_encoder_for_layer_select
            patch_encoder_for_layer_select(tokenizer)

        hidden_states, _ = tokenizer.encoder.get_features_layer_select(
            mels, output_length,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
            output_50hz=False,
        )
        return hidden_states

    def extract_50hz(
        self,
        tokenizer,
        mels: Tensor,
        mels_lens: Tensor,
        output_length: Tensor,
    ) -> Tensor:
        """Extract features at native 50Hz using layer selection."""
        # Ensure layer selection patch is available
        if not hasattr(tokenizer.encoder, 'get_features_layer_select'):
            from .mimo_native50hz import patch_encoder_for_layer_select
            patch_encoder_for_layer_select(tokenizer)

        hidden_states, _ = tokenizer.encoder.get_features_layer_select(
            mels, output_length,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
            output_50hz=True,
        )
        return hidden_states


# Strategy registry
_STRATEGY_REGISTRY: Dict[str, type] = {
    'continuous': ContinuousStrategy,
    'rvq_sum': RVQSumStrategy,
    'rvq_concat': RVQConcatStrategy,
    'rvq_fine': RVQFineStrategy,
    'dual_stream': DualStreamStrategy,
    'weighted': WeightedRVQStrategy,
    'layer_select': LayerSelectStrategy,
}


def get_feature_strategy(name: str, **kwargs) -> FeatureStrategy:
    """
    Factory function to create a feature extraction strategy.

    Args:
        name: Strategy name (continuous, rvq_sum, rvq_concat, rvq_fine,
              dual_stream, weighted)
        **kwargs: Strategy-specific configuration

    Returns:
        Initialized FeatureStrategy instance

    Example:
        strategy = get_feature_strategy('rvq_fine', start_layer=2, end_layer=8)
    """
    if name not in _STRATEGY_REGISTRY:
        available = ', '.join(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")

    return _STRATEGY_REGISTRY[name](**kwargs)


def list_strategies() -> Dict[str, str]:
    """List available strategies with their descriptions."""
    return {
        'continuous': 'Pre-quantization hidden states (1280-dim)',
        'rvq_sum': 'Sum of all RVQ layer embeddings (1280-dim)',
        'rvq_concat': 'Concatenated RVQ embeddings (25,600-dim)',
        'rvq_fine': 'Fine-detail layers only (configurable)',
        'dual_stream': 'Continuous + RVQ sum (2,560-dim)',
        'weighted': 'Learnable layer weights (1280-dim)',
        'layer_select': 'Subset of transformer layers (1280-dim, recommended)',
    }
