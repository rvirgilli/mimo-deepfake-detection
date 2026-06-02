"""
MiMo-Audio-Tokenizer Frontend for audio feature extraction.

This module wraps the MiMo encoder to extract features for use with the
AASIST backend. Supports multiple feature extraction strategies:

Feature strategies (see mimo_features.py):
- continuous: Pre-quantization hidden states (1280-dim, default)
- rvq_sum: Sum of all RVQ layer embeddings (1280-dim)
- rvq_concat: Concatenated RVQ embeddings (25,600-dim)
- rvq_fine: Fine-detail layers only (configurable dim)
- dual_stream: Continuous + RVQ sum (2,560-dim)
- weighted: Learnable layer weights (1280-dim)

Fine-tuning strategies (see mimo_finetune.py):
- frozen: All encoder params frozen (~0.1M trainable in projection only)
- full: All encoder params trainable (encoder-only, not full tokenizer)
- adapter: Bottleneck adapters (~2M params) [RECOMMENDED]
- lora: Low-rank adaptation (~1M params)
- partial: Last N layers only (~150-600M params)

Requirements:
- mimo_audio_tokenizer package
- flash-attn (for optimal performance)
- Model weights from HuggingFace: XiaomiMiMo/MiMo-Audio-Tokenizer
"""

import os
from typing import Optional, List, Dict, Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseFrontend

# Guarded import for MiMo (requires Python 3.12 with flash-attn)
try:
    import mimo_audio_tokenizer
    MIMO_AVAILABLE = True
except ImportError:
    MIMO_AVAILABLE = False
    mimo_audio_tokenizer = None


class LearnableUpsample(nn.Module):
    """
    Learnable 2x upsampling using transposed convolution.

    Unlike linear interpolation, this learns the upsampling kernel,
    potentially capturing better temporal patterns for audio features.

    Args:
        in_channels: Number of input channels (feature dimension, default: 1280)
        kernel_size: Kernel size for transposed conv (default: 4)

    Note:
        With kernel_size=4, stride=2, padding=1:
        output_length = (input_length - 1) * 2 + 4 - 2 * 1 = input_length * 2
    """

    def __init__(self, in_channels: int = 1280, kernel_size: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Calculate padding to achieve exact 2x upsampling
        # For kernel_size=4, stride=2: padding=1 gives exact 2x
        padding = kernel_size // 2 - 1

        self.upsample = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=0,
        )

        # Initialize close to linear interpolation for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize to approximate linear interpolation."""
        # Initialize with small values for stable training
        nn.init.xavier_uniform_(self.upsample.weight, gain=0.1)
        if self.upsample.bias is not None:
            nn.init.zeros_(self.upsample.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, dim) - batch, time, features

        Returns:
            (batch, seq_len*2, dim) - upsampled features
        """
        # ConvTranspose1d expects (B, C, T)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.upsample(x)   # (B, D, T*2)
        x = x.transpose(1, 2)  # (B, T*2, D)
        return x


class MiMoFrontend(BaseFrontend):
    """
    MiMo-Audio-Tokenizer frontend feature extractor.

    This frontend uses the MiMo-Audio-Tokenizer encoder to extract
    features at 25Hz (or 50Hz with upsampling). Supports multiple
    feature extraction strategies for different representations.

    Args:
        model_path: Path to MiMo model directory
        use_bfloat16: Whether to use bfloat16 precision (required for Flash Attention)
        upsample_to_50hz: Whether to upsample features from 25Hz to 50Hz (default: False)
            This matches wav2vec2's frame rate for better AASIST compatibility.
        upsample_mode: Upsampling method when upsample_to_50hz=True (default: 'linear')
            - 'linear': Linear interpolation (simple, no learnable params)
            - 'nearest': Nearest neighbor (duplicates frames)
            - 'learnable': ConvTranspose1d (learnable upsampling kernel)
        native_50hz: Whether to extract 50Hz features directly (default: False)
            Extracts features before final downsampling for higher temporal resolution.
            Mutually exclusive with upsample_to_50hz - native_50hz takes precedence.
        feature_type: Feature extraction strategy (default: 'continuous')
            - 'continuous': Pre-quantization hidden states (1280-dim)
            - 'rvq_sum': Sum of all RVQ layer embeddings (1280-dim)
            - 'rvq_concat': Concatenated RVQ embeddings (25,600-dim)
            - 'rvq_fine': Fine-detail layers only (configurable)
            - 'dual_stream': Continuous + RVQ sum (2,560-dim)
            - 'weighted': Learnable layer weights (1280-dim)
        feature_config: Feature-specific configuration dict
        finetune_config: Fine-tuning configuration dict with keys:
            - strategy: 'frozen', 'full', 'adapter', 'lora', 'partial'
            - adapter: dict with dim, dropout, layers, n_layers
            - lora: dict with rank, alpha, dropout, target_modules
            - partial: dict with n_trainable_layers

    Attributes:
        out_dim: Feature dimension (varies by strategy)
        sample_rate: 24000 Hz
    """

    _sample_rate: int = 24000

    def __init__(
        self,
        model_path: str = "./models/MiMo-Audio-Tokenizer",
        use_bfloat16: bool = True,
        upsample_to_50hz: bool = False,
        upsample_mode: str = "linear",
        native_50hz: bool = False,
        feature_type: str = "continuous",
        feature_config: Optional[Dict[str, Any]] = None,
        finetune_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if not MIMO_AVAILABLE:
            raise ImportError(
                "mimo_audio_tokenizer is required for MiMoFrontend. "
                "Please install it: pip install -e MiMo-Audio-Tokenizer"
            )

        # Resolve model path
        if not os.path.isabs(model_path):
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(project_root, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MiMo model not found: {model_path}. "
                "Download from HuggingFace: "
                "huggingface-cli download XiaomiMiMo/MiMo-Audio-Tokenizer --local-dir ./models/MiMo-Audio-Tokenizer"
            )

        # Load MiMo tokenizer
        self.tokenizer = mimo_audio_tokenizer.load_model(model_path)
        self.config = self.tokenizer.config

        # Use bfloat16 for Flash Attention compatibility
        self.use_bfloat16 = use_bfloat16
        if use_bfloat16:
            self.tokenizer = self.tokenizer.bfloat16()

        # Store frame rate settings
        self.upsample_to_50hz = upsample_to_50hz
        self.upsample_mode = upsample_mode
        self.native_50hz = native_50hz

        # Validate upsample_mode
        valid_modes = ('linear', 'nearest', 'learnable')
        if upsample_mode not in valid_modes:
            raise ValueError(f"upsample_mode must be one of {valid_modes}, got '{upsample_mode}'")

        # Create learnable upsampler if needed (feature dim = 1280 for MiMo)
        self.learnable_upsampler: Optional[LearnableUpsample] = None
        if upsample_to_50hz and upsample_mode == 'learnable':
            self.learnable_upsampler = LearnableUpsample(in_channels=1280)
            print("Using learnable upsampling (ConvTranspose1d)")

        # Native 50Hz extraction (skip final downsampling in encoder)
        if native_50hz:
            from .mimo_native50hz import patch_encoder_for_50hz
            patch_encoder_for_50hz(self.tokenizer)
            if upsample_to_50hz:
                print("Warning: native_50hz=True takes precedence over upsample_to_50hz")
            print("Using native 50Hz feature extraction (no final downsampling)")

        # Feature extraction strategy
        from .mimo_features import get_feature_strategy
        self._feature_type = feature_type
        self.feature_strategy = get_feature_strategy(
            feature_type, **(feature_config or {})
        )
        self._out_dim = self.feature_strategy.out_dim
        print(f"Feature strategy: {feature_type} (output dim: {self._out_dim})")

        # Fine-tuning setup - use register_module for proper parameter tracking
        self._finetune_strategy = 'frozen'
        self.finetune_wrapper: Optional[nn.Module] = None  # Will be set by _apply_finetune_strategy

        if finetune_config is not None:
            self._apply_finetune_strategy(finetune_config)
        else:
            # Default: freeze everything
            self.freeze()

    def _apply_finetune_strategy(self, config: Dict[str, Any]) -> None:
        """Apply fine-tuning strategy based on config."""
        from .mimo_finetune import (
            MiMoWithAdapters,
            MiMoWithLoRA,
            MiMoPartialFinetune,
        )

        strategy = config.get('strategy', 'frozen')
        self._finetune_strategy = strategy

        print(f"\nApplying fine-tuning strategy: {strategy}")

        if strategy == 'frozen':
            self.freeze()
            self.finetune_wrapper = None

        elif strategy == 'full':
            self.unfreeze()
            self.finetune_wrapper = None
            total_params = sum(p.numel() for p in self.tokenizer.parameters())
            print(f"Full fine-tuning: {total_params:,} parameters ({total_params/1e9:.2f}B)")

        elif strategy == 'adapter':
            adapter_cfg = config.get('adapter', {})
            # Register as submodule so adapter params are included in model.parameters()
            self.finetune_wrapper = MiMoWithAdapters(
                self.tokenizer,
                adapter_dim=adapter_cfg.get('dim', 64),
                adapter_dropout=adapter_cfg.get('dropout', 0.1),
                adapter_layers=adapter_cfg.get('layers', 'last_n'),
                n_adapter_layers=adapter_cfg.get('n_layers', 8),
            )

        elif strategy == 'lora':
            lora_cfg = config.get('lora', {})
            # Register as submodule so LoRA params are included in model.parameters()
            self.finetune_wrapper = MiMoWithLoRA(
                self.tokenizer,
                lora_rank=lora_cfg.get('rank', 8),
                lora_alpha=lora_cfg.get('alpha', 16.0),
                lora_dropout=lora_cfg.get('dropout', 0.0),
                target_modules=lora_cfg.get('target_modules', None),
            )

        elif strategy == 'partial':
            partial_cfg = config.get('partial', {})
            self.finetune_wrapper = MiMoPartialFinetune(
                self.tokenizer,
                n_trainable_layers=partial_cfg.get('n_trainable_layers', 4),
            )

        else:
            raise ValueError(
                f"Unknown finetune strategy: {strategy}. "
                "Choose from: frozen, full, adapter, lora, partial"
            )

    @property
    def finetune_strategy(self) -> str:
        """Return the current fine-tuning strategy."""
        return self._finetune_strategy

    @property
    def feature_type(self) -> str:
        """Return the current feature extraction strategy."""
        return self._feature_type

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.tokenizer.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.tokenizer.parameters():
            param.requires_grad = True

    @property
    def num_params(self) -> int:
        """Total number of parameters in encoder."""
        return sum(p.numel() for p in self.tokenizer.parameters())

    @property
    def num_trainable_params(self) -> int:
        """Number of trainable parameters (encoder + adapters/LoRA + feature strategy + upsampler)."""
        # Encoder trainable params
        encoder_trainable = sum(
            p.numel() for p in self.tokenizer.parameters() if p.requires_grad
        )

        # Wrapper trainable params (adapters/LoRA)
        wrapper_trainable = 0
        if self.finetune_wrapper is not None:
            wrapper_trainable = self.finetune_wrapper.num_trainable_params()

        # Feature strategy trainable params (e.g., WeightedRVQStrategy)
        strategy_trainable = sum(
            p.numel() for p in self.feature_strategy.parameters() if p.requires_grad
        )

        # Learnable upsampler params
        upsampler_trainable = 0
        if self.learnable_upsampler is not None:
            upsampler_trainable = sum(
                p.numel() for p in self.learnable_upsampler.parameters() if p.requires_grad
            )

        return encoder_trainable + wrapper_trainable + strategy_trainable + upsampler_trainable

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """
        Get iterator over trainable parameters.

        Use this to create optimizer:
            optimizer = Adam(frontend.get_trainable_params(), lr=1e-4)
        """
        # Encoder trainable params
        for p in self.tokenizer.parameters():
            if p.requires_grad:
                yield p

        # Wrapper trainable params (adapters/LoRA)
        if self.finetune_wrapper is not None:
            yield from self.finetune_wrapper.get_trainable_params()

        # Feature strategy trainable params (e.g., WeightedRVQStrategy)
        for p in self.feature_strategy.parameters():
            if p.requires_grad:
                yield p

        # Learnable upsampler params
        if self.learnable_upsampler is not None:
            for p in self.learnable_upsampler.parameters():
                if p.requires_grad:
                    yield p

    def _prepare_mel(self, waveforms: Tensor) -> tuple:
        """
        Convert waveforms to mel spectrograms with proper padding.

        Args:
            waveforms: (batch, samples) at 24kHz

        Returns:
            mels: (batch, n_mels, max_seq_len)
            mels_lens: (batch,)
        """
        mels_list = []
        for wav in waveforms:
            mel = mimo_audio_tokenizer.mel_spectrogram(wav.cpu(), self.config)
            mels_list.append(mel)

        # Pad and stack
        mels, mels_lens = mimo_audio_tokenizer.padding(mels_list)
        return mels, mels_lens

    def extract_feat(self, input_data: Tensor) -> Tensor:
        """
        Extract features from raw audio waveform using the configured strategy.

        Note: Gradients are enabled/disabled based on the fine-tuning strategy.
        For 'frozen' strategy, uses torch.no_grad(). For other strategies,
        gradients flow through the encoder.

        Args:
            input_data: Raw audio waveform at 24kHz
                Shape: (batch, samples) or (batch, samples, 1)

        Returns:
            features: Extracted feature tensor
                Shape: (batch, seq_len, out_dim) where out_dim depends on strategy
        """
        # Handle input shape
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        device = input_data.device

        # Convert to mel spectrogram (always on CPU for consistency)
        mels, mels_lens = self._prepare_mel(input_data)
        mels = mels.to(device)
        mels_lens = mels_lens.to(device)

        # Cast to model dtype
        if self.use_bfloat16:
            mels = mels.bfloat16()

        # Get output length (this is 50Hz length, before final downsampling)
        output_length = self.tokenizer.encoder.get_output_length(mels_lens)

        # Extract features using strategy - gradient handling based on finetune strategy
        if self._finetune_strategy == 'frozen':
            with torch.no_grad():
                if self.native_50hz:
                    # Native 50Hz extraction (skip final downsampling)
                    features = self.feature_strategy.extract_50hz(
                        self.tokenizer, mels, mels_lens, output_length
                    )
                else:
                    # Standard 25Hz extraction
                    features = self.feature_strategy.extract(
                        self.tokenizer, mels, mels_lens, output_length
                    )
        else:
            # Gradients flow for adapter/lora/partial/full strategies
            if self.native_50hz:
                features = self.feature_strategy.extract_50hz(
                    self.tokenizer, mels, mels_lens, output_length
                )
            else:
                features = self.feature_strategy.extract(
                    self.tokenizer, mels, mels_lens, output_length
                )

        # Convert back to float32 for downstream processing
        if features.dtype == torch.bfloat16:
            features = features.float()

        # Upsample from 25Hz to 50Hz if enabled (only when not using native 50Hz)
        if self.upsample_to_50hz and not self.native_50hz:
            if self.upsample_mode == 'learnable':
                # Use learnable ConvTranspose1d upsampling
                features = self.learnable_upsampler(features)
            else:
                # Use interpolation (linear or nearest)
                # features: (batch, seq_len, dim) -> transpose -> interpolate -> transpose back
                features = features.transpose(1, 2)  # (batch, dim, seq_len)
                features = F.interpolate(
                    features, scale_factor=2, mode=self.upsample_mode, align_corners=False if self.upsample_mode == 'linear' else None
                )
                features = features.transpose(1, 2)  # (batch, seq_len*2, dim)

        return features
