"""
Native 50Hz feature extraction for MiMo encoder.

This module provides methods to extract features at 50Hz (before final
downsampling) rather than the default 25Hz output. MiMo's encoder performs
two temporal downsampling steps:

1. Conv2 with stride=2: 100Hz → 50Hz
2. down_sample_layer with avg_pooler=2: 50Hz → 25Hz

By skipping the second downsampling step, we can extract richer 50Hz features
that better match wav2vec2's frame rate for AASIST compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


def get_features_50hz(encoder, input_features: Tensor, output_length: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Extract 50Hz features from MiMo encoder (before final downsampling).

    This replicates the encoder's get_features() but stops before
    the down_sample_layer, yielding 50Hz output instead of 25Hz.

    Args:
        encoder: MiMo AudioEncoder instance
        input_features: Mel spectrogram [batch, n_mels, seq_len]
        output_length: Sequence lengths [batch] (at 50Hz resolution)

    Returns:
        hidden_states: 50Hz features [batch, seq_len_50hz, 1280]
        output_length: Unchanged (already 50Hz length)
    """
    from mimo_audio_tokenizer.utils import (
        get_position_ids, packing, unpacking
    )

    # Conv layers - process mel spectrogram
    input_features = input_features.to(encoder.conv1.weight)
    inputs_embeds = nn.functional.gelu(encoder.conv1(input_features))
    inputs_embeds = nn.functional.gelu(encoder.conv2(inputs_embeds))
    # Now at 50Hz: [batch, d_model, seq_len_50hz]

    # Transpose to [batch, seq_len_50hz, d_model]
    hidden_states = inputs_embeds.permute(0, 2, 1)

    # Get position embeddings (RoPE)
    position_ids = get_position_ids(output_length).long().to(input_features.device)
    rope_position_embeddings = encoder.position_embedding(input_features, position_ids)

    # Pack sequences for efficient transformer processing
    hidden_states = packing(hidden_states, output_length)
    skip_connect_hidden_states = 0.0

    # Process through all transformer layers (matches original get_features)
    for idx, encoder_layer in enumerate(encoder.layers):
        hidden_states = encoder_layer(
            hidden_states,
            output_length,
            rope_position_embeddings=rope_position_embeddings,
        )
        # Handle skip connection at specific layer
        if (encoder.skip_layer_idx is not None) and idx == encoder.skip_layer_idx - 1:
            skip_connect_hidden_states = hidden_states.clone()

    # Apply skip connection and layer normalization
    hidden_states = hidden_states + skip_connect_hidden_states
    hidden_states = encoder.layer_norm(hidden_states)

    # Unpack back to batch format
    hidden_states = unpacking(hidden_states, output_length)
    # Shape: [batch, seq_len_50hz, d_model]

    # NOTE: We intentionally SKIP the down_sample_layer here!
    # Standard get_features() would do:
    #   hidden_states = encoder.down_sample_layer(hidden_states.transpose(1, 2))
    #   output_length = output_length // encoder.config.avg_pooler + ...
    # But we return 50Hz features directly without the final 2x downsampling

    return hidden_states, output_length


def patch_encoder_for_50hz(tokenizer) -> None:
    """
    Patch the tokenizer's encoder to support 50Hz extraction.

    Adds a `get_features_50hz` method to the encoder that extracts
    features before the final downsampling step.

    Args:
        tokenizer: MiMoAudioTokenizer instance

    Usage:
        patch_encoder_for_50hz(tokenizer)
        features_50hz, lengths = tokenizer.encoder.get_features_50hz(mels, output_length)
    """
    import types

    def _get_features_50hz(self, input_features: Tensor, output_length: Tensor) -> Tuple[Tensor, Tensor]:
        return get_features_50hz(self, input_features, output_length)

    tokenizer.encoder.get_features_50hz = types.MethodType(
        _get_features_50hz,
        tokenizer.encoder
    )


def upsample_to_length(features: Tensor, target_length: int, mode: str = 'linear') -> Tensor:
    """
    Upsample features to a target temporal length.

    Args:
        features: Input features [batch, seq_len, dim]
        target_length: Target sequence length
        mode: Interpolation mode ('linear', 'nearest')

    Returns:
        Upsampled features [batch, target_length, dim]
    """
    # Transpose for F.interpolate: [batch, dim, seq_len]
    features = features.transpose(1, 2)
    features = F.interpolate(features, size=target_length, mode=mode, align_corners=False if mode == 'linear' else None)
    # Transpose back: [batch, target_length, dim]
    return features.transpose(1, 2)


def get_features_layer_select(
    encoder,
    input_features: Tensor,
    output_length: Tensor,
    num_layers: int = 12,
    start_layer: int = 0,
    output_50hz: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Extract features using only a subset of transformer layers.

    Research shows using 12 of 32 layers achieves 0.22% EER vs ~10% with all layers.
    The hypothesis is that middle layers capture discriminative artifacts while
    early layers are too raw and later layers too semantic.

    Args:
        encoder: MiMo AudioEncoder instance
        input_features: Mel spectrogram [batch, n_mels, seq_len]
        output_length: Sequence lengths [batch] (at 50Hz resolution)
        num_layers: Number of transformer layers to use (default: 12)
        start_layer: First layer to use (default: 0)
        output_50hz: If True, skip final downsampling (50Hz output)

    Returns:
        hidden_states: Features [batch, seq_len, 1280]
        output_length: Updated output length
    """
    from mimo_audio_tokenizer.utils import (
        get_position_ids, packing, unpacking
    )

    # Validate layer selection
    total_layers = len(encoder.layers)
    end_layer = start_layer + num_layers
    if end_layer > total_layers:
        raise ValueError(
            f"Layer selection out of range: start={start_layer}, num={num_layers}, "
            f"but encoder only has {total_layers} layers"
        )

    # Conv layers - process mel spectrogram (same as original)
    input_features = input_features.to(encoder.conv1.weight)
    inputs_embeds = nn.functional.gelu(encoder.conv1(input_features))
    inputs_embeds = nn.functional.gelu(encoder.conv2(inputs_embeds))
    # Now at 50Hz: [batch, d_model, seq_len_50hz]

    # Transpose to [batch, seq_len_50hz, d_model]
    hidden_states = inputs_embeds.permute(0, 2, 1)

    # Get position embeddings (RoPE)
    position_ids = get_position_ids(output_length).long().to(input_features.device)
    rope_position_embeddings = encoder.position_embedding(input_features, position_ids)

    # Pack sequences for efficient transformer processing
    hidden_states = packing(hidden_states, output_length)
    skip_connect_hidden_states = 0.0

    # Process through SELECTED transformer layers only
    for idx in range(start_layer, end_layer):
        encoder_layer = encoder.layers[idx]
        hidden_states = encoder_layer(
            hidden_states,
            output_length,
            rope_position_embeddings=rope_position_embeddings,
        )
        # Handle skip connection if within selected range
        # Original MiMo uses skip_layer_idx=3 (layer index 2)
        if (encoder.skip_layer_idx is not None) and idx == encoder.skip_layer_idx - 1:
            skip_connect_hidden_states = hidden_states.clone()

    # Apply skip connection (only if skip layer was processed)
    if isinstance(skip_connect_hidden_states, Tensor):
        hidden_states = hidden_states + skip_connect_hidden_states

    # Apply layer normalization
    hidden_states = encoder.layer_norm(hidden_states)

    # Unpack back to batch format
    hidden_states = unpacking(hidden_states, output_length)
    # Shape: [batch, seq_len_50hz, d_model]

    # Optionally apply downsampling (50Hz -> 25Hz)
    if not output_50hz:
        if hidden_states.size(1) % encoder.config.avg_pooler:
            pad_len = encoder.config.avg_pooler - hidden_states.size(1) % encoder.config.avg_pooler
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, pad_len), mode='constant', value=0.
            )
        hidden_states = encoder.down_sample_layer(hidden_states.transpose(1, 2))
        output_length = output_length // encoder.config.avg_pooler + (
            output_length % encoder.config.avg_pooler != 0
        ).int()
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = encoder.down_sample_norm(hidden_states)

    return hidden_states, output_length


def patch_encoder_for_layer_select(tokenizer) -> None:
    """
    Patch the tokenizer's encoder to support layer selection.

    Adds a `get_features_layer_select` method to the encoder that extracts
    features using only a subset of transformer layers.

    Args:
        tokenizer: MiMoAudioTokenizer instance

    Usage:
        patch_encoder_for_layer_select(tokenizer)
        # Use 12 layers starting from layer 0
        features, lengths = tokenizer.encoder.get_features_layer_select(
            mels, output_length, num_layers=12, start_layer=0
        )
    """
    import types

    def _get_features_layer_select(
        self,
        input_features: Tensor,
        output_length: Tensor,
        num_layers: int = 12,
        start_layer: int = 0,
        output_50hz: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        return get_features_layer_select(
            self, input_features, output_length,
            num_layers=num_layers, start_layer=start_layer, output_50hz=output_50hz
        )

    tokenizer.encoder.get_features_layer_select = types.MethodType(
        _get_features_layer_select,
        tokenizer.encoder
    )
