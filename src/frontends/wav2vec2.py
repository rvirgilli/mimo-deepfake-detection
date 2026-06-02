"""
Wav2Vec 2.0 XLSR Frontend for audio feature extraction.

This module wraps the wav2vec 2.0 XLSR model from fairseq to provide
a consistent interface for the AASIST backend.

Supports:
- Frozen feature extraction (default)
- Adapter fine-tuning (parameter-efficient, ~1.3M trainable params)

Requirements:
- fairseq (Python 3.10 environment)
- XLSR checkpoint: xlsr2_300m.pt

Based on SSLModel from SSL_Anti-spoofing/model.py by Hemlata Tak.
"""

import os
from typing import Optional, Dict, Any, Iterator

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseFrontend
from .mimo_finetune import AdapterLayer

# Guarded import for fairseq (requires Python 3.10)
try:
    import fairseq
    FAIRSEQ_AVAILABLE = True
except ImportError:
    FAIRSEQ_AVAILABLE = False
    fairseq = None


class Wav2Vec2Frontend(BaseFrontend):
    """
    Wav2Vec 2.0 XLSR frontend feature extractor.

    This frontend uses the pre-trained XLSR 300M model to extract
    1024-dimensional features at approximately 50Hz.

    Args:
        checkpoint_path: Path to xlsr2_300m.pt checkpoint
        freeze: Whether to freeze the model weights (default: True)
        device: Device to load the model on
        finetune_config: Fine-tuning configuration dict. Supported strategies:
            - None: frozen (default)
            - {"strategy": "adapter", "adapter": {"dim": 64, "dropout": 0.1,
               "layers": "last_n", "n_layers": 8}}

    Attributes:
        out_dim: 1024 (fixed for XLSR)
        sample_rate: 16000 Hz
    """

    _out_dim: int = 1024
    _sample_rate: int = 16000

    def __init__(
        self,
        checkpoint_path: str = "xlsr2_300m.pt",
        freeze: bool = True,
        device: Optional[str] = None,
        finetune_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if not FAIRSEQ_AVAILABLE:
            raise ImportError(
                "fairseq is required for Wav2Vec2Frontend. "
                "Please install it in a Python 3.10 environment: "
                "pip install fairseq"
            )

        # Resolve checkpoint path
        if not os.path.isabs(checkpoint_path):
            # Try relative to SSL_Anti-spoofing directory
            ssl_path = os.path.join(
                os.path.dirname(__file__),
                "../../SSL_Anti-spoofing",
                checkpoint_path
            )
            if os.path.exists(ssl_path):
                checkpoint_path = ssl_path

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"XLSR checkpoint not found: {checkpoint_path}. "
                "Download xlsr2_300m.pt and place it in SSL_Anti-spoofing/"
            )

        # Load the pre-trained XLSR model
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0]
        self._device = device

        # Apply fine-tuning strategy
        self.finetune_config = finetune_config
        self._adapter_hooks = []
        self.adapters = None

        strategy = finetune_config.get("strategy", "frozen") if finetune_config else "frozen"

        if strategy == "adapter":
            self._apply_adapters(finetune_config.get("adapter", {}))
        elif strategy == "frozen" or freeze:
            self.freeze()

    def _apply_adapters(self, adapter_cfg: Dict[str, Any]) -> None:
        """Inject adapter layers into wav2vec2 transformer layers."""
        # Freeze base model first
        for p in self.model.parameters():
            p.requires_grad = False

        adapter_dim = adapter_cfg.get("dim", 64)
        adapter_dropout = adapter_cfg.get("dropout", 0.1)
        adapter_layers_mode = adapter_cfg.get("layers", "last_n")
        n_adapter_layers = adapter_cfg.get("n_layers", 8)

        # fairseq wav2vec2: self.model.encoder.layers is a ModuleList
        encoder_layers = self.model.encoder.layers
        num_layers = len(encoder_layers)
        print(f"wav2vec2 encoder has {num_layers} transformer layers, dim={self._out_dim}")

        # Determine which layers get adapters
        if adapter_layers_mode == "all":
            adapter_indices = list(range(num_layers))
        elif adapter_layers_mode == "last_n":
            n = min(n_adapter_layers, num_layers)
            adapter_indices = list(range(num_layers - n, num_layers))
        elif adapter_layers_mode == "first_n":
            n = min(n_adapter_layers, num_layers)
            adapter_indices = list(range(n))
        else:
            raise ValueError(f"Unknown adapter_layers mode: {adapter_layers_mode}")

        print(f"Adding adapters to layers: {adapter_indices}")

        # Create adapter modules (reuses AdapterLayer from mimo_finetune)
        self.adapters = nn.ModuleDict({
            str(i): AdapterLayer(self._out_dim, adapter_dim, adapter_dropout)
            for i in adapter_indices
        })

        # Register forward hooks on transformer layers
        for i in adapter_indices:
            layer = encoder_layers[i]
            hook = layer.register_forward_hook(self._make_adapter_hook(i))
            self._adapter_hooks.append(hook)
            print(f"  Registered adapter hook on encoder.layers.{i}")

        total_params = sum(a.num_params() for a in self.adapters.values())
        print(f"Total adapter parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    def _make_adapter_hook(self, layer_idx: int):
        """Create a forward hook that applies the adapter after a transformer layer."""
        def hook(module, input, output):
            # fairseq TransformerSentenceEncoderLayer returns (x, layer_attn, _)
            if isinstance(output, tuple):
                hidden_states = output[0]
                adapted = self.adapters[str(layer_idx)](hidden_states)
                return (adapted,) + output[1:]
            else:
                return self.adapters[str(layer_idx)](output)
        return hook

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return only trainable (adapter) parameters."""
        if self.adapters is not None:
            return self.adapters.parameters()
        return iter([])

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def extract_feat(self, input_data: Tensor) -> Tensor:
        """
        Extract features from raw audio waveform.

        Args:
            input_data: Raw audio waveform at 16kHz
                Shape: (batch, samples) or (batch, samples, 1)

        Returns:
            features: Extracted feature tensor
                Shape: (batch, seq_len, 1024)
        """
        # Ensure model is on the same device and dtype as input
        if (next(self.model.parameters()).device != input_data.device or
            next(self.model.parameters()).dtype != input_data.dtype):
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        # Also move adapters to correct device/dtype
        if self.adapters is not None:
            adapter_param = next(self.adapters.parameters())
            if adapter_param.device != input_data.device or adapter_param.dtype != input_data.dtype:
                self.adapters.to(input_data.device, dtype=input_data.dtype)

        # Handle input shape: (batch, length) or (batch, length, 1)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # Extract features [batch, length, 1024]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']

        return emb
