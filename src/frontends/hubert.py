"""
HuBERT Frontend for audio feature extraction.

Uses HuBERT-Large (facebook/hubert-large-ll60k) from HuggingFace transformers.
315M params, 1024-dim output at ~50Hz. Same architecture as wav2vec2 but
pretrained with masked prediction over discrete targets (k-means clusters),
making it a useful third comparison point alongside contrastive (wav2vec2)
and reconstruction (MiMo) pretraining objectives.

Supports:
- Frozen feature extraction (default)
- Adapter fine-tuning (parameter-efficient, reuses AdapterLayer from mimo_finetune)
"""

import os
from typing import Optional, Dict, Any, Iterator

import torch
import torch.nn as nn
from torch import Tensor
from transformers import HubertModel

from .base import BaseFrontend
from .mimo_finetune import AdapterLayer


class HuBERTFrontend(BaseFrontend):
    """
    HuBERT-Large frontend feature extractor.

    Args:
        model_name: HuggingFace model name (default: facebook/hubert-large-ll60k)
        freeze: Whether to freeze the model weights (default: True)
        finetune_config: Fine-tuning configuration dict. Supported strategies:
            - None: frozen (default)
            - {"strategy": "adapter", "adapter": {"dim": 64, "dropout": 0.1,
               "layers": "last_n", "n_layers": 8}}

    Attributes:
        out_dim: 1024 (fixed for HuBERT-Large)
        sample_rate: 16000 Hz
    """

    _out_dim: int = 1024
    _sample_rate: int = 16000

    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ll60k",
        freeze: bool = True,
        finetune_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.model = HubertModel.from_pretrained(model_name)
        self.finetune_config = finetune_config
        self._adapter_hooks = []
        self.adapters = None

        strategy = finetune_config.get("strategy", "frozen") if finetune_config else "frozen"

        if strategy == "adapter":
            self._apply_adapters(finetune_config.get("adapter", {}))
        elif strategy == "frozen" or freeze:
            self.freeze()

    def _apply_adapters(self, adapter_cfg: Dict[str, Any]) -> None:
        """Inject adapter layers into HuBERT transformer layers."""
        for p in self.model.parameters():
            p.requires_grad = False

        adapter_dim = adapter_cfg.get("dim", 64)
        adapter_dropout = adapter_cfg.get("dropout", 0.1)
        adapter_layers_mode = adapter_cfg.get("layers", "last_n")
        n_adapter_layers = adapter_cfg.get("n_layers", 8)

        # HuggingFace HuBERT: self.model.encoder.layers is a ModuleList
        encoder_layers = self.model.encoder.layers
        num_layers = len(encoder_layers)
        print(f"HuBERT encoder has {num_layers} transformer layers, dim={self._out_dim}")

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

        self.adapters = nn.ModuleDict({
            str(i): AdapterLayer(self._out_dim, adapter_dim, adapter_dropout)
            for i in adapter_indices
        })

        for i in adapter_indices:
            layer = encoder_layers[i]
            hook = layer.register_forward_hook(self._make_adapter_hook(i))
            self._adapter_hooks.append(hook)

        total_params = sum(a.num_params() for a in self.adapters.values())
        print(f"Total adapter parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    def _make_adapter_hook(self, layer_idx: int):
        """Create a forward hook that applies the adapter after a transformer layer."""
        def hook(module, input, output):
            # HuggingFace HubertEncoderLayer returns (hidden_states, attn_weights)
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
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        outputs = self.model(input_tmp)
        return outputs.last_hidden_state
