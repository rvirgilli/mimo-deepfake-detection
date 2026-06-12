"""WavLM frontend wrapper for AASIST-style raw-audio training.

This module intentionally supports the frozen-backend diagnostic only. It wraps
Hugging Face WavLM and exposes the small BaseFrontend interface used by
``src.model.Model``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .base import BaseFrontend


class WavLMFrontend(BaseFrontend):
    """Frozen WavLM feature extractor.

    Args:
        model_id: Hugging Face model identifier.
        revision: Pinned model revision used by the existing feature cache.
        local_files_only: If true, do not attempt network downloads.
    """

    _out_dim = 768
    _sample_rate = 16_000

    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base-plus",
        *,
        revision: str = "b21194173c0af7e94822c1776d162e2659fd4761",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - dependency-specific
            raise RuntimeError("transformers is required for WavLMFrontend") from exc

        self.model_id = model_id
        self.revision = revision
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.freeze()
        self.model.eval()

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def train(self, mode: bool = True) -> "WavLMFrontend":
        """Keep the frozen WavLM module in eval mode during backend training."""

        super().train(mode)
        self.model.eval()
        return self

    def extract_feat(self, input_data: Tensor) -> Tensor:
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        if input_data.ndim != 2:
            raise ValueError(f"expected waveform tensor with shape (batch, samples), got {tuple(input_data.shape)}")
        # WavLM parameters are frozen. no_grad avoids retaining a huge frontend
        # graph while still allowing backend gradients from returned features.
        with torch.no_grad():
            output: Any = self.model(input_values=input_data)
            return output.last_hidden_state.float()
