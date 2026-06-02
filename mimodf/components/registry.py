"""Stable component registry for versioned experiment specs.

The registry is metadata-first. Model factories stay in the existing legacy seams until the
experiment contracts are stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComponentMetadata:
    component_id: str
    api_version: str
    kind: str
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["caveats"] = list(self.caveats)
        return data


_REGISTRY: dict[str, ComponentMetadata] = {
    "frontend:wav2vec2-xlsr-300m/v1": ComponentMetadata(
        component_id="frontend:wav2vec2-xlsr-300m/v1",
        api_version="frontend/v1",
        kind="frontend",
        summary="wav2vec2 XLS-R 300M frame encoder",
        metadata={"sample_rate": 16000, "feature_dim": 1024, "frame_rate_hz": 50},
    ),
    "frontend:wavlm-base-plus/hf-b211941/v1": ComponentMetadata(
        component_id="frontend:wavlm-base-plus/hf-b211941/v1",
        api_version="frontend/v1",
        kind="frontend",
        summary="WavLM-Base+ Hugging Face frame encoder smoke candidate",
        metadata={
            "sample_rate": 16000,
            "feature_dim": 768,
            "frame_rate_hz": 50,
            "model_id": "microsoft/wavlm-base-plus",
            "revision": "b21194173c0af7e94822c1776d162e2659fd4761",
        },
        caveats=("wave2_full_source_holdout_completed",),
    ),
    "frontend:logmel-meanstd/v1": ComponentMetadata(
        component_id="frontend:logmel-meanstd/v1",
        api_version="frontend/v1",
        kind="frontend",
        summary="Boring log-mel acoustic baseline for mean/std feature probes",
        metadata={
            "sample_rate": 16000,
            "n_mels": 80,
            "frame_rate_hz": 100,
            "pooling": "mean_std",
        },
        caveats=("baseline_not_trainable_frontend",),
    ),
    "frontend:mimo-continuous-native50/v1": ComponentMetadata(
        component_id="frontend:mimo-continuous-native50/v1",
        api_version="frontend/v1",
        kind="frontend",
        summary="MiMo continuous pre-quantization features with native 50Hz extraction",
        metadata={
            "sample_rate": 24000,
            "feature_dim": 1280,
            "frame_rate_hz": 50,
            "precision": "bf16",
        },
        caveats=("bf16_flashattention_batch_size_sensitive",),
    ),
    "frontend:mimo-rvq-sum-25hz/v1": ComponentMetadata(
        component_id="frontend:mimo-rvq-sum-25hz/v1",
        api_version="frontend/v1",
        kind="frontend",
        summary="MiMo RVQ-sum 25Hz diagnostic feature variant",
        metadata={"sample_rate": 24000, "feature_dim": 1280, "frame_rate_hz": 25},
    ),
    "backend:aasist/v1": ComponentMetadata(
        component_id="backend:aasist/v1",
        api_version="backend/v1",
        kind="backend",
        summary="AASIST-style graph attention backend",
        metadata={"outputs": "binary_logits"},
    ),
    "backend:mlp-pool/v1": ComponentMetadata(
        component_id="backend:mlp-pool/v1",
        api_version="backend/v1",
        kind="backend",
        summary="Pooled MLP diagnostic backend",
        metadata={"outputs": "binary_logits"},
    ),
    "adaptation:frozen/v1": ComponentMetadata(
        component_id="adaptation:frozen/v1",
        api_version="adaptation/v1",
        kind="adaptation",
        summary="Frozen frontend; train projection/backend only",
        metadata={"trainable_scope": "projection_backend"},
    ),
    "adaptation:houlsby-adapter-last8/v1": ComponentMetadata(
        component_id="adaptation:houlsby-adapter-last8/v1",
        api_version="adaptation/v1",
        kind="adaptation",
        summary="Houlsby adapters in the last eight transformer layers",
        metadata={"trainable_scope": "last_8_layers_adapters"},
    ),
    "adaptation:full-finetune/v1": ComponentMetadata(
        component_id="adaptation:full-finetune/v1",
        api_version="adaptation/v1",
        kind="adaptation",
        summary="Full frontend fine-tuning",
        metadata={"trainable_scope": "all_frontend_parameters"},
    ),
    "optimizer:adam/v1": ComponentMetadata(
        component_id="optimizer:adam/v1",
        api_version="optimizer/v1",
        kind="optimizer",
        summary="Adam optimizer with one learning-rate group",
    ),
    "optimizer:adamw-param-groups/v1": ComponentMetadata(
        component_id="optimizer:adamw-param-groups/v1",
        api_version="optimizer/v1",
        kind="optimizer",
        summary="AdamW optimizer with explicit encoder/backend parameter groups",
    ),
    "dataset:asvspoof2019-la-train/v1": ComponentMetadata(
        component_id="dataset:asvspoof2019-la-train/v1",
        api_version="dataset/v1",
        kind="dataset",
        summary="ASVspoof 2019 logical access train partition",
    ),
    "dataset:asvspoof2019-la-dev/v1": ComponentMetadata(
        component_id="dataset:asvspoof2019-la-dev/v1",
        api_version="dataset/v1",
        kind="dataset",
        summary="ASVspoof 2019 logical access development partition",
    ),
    "dataset:asvspoof2021-la-eval/v1": ComponentMetadata(
        component_id="dataset:asvspoof2021-la-eval/v1",
        api_version="dataset/v1",
        kind="dataset",
        summary="ASVspoof 2021 logical access evaluation partition",
    ),
    "dataset:asvspoof2021-df-eval/v1": ComponentMetadata(
        component_id="dataset:asvspoof2021-df-eval/v1",
        api_version="dataset/v1",
        kind="dataset",
        summary="ASVspoof 2021 deepfake evaluation partition",
    ),
    "scorer:asvspoof2021-la-official/v1": ComponentMetadata(
        component_id="scorer:asvspoof2021-la-official/v1",
        api_version="scorer/v1",
        kind="scorer",
        summary="Official ASVspoof 2021 LA EER/min-tDCF scorer",
    ),
    "scorer:asvspoof-df-eer/v1": ComponentMetadata(
        component_id="scorer:asvspoof-df-eer/v1",
        api_version="scorer/v1",
        kind="scorer",
        summary="ASVspoof DF EER scorer",
    ),
}


def get_component(component_id: str) -> ComponentMetadata:
    try:
        return _REGISTRY[component_id]
    except KeyError as exc:
        raise ValueError(f"unknown component id: {component_id}") from exc


def component_exists(component_id: str) -> bool:
    return component_id in _REGISTRY


def list_components(*, kind: str | None = None) -> list[ComponentMetadata]:
    values = sorted(_REGISTRY.values(), key=lambda item: item.component_id)
    if kind is None:
        return values
    return [item for item in values if item.kind == kind]
