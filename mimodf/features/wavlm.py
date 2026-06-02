"""Bounded WavLM feature-extraction smoke path.

This module is intentionally smoke-only: it caps extracted items so adding the
next SSL frontend cannot accidentally become a full Wave 2 run. Full CoSG
extraction should get a separate, logged command after the smoke proves useful.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mimodf.features.common import (
    FEATURE_MANIFEST_SCHEMA,
    FEATURE_RECORD_SCHEMA,
    FeatureExtractionResult,
    batched,
    command_argv,
    git_revision,
    load_audio_protocol,
    safe_id,
)

MAX_SMOKE_ITEMS = 16


@dataclass(frozen=True)
class WavLMFeatureExtractionSettings:
    protocol: Path
    out_dir: Path
    model_id: str = "microsoft/wavlm-base-plus"
    revision: str = "b21194173c0af7e94822c1776d162e2659fd4761"
    component_id: str = "frontend:wavlm-base-plus/hf-b211941/v1"
    max_items: int | None = None
    batch_size: int = 1
    device: str = "cpu"
    sample_rate: int = 16000
    cache_dir: Path | None = None
    local_files_only: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class WavLMSmokeExtractionSettings:
    protocol: Path
    out_dir: Path
    model_id: str = "microsoft/wavlm-base-plus"
    revision: str = "b21194173c0af7e94822c1776d162e2659fd4761"
    component_id: str = "frontend:wavlm-base-plus/hf-b211941/v1"
    max_items: int = 8
    batch_size: int = 1
    device: str = "cpu"
    sample_rate: int = 16000
    cache_dir: Path | None = None
    local_files_only: bool = False
    overwrite: bool = False


def extract_wavlm_features(settings: WavLMFeatureExtractionSettings) -> FeatureExtractionResult:
    return _extract_wavlm_features(
        settings, max_items_cap=None, caveat_prefix="WavLM feature extraction"
    )


def extract_wavlm_smoke_features(settings: WavLMSmokeExtractionSettings) -> FeatureExtractionResult:
    return _extract_wavlm_features(
        settings,
        max_items_cap=MAX_SMOKE_ITEMS,
        caveat_prefix="bounded WavLM smoke only; not a full CoSG extraction",
    )


def _extract_wavlm_features(
    settings: WavLMFeatureExtractionSettings | WavLMSmokeExtractionSettings,
    *,
    max_items_cap: int | None,
    caveat_prefix: str,
) -> FeatureExtractionResult:
    _validate_settings(settings, max_items_cap=max_items_cap)
    records = load_audio_protocol(settings.protocol, max_items=settings.max_items)
    if not records:
        raise ValueError("protocol contains no usable audio_path rows")

    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = settings.out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    torch, torchaudio, auto_feature_extractor, auto_model = _import_runtime()
    loader_kwargs = _loader_kwargs(settings)
    feature_extractor = auto_feature_extractor.from_pretrained(settings.model_id, **loader_kwargs)
    model = (
        auto_model.from_pretrained(settings.model_id, **loader_kwargs).eval().to(settings.device)
    )

    started = time.time()
    feature_records: list[dict[str, object]] = []
    for batch in batched(records, settings.batch_size):
        waveforms = [
            _load_waveform(
                record["audio_path"], target_sample_rate=settings.sample_rate, torchaudio=torchaudio
            )
            for record in batch
        ]
        inputs = feature_extractor(
            waveforms,
            sampling_rate=settings.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs["input_values"].to(settings.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(settings.device)

        with torch.no_grad():
            if attention_mask is None:
                output = model(input_values=input_values)
            else:
                output = model(input_values=input_values, attention_mask=attention_mask)
            features_tensor = output.last_hidden_state.float().cpu()
            features = features_tensor.numpy()
            frame_lengths = (
                _frame_lengths(
                    model,
                    attention_mask,
                    features_tensor.shape[1],
                    batch_size=features_tensor.shape[0],
                    torch=torch,
                )
                .cpu()
                .numpy()
            )

        for index, record in enumerate(batch):
            utterance_id = str(record["utterance_id"])
            length = int(frame_lengths[index])
            array = features[index, :length]
            array_path = arrays_dir / f"{safe_id(utterance_id)}.npz"
            np.savez_compressed(array_path, values=array, length=np.array(length, dtype=np.int64))
            feature_records.append(_feature_record(record, utterance_id, array_path, array, length))

    records_path = settings.out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for record in feature_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    manifest = {
        "schema": FEATURE_MANIFEST_SCHEMA,
        "component_id": settings.component_id,
        "representation": "wavlm_last_hidden_state",
        "model_id": settings.model_id,
        "revision": settings.revision,
        "sample_rate": settings.sample_rate,
        "batch_size": settings.batch_size,
        "max_items": settings.max_items,
        "device": settings.device,
        "dtype": "float32",
        "protocol": str(settings.protocol),
        "output_dir": str(settings.out_dir),
        "records_path": str(records_path),
        "records": len(feature_records),
        "started_unix": started,
        "finished_unix": time.time(),
        "git_revision": git_revision(),
        "command_argv": command_argv(),
        "caveats": [
            caveat_prefix,
            "no classifier training or evaluation claim",
            "log a separate planned run before probes or downstream summaries",
        ],
    }
    manifest_path = settings.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return FeatureExtractionResult(
        manifest_path=manifest_path,
        records_path=records_path,
        records=len(feature_records),
        output_dir=settings.out_dir,
    )


def _validate_settings(
    settings: WavLMFeatureExtractionSettings | WavLMSmokeExtractionSettings,
    *,
    max_items_cap: int | None,
) -> None:
    if not settings.model_id.strip():
        raise ValueError("model_id must be non-empty")
    if not settings.revision.strip():
        raise ValueError("revision must be non-empty")
    if settings.max_items is not None and settings.max_items <= 0:
        raise ValueError("max_items must be positive")
    if max_items_cap is not None and (
        settings.max_items is None or settings.max_items > max_items_cap
    ):
        raise ValueError(
            f"max_items must be between 1 and {max_items_cap} for WavLM smoke extraction"
        )
    if settings.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if settings.max_items is not None and settings.batch_size > settings.max_items:
        raise ValueError("batch_size must not exceed max_items")
    if settings.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")


def _loader_kwargs(
    settings: WavLMFeatureExtractionSettings | WavLMSmokeExtractionSettings,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "revision": settings.revision,
        "local_files_only": settings.local_files_only,
    }
    if settings.cache_dir is not None:
        kwargs["cache_dir"] = str(settings.cache_dir)
    return kwargs


def _import_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torchaudio
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:  # pragma: no cover - optional runtime path
        raise RuntimeError("Torch, torchaudio, and transformers are required") from exc
    return torch, torchaudio, AutoFeatureExtractor, AutoModel


def _load_waveform(path: str, *, target_sample_rate: int, torchaudio: Any) -> np.ndarray:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if int(sample_rate) != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), target_sample_rate)
    return waveform.cpu().numpy().astype(np.float32, copy=False)


def _frame_lengths(
    model: Any, attention_mask: Any, max_length: int, *, batch_size: int, torch: Any
) -> Any:
    if attention_mask is None:
        return torch.full((batch_size,), max_length, dtype=torch.long)
    sample_lengths = attention_mask.sum(dim=1)
    if hasattr(model, "_get_feat_extract_output_lengths"):
        lengths = model._get_feat_extract_output_lengths(sample_lengths)
    else:
        # Defensive fallback for compatible HF audio models. This branch keeps
        # padded frames bounded even if a model lacks the helper.
        max_input = int(attention_mask.shape[1])
        lengths = torch.ceil(sample_lengths.float() * float(max_length) / float(max_input)).long()
    return lengths.clamp(min=0, max=max_length)


def _feature_record(
    source: dict[str, Any], utterance_id: str, array_path: Path, array: np.ndarray, length: int
) -> dict[str, object]:
    return {
        "schema": FEATURE_RECORD_SCHEMA,
        "utterance_id": utterance_id,
        "source_audio_path": source["audio_path"],
        "array_path": str(array_path),
        "representation": "wavlm_last_hidden_state",
        "value_kind": "continuous",
        "shape": list(array.shape),
        "length": length,
        "frame_rate_hz": 50,
        "selected_quantizers": None,
        "label": source.get("label"),
        "subset": source.get("subset"),
        "clip_id": source.get("clip_id"),
        "speaker_id": source.get("speaker_id"),
        "source_model": source.get("source_model"),
        "quantizer_type": source.get("quantizer_type"),
        "auxiliary_objective": source.get("auxiliary_objective"),
        "decoder_type": source.get("decoder_type"),
    }
