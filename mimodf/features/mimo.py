"""MiMo feature extraction for small research-wave probes.

This module is intentionally narrow: it writes cached per-utterance arrays plus a
manifest. It does not train classifiers and it does not hide MiMo's batch-size
sensitivity. Batch size is recorded as protocol metadata.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

Representation = Literal["continuous_25hz", "continuous_50hz_native", "rvq_codes"]
QuantizerGroup = Literal["all", "early", "late"]


@dataclass(frozen=True)
class MimoFeatureExtractionSettings:
    protocol: Path
    out_dir: Path
    model_path: Path
    representation: Representation
    quantizer_group: QuantizerGroup = "all"
    max_items: int | None = None
    batch_size: int = 1
    device: str = "cpu"
    use_bfloat16: bool = True
    sample_rate: int = 24000
    overwrite: bool = False


def selected_quantizers(group: QuantizerGroup, *, num_quantizers: int = 20) -> list[int]:
    if num_quantizers <= 0:
        raise ValueError("num_quantizers must be positive")
    if group == "all":
        return list(range(num_quantizers))
    if group == "early":
        return list(range(min(2, num_quantizers)))
    if group == "late":
        if num_quantizers <= 2:
            return []
        return list(range(2, num_quantizers))
    raise ValueError(f"unsupported quantizer group: {group}")


def extract_mimo_features(settings: MimoFeatureExtractionSettings) -> FeatureExtractionResult:
    if settings.max_items is not None and settings.max_items <= 0:
        raise ValueError("max_items must be positive")
    if settings.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if settings.representation == "continuous_50hz_native":
        caveats = [
            "native 50Hz is a local non-official MiMo extraction path",
            "MiMo features are batch-size-sensitive; compare only within pinned batch-size protocol",
        ]
    else:
        caveats = [
            "MiMo features are batch-size-sensitive; compare only within pinned batch-size protocol",
        ]

    records = load_audio_protocol(settings.protocol, max_items=settings.max_items)
    if not records:
        raise ValueError("protocol contains no usable audio_path rows")

    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = settings.out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    torch, torchaudio, mimo_audio_tokenizer = _import_runtime()
    tokenizer = mimo_audio_tokenizer.load_model(str(settings.model_path))
    tokenizer.eval()
    tokenizer.to(settings.device)
    if settings.use_bfloat16:
        tokenizer = tokenizer.bfloat16()

    if settings.representation == "continuous_50hz_native":
        from src.frontends.mimo_native50hz import patch_encoder_for_50hz

        patch_encoder_for_50hz(tokenizer)

    started = time.time()
    feature_records: list[dict[str, object]] = []
    for batch in batched(records, settings.batch_size):
        waveforms = [
            _load_waveform(
                record["audio_path"],
                target_sample_rate=settings.sample_rate,
                torch=torch,
                torchaudio=torchaudio,
            )
            for record in batch
        ]
        mels_list = [
            mimo_audio_tokenizer.mel_spectrogram(wav.cpu(), tokenizer.config) for wav in waveforms
        ]
        mels, mels_lens = mimo_audio_tokenizer.padding(mels_list)
        mels = mels.to(settings.device)
        mels_lens = mels_lens.to(settings.device)
        if settings.use_bfloat16:
            mels = mels.bfloat16()
        output_length = tokenizer.encoder.get_output_length(mels_lens)

        with torch.no_grad():
            if settings.representation == "continuous_25hz":
                values, lengths = tokenizer.encoder.get_features(mels, output_length)
                values = values.float().cpu().numpy()
                lengths_np = lengths.cpu().numpy()
                value_kind = "continuous"
                quantizers: list[int] | None = None
                frame_rate_hz = 25
            elif settings.representation == "continuous_50hz_native":
                values, lengths = tokenizer.encoder.get_features_50hz(mels, output_length)
                values = values.float().cpu().numpy()
                lengths_np = lengths.cpu().numpy()
                value_kind = "continuous"
                quantizers = None
                frame_rate_hz = 50
            elif settings.representation == "rvq_codes":
                codes, lengths, _timing = tokenizer.encode(mels, mels_lens, n_q=None)
                codes = codes.cpu().numpy()
                lengths_np = lengths.cpu().numpy()
                all_count = int(codes.shape[-1])
                quantizers = selected_quantizers(settings.quantizer_group, num_quantizers=all_count)
                values = codes[:, :, quantizers].astype(np.int16, copy=False)
                value_kind = "rvq_codes"
                frame_rate_hz = 25
            else:  # pragma: no cover - argparse/type guards should prevent this
                raise ValueError(f"unsupported representation: {settings.representation}")

        for index, record in enumerate(batch):
            utterance_id = str(record["utterance_id"])
            safe_id_value = safe_id(utterance_id)
            length = int(lengths_np[index])
            array = values[index, :length]
            array_path = arrays_dir / f"{safe_id_value}.npz"
            np.savez_compressed(array_path, values=array, length=np.array(length, dtype=np.int64))
            feature_records.append(
                {
                    "schema": FEATURE_RECORD_SCHEMA,
                    "utterance_id": utterance_id,
                    "source_audio_path": record["audio_path"],
                    "array_path": str(array_path),
                    "representation": settings.representation,
                    "value_kind": value_kind,
                    "shape": list(array.shape),
                    "length": length,
                    "frame_rate_hz": frame_rate_hz,
                    "selected_quantizers": quantizers,
                    "label": record.get("label"),
                    "subset": record.get("subset"),
                    "clip_id": record.get("clip_id"),
                    "speaker_id": record.get("speaker_id"),
                    "source_model": record.get("source_model"),
                    "quantizer_type": record.get("quantizer_type"),
                    "auxiliary_objective": record.get("auxiliary_objective"),
                    "decoder_type": record.get("decoder_type"),
                }
            )

    records_path = settings.out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for record in feature_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    manifest = {
        "schema": FEATURE_MANIFEST_SCHEMA,
        "component_id": "frontend/mimo-audio-tokenizer",
        "representation": settings.representation,
        "quantizer_group": settings.quantizer_group
        if settings.representation == "rvq_codes"
        else None,
        "model_path": str(settings.model_path),
        "model_config": _model_config_summary(tokenizer.config),
        "sample_rate": settings.sample_rate,
        "batch_size": settings.batch_size,
        "device": settings.device,
        "use_bfloat16": settings.use_bfloat16,
        "protocol": str(settings.protocol),
        "output_dir": str(settings.out_dir),
        "records_path": str(records_path),
        "records": len(feature_records),
        "started_unix": started,
        "finished_unix": time.time(),
        "git_revision": git_revision(),
        "command_argv": command_argv(),
        "caveats": caveats,
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


def _import_runtime() -> tuple[Any, Any, Any]:
    try:
        import mimo_audio_tokenizer
        import torch
        import torchaudio
    except ImportError as exc:  # pragma: no cover - optional runtime path
        raise RuntimeError("Torch, torchaudio, and mimo_audio_tokenizer are required") from exc
    return torch, torchaudio, mimo_audio_tokenizer


def _load_waveform(path: str, *, target_sample_rate: int, torch: Any, torchaudio: Any) -> Any:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if int(sample_rate) != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), target_sample_rate)
    return waveform


def _model_config_summary(config: Any) -> dict[str, object]:
    fields = [
        "sampling_rate",
        "hop_length",
        "stride_size",
        "avg_pooler",
        "num_quantizers",
        "codebook_size",
        "d_model",
    ]
    return {field: getattr(config, field) for field in fields if hasattr(config, field)}
