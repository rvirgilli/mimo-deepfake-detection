"""Frozen wav2vec2/XLSR feature extraction for Wave 0 probes."""

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


@dataclass(frozen=True)
class Wav2Vec2FeatureExtractionSettings:
    protocol: Path
    out_dir: Path
    checkpoint: Path
    max_items: int | None = None
    batch_size: int = 1
    device: str = "cpu"
    sample_rate: int = 16000
    overwrite: bool = False


def extract_wav2vec2_features(
    settings: Wav2Vec2FeatureExtractionSettings,
) -> FeatureExtractionResult:
    if settings.max_items is not None and settings.max_items <= 0:
        raise ValueError("max_items must be positive")
    if settings.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not settings.checkpoint.is_file():
        raise FileNotFoundError(settings.checkpoint)

    records = load_audio_protocol(settings.protocol, max_items=settings.max_items)
    if not records:
        raise ValueError("protocol contains no usable audio_path rows")

    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = settings.out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    torch, torchaudio, fairseq = _import_runtime()
    model, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(settings.checkpoint)]
    )
    encoder = model[0].float().eval().to(settings.device)

    started = time.time()
    feature_records: list[dict[str, object]] = []
    for batch in batched(records, settings.batch_size):
        waveforms = [
            _load_waveform(
                record["audio_path"],
                target_sample_rate=settings.sample_rate,
                torchaudio=torchaudio,
            )
            for record in batch
        ]
        sample_lengths = torch.tensor(
            [int(wav.numel()) for wav in waveforms], dtype=torch.long, device=settings.device
        )
        padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).to(settings.device)

        with torch.no_grad():
            output = encoder(padded, mask=False, features_only=True)
            features = output["x"].float().cpu().numpy()
            frame_lengths = (
                _feature_lengths(encoder, sample_lengths, max_length=features.shape[1])
                .cpu()
                .numpy()
            )

        for index, record in enumerate(batch):
            utterance_id = str(record["utterance_id"])
            length = int(frame_lengths[index])
            array = features[index, :length]
            array_path = arrays_dir / f"{safe_id(utterance_id)}.npz"
            np.savez_compressed(array_path, values=array, length=np.array(length, dtype=np.int64))
            feature_records.append(
                {
                    "schema": FEATURE_RECORD_SCHEMA,
                    "utterance_id": utterance_id,
                    "source_audio_path": record["audio_path"],
                    "array_path": str(array_path),
                    "representation": "continuous_50hz",
                    "value_kind": "continuous",
                    "shape": list(array.shape),
                    "length": length,
                    "frame_rate_hz": 50,
                    "selected_quantizers": None,
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
        "component_id": "frontend:wav2vec2-xlsr-300m/v1",
        "representation": "continuous_50hz",
        "checkpoint": str(settings.checkpoint),
        "sample_rate": settings.sample_rate,
        "batch_size": settings.batch_size,
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
            "frozen SSL feature smoke only; no classifier training or evaluation claim",
            "compare only against features extracted from the same checkpoint and sample-rate protocol",
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


def _import_runtime() -> tuple[Any, Any, Any]:
    try:
        import fairseq
        import torch
        import torchaudio
    except ImportError as exc:  # pragma: no cover - optional runtime path
        raise RuntimeError("Torch, torchaudio, and fairseq are required") from exc
    return torch, torchaudio, fairseq


def _load_waveform(path: str, *, target_sample_rate: int, torchaudio: Any) -> Any:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if int(sample_rate) != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), target_sample_rate)
    return waveform


def _feature_lengths(encoder: Any, sample_lengths: Any, *, max_length: int) -> Any:
    if hasattr(encoder, "_get_feat_extract_output_lengths"):
        lengths = encoder._get_feat_extract_output_lengths(sample_lengths)
    else:
        # XLSR uses a 20 ms stride after convolutional subsampling. This branch
        # is only a defensive fallback for fairseq API drift.
        lengths = sample_lengths // 320
    return lengths.clamp(min=0, max=max_length)
