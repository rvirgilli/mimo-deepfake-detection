"""Boring log-mel baseline feature extraction.

The extractor intentionally produces simple frame-level log-mel values. Probes
pool them with the same mean/std code path used for SSL and MiMo continuous
features, so this is a sanity-check frontend rather than another model.
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
    command_argv,
    git_revision,
    load_audio_protocol,
    safe_id,
)


@dataclass(frozen=True)
class LogMelFeatureExtractionSettings:
    protocol: Path
    out_dir: Path
    max_items: int | None = None
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    fmin: float = 20.0
    fmax: float | None = 7600.0
    overwrite: bool = False


def extract_logmel_features(
    settings: LogMelFeatureExtractionSettings,
) -> FeatureExtractionResult:
    _validate_settings(settings)
    records = load_audio_protocol(settings.protocol, max_items=settings.max_items)
    if not records:
        raise ValueError("protocol contains no usable audio_path rows")

    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = settings.out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    librosa = _import_runtime()
    started = time.time()
    feature_records: list[dict[str, object]] = []
    for record in records:
        utterance_id = str(record["utterance_id"])
        values = _logmel_array(record["audio_path"], settings=settings, librosa=librosa)
        array_path = arrays_dir / f"{safe_id(utterance_id)}.npz"
        np.savez_compressed(
            array_path,
            values=values,
            length=np.array(values.shape[0], dtype=np.int64),
        )
        feature_records.append(
            {
                "schema": FEATURE_RECORD_SCHEMA,
                "utterance_id": utterance_id,
                "source_audio_path": record["audio_path"],
                "array_path": str(array_path),
                "representation": "logmel_80_100hz",
                "value_kind": "continuous",
                "shape": list(values.shape),
                "length": int(values.shape[0]),
                "frame_rate_hz": settings.sample_rate / settings.hop_length,
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
        "component_id": "frontend:logmel-meanstd/v1",
        "representation": "logmel_80_100hz",
        "sample_rate": settings.sample_rate,
        "n_mels": settings.n_mels,
        "n_fft": settings.n_fft,
        "hop_length": settings.hop_length,
        "win_length": settings.win_length,
        "fmin": settings.fmin,
        "fmax": settings.fmax,
        "frame_rate_hz": settings.sample_rate / settings.hop_length,
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
            "boring acoustic baseline for feature-only probes; not a trained frontend",
            "intended for train-only standardized mean/std pooling in downstream probes",
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


def _validate_settings(settings: LogMelFeatureExtractionSettings) -> None:
    if settings.max_items is not None and settings.max_items <= 0:
        raise ValueError("max_items must be positive")
    if settings.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if settings.n_mels <= 0:
        raise ValueError("n_mels must be positive")
    if settings.n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if settings.hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if settings.win_length <= 0:
        raise ValueError("win_length must be positive")
    if settings.win_length > settings.n_fft:
        raise ValueError("win_length must be <= n_fft")


def _import_runtime() -> Any:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - optional runtime path
        raise RuntimeError("librosa and soundfile are required for log-mel extraction") from exc
    return librosa


def _logmel_array(
    path: str, *, settings: LogMelFeatureExtractionSettings, librosa: Any
) -> np.ndarray:
    waveform, sample_rate = librosa.load(path, sr=settings.sample_rate, mono=True)
    if waveform.size == 0:
        raise ValueError(f"empty audio: {path}")
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        win_length=settings.win_length,
        n_mels=settings.n_mels,
        fmin=settings.fmin,
        fmax=settings.fmax,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).T
    return np.asarray(logmel, dtype=np.float32)
