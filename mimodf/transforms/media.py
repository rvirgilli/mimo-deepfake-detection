"""Deterministic tiny media transforms for robustness smokes.

This module is intentionally narrow. It generates auditable transformed WAV
files for a small protocol sample and writes enough provenance to reproduce the
smoke. It is not a dataset builder.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mimodf.features.common import command_argv, git_revision, safe_id

TRANSFORM_MANIFEST_SCHEMA = "mimodf-media-transform-manifest/v1"
TRANSFORM_RECORD_SCHEMA = "mimodf-media-transform-record/v1"
TRANSFORMED_PROTOCOL_SCHEMA = "mimodf-transformed-protocol-record/v1"
DEFAULT_TRANSFORMS = ("resample_8k_16k", "mp3_64k_16k", "noise_snr20")


@dataclass(frozen=True)
class AddNoiseSettings:
    input_path: Path
    output_path: Path
    sample_rate: int = 16000
    snr_db: float = 20.0
    seed: int = 42
    overwrite: bool = False


@dataclass(frozen=True)
class MediaTransformSettings:
    protocol: Path
    out_root: Path
    transforms: tuple[str, ...] = DEFAULT_TRANSFORMS
    sample_rate: int = 16000
    seed: int = 42
    overwrite: bool = False


@dataclass(frozen=True)
class MediaTransformResult:
    manifest_path: Path
    records_path: Path
    protocol_path: Path
    input_records: int
    transformed_records: int
    output_root: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": str(self.manifest_path),
            "records_path": str(self.records_path),
            "protocol_path": str(self.protocol_path),
            "input_records": self.input_records,
            "transformed_records": self.transformed_records,
            "output_root": str(self.output_root),
        }


@dataclass(frozen=True)
class NoiseResult:
    output_path: Path
    sample_rate: int
    samples: int
    snr_db: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"output_path": str(self.output_path)}


def add_noise(settings: AddNoiseSettings) -> NoiseResult:
    _validate_noise_settings(settings)
    if settings.output_path.exists() and not settings.overwrite:
        raise FileExistsError(settings.output_path)
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = _decode_audio_float32(settings.input_path, sample_rate=settings.sample_rate)
    noisy = _add_noise_array(audio, snr_db=settings.snr_db, seed=settings.seed)
    _encode_wav_float32(noisy, settings.output_path, sample_rate=settings.sample_rate)
    return NoiseResult(
        output_path=settings.output_path,
        sample_rate=settings.sample_rate,
        samples=int(noisy.size),
        snr_db=settings.snr_db,
    )


def generate_media_transform_smoke(settings: MediaTransformSettings) -> MediaTransformResult:
    _validate_transform_settings(settings)
    if (
        settings.out_root.exists()
        and _has_transform_outputs(settings.out_root)
        and not settings.overwrite
    ):
        raise FileExistsError(settings.out_root)
    settings.out_root.mkdir(parents=True, exist_ok=True)
    audio_root = settings.out_root / "audio"
    tmp_root = settings.out_root / "tmp"
    audio_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    rows = _load_protocol(settings.protocol)
    started = time.time()
    ffmpeg_version = _tool_version("ffmpeg")
    ffprobe_version = _tool_version("ffprobe")

    transformed_protocol: list[dict[str, object]] = []
    transform_records: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        source_audio = Path(str(row["audio_path"]))
        if not source_audio.is_file():
            raise FileNotFoundError(source_audio)
        source_probe = _probe_audio(source_audio)
        input_hash = _sha256_file(source_audio)
        source_id = str(row["utterance_id"])
        for transform_id in settings.transforms:
            output_path, command = _run_transform(
                transform_id,
                source_audio,
                source_id=source_id,
                row_index=row_index,
                audio_root=audio_root,
                tmp_root=tmp_root,
                sample_rate=settings.sample_rate,
                seed=settings.seed,
                overwrite=settings.overwrite,
            )
            output_probe = _probe_audio(output_path)
            output_hash = _sha256_file(output_path)
            transformed_id = f"{source_id}__{transform_id}"
            transformed_row = dict(row)
            transformed_row.update(
                {
                    "schema": TRANSFORMED_PROTOCOL_SCHEMA,
                    "utterance_id": transformed_id,
                    "original_utterance_id": source_id,
                    "audio_path": str(output_path),
                    "original_audio_path": str(source_audio),
                    "original_label": row.get("label"),
                    "label": row.get("label"),
                    "media_transform_id": transform_id,
                    "media_transform_family": _transform_family(transform_id),
                    "label_policy": "inherited_for_stress_test_not_new_ground_truth",
                    "caveats": _extend_caveats(row),
                }
            )
            transformed_protocol.append(transformed_row)
            transform_records.append(
                {
                    "schema": TRANSFORM_RECORD_SCHEMA,
                    "transform_id": transform_id,
                    "transform_family": _transform_family(transform_id),
                    "utterance_id": transformed_id,
                    "original_utterance_id": source_id,
                    "input_path": str(source_audio),
                    "output_path": str(output_path),
                    "command": command,
                    "tool": "ffmpeg" if transform_id != "noise_snr20" else "mimodf+ffmpeg",
                    "tool_version": ffmpeg_version,
                    "sha256_input": input_hash,
                    "sha256_output": output_hash,
                    "input_audio": source_probe,
                    "output_audio": output_probe,
                    "label": row.get("label"),
                    "source_model": row.get("source_model"),
                    "label_policy": "inherited_for_stress_test_not_new_ground_truth",
                }
            )

    protocol_path = settings.out_root / "transformed_protocol.jsonl"
    records_path = settings.out_root / "transform_records.jsonl"
    _write_jsonl(protocol_path, transformed_protocol)
    _write_jsonl(records_path, transform_records)

    manifest = {
        "schema": TRANSFORM_MANIFEST_SCHEMA,
        "protocol": str(settings.protocol),
        "output_root": str(settings.out_root),
        "transformed_protocol": str(protocol_path),
        "transform_records": str(records_path),
        "input_records": len(rows),
        "transforms": list(settings.transforms),
        "transformed_records": len(transformed_protocol),
        "sample_rate": settings.sample_rate,
        "seed": settings.seed,
        "ffmpeg_path": shutil.which("ffmpeg"),
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_path": shutil.which("ffprobe"),
        "ffprobe_version": ffprobe_version,
        "started_unix": started,
        "finished_unix": time.time(),
        "git_revision": git_revision(),
        "command_argv": command_argv(),
        "caveats": [
            "tiny media-transform smoke only; not robustness metric evidence",
            "labels are inherited for stress-test bookkeeping, not new ground truth",
            "outputs are local /tmp-style generated artifacts and should not be committed",
        ],
    }
    manifest_path = settings.out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return MediaTransformResult(
        manifest_path=manifest_path,
        records_path=records_path,
        protocol_path=protocol_path,
        input_records=len(rows),
        transformed_records=len(transformed_protocol),
        output_root=settings.out_root,
    )


def _has_transform_outputs(out_root: Path) -> bool:
    protected_files = (
        out_root / "manifest.json",
        out_root / "transform_records.jsonl",
        out_root / "transformed_protocol.jsonl",
    )
    if any(path.exists() for path in protected_files):
        return True
    audio_root = out_root / "audio"
    return audio_root.exists() and any(audio_root.iterdir())


def _validate_noise_settings(settings: AddNoiseSettings) -> None:
    if not settings.input_path.is_file():
        raise FileNotFoundError(settings.input_path)
    if settings.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(settings.snr_db):
        raise ValueError("snr_db must be finite")


def _validate_transform_settings(settings: MediaTransformSettings) -> None:
    if not settings.protocol.is_file():
        raise FileNotFoundError(settings.protocol)
    if not settings.transforms:
        raise ValueError("at least one transform is required")
    unknown = [item for item in settings.transforms if item not in DEFAULT_TRANSFORMS]
    if unknown:
        raise ValueError(f"unsupported transforms: {', '.join(unknown)}")
    if settings.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required")
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required")


def _load_protocol(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            if not row.get("utterance_id") or not row.get("audio_path"):
                raise ValueError(f"{path}:{line_number}: utterance_id and audio_path are required")
            rows.append(row)
    if not rows:
        raise ValueError("protocol contains no rows")
    return rows


def _run_transform(
    transform_id: str,
    input_path: Path,
    *,
    source_id: str,
    row_index: int,
    audio_root: Path,
    tmp_root: Path,
    sample_rate: int,
    seed: int,
    overwrite: bool,
) -> tuple[Path, str]:
    transform_dir = audio_root / transform_id
    transform_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{row_index:04d}_{safe_id(source_id)}"
    output_path = transform_dir / f"{stem}.{transform_id}.wav"
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    if transform_id == "resample_8k_16k":
        temp = tmp_root / f"{stem}.8k.wav"
        command = (
            f"ffmpeg -nostdin -hide_banner -loglevel error -y -i {input_path} -ac 1 -ar 8000 "
            f"{temp} && ffmpeg -nostdin -hide_banner -loglevel error -y -i {temp} "
            f"-ac 1 -ar {sample_rate} {output_path}"
        )
        _run_ffmpeg(["-i", str(input_path), "-ac", "1", "-ar", "8000", str(temp)])
        _run_ffmpeg(["-i", str(temp), "-ac", "1", "-ar", str(sample_rate), str(output_path)])
        return output_path, command

    if transform_id == "mp3_64k_16k":
        temp = tmp_root / f"{stem}.mp3"
        command = (
            f"ffmpeg -nostdin -hide_banner -loglevel error -y -i {input_path} -ac 1 "
            f"-ar {sample_rate} -codec:a libmp3lame -b:a 64k {temp} && ffmpeg "
            f"-nostdin -hide_banner -loglevel error -y -i {temp} -ac 1 -ar {sample_rate} "
            f"{output_path}"
        )
        _run_ffmpeg(
            [
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "64k",
                str(temp),
            ]
        )
        _run_ffmpeg(["-i", str(temp), "-ac", "1", "-ar", str(sample_rate), str(output_path)])
        return output_path, command

    if transform_id == "noise_snr20":
        command = (
            f"python -m mimodf transforms add-noise --input {input_path} --output {output_path} "
            f"--sample-rate {sample_rate} --snr-db 20 --seed {seed + row_index}"
        )
        add_noise(
            AddNoiseSettings(
                input_path=input_path,
                output_path=output_path,
                sample_rate=sample_rate,
                snr_db=20.0,
                seed=seed + row_index,
                overwrite=overwrite,
            )
        )
        return output_path, command

    raise ValueError(f"unsupported transform: {transform_id}")


def _run_ffmpeg(args: list[str]) -> None:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args]
    subprocess.run(command, check=True, text=True, capture_output=True)


def _decode_audio_float32(path: Path, *, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise ValueError(f"empty decoded audio: {path}")
    return audio


def _encode_wav_float32(audio: np.ndarray, path: Path, *, sample_rate: int) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "-",
        "-acodec",
        "pcm_s16le",
        str(path),
    ]
    subprocess.run(
        command, input=audio.astype(np.float32).tobytes(), check=True, capture_output=True
    )


def _add_noise_array(audio: np.ndarray, *, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal_rms = float(np.sqrt(np.mean(np.square(audio))))
    if signal_rms == 0.0:
        signal_rms = 1e-6
    noise = rng.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noisy = audio + noise * (target_noise_rms / max(noise_rms, 1e-12))
    return np.clip(noisy, -1.0, 1.0).astype(np.float32, copy=False)


def _probe_audio(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(command, check=True, text=True, capture_output=True)
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"no audio stream: {path}")
    stream = streams[0]
    duration = stream.get("duration")
    return {
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "duration_sec": None if duration in {None, "N/A"} else float(duration),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version(tool: str) -> str:
    try:
        proc = subprocess.run([tool, "-version"], check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return proc.stdout.splitlines()[0] if proc.stdout else "unknown"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _extend_caveats(row: dict[str, Any]) -> list[str]:
    caveats = row.get("caveats")
    values = list(caveats) if isinstance(caveats, list) else []
    values.append("media-transformed smoke row; inherited label is stress-test metadata only")
    return values


def _transform_family(transform_id: str) -> str:
    if transform_id.startswith("resample"):
        return "resampling"
    if transform_id.startswith("mp3"):
        return "lossy_codec"
    if transform_id.startswith("noise"):
        return "additive_noise"
    return "unknown"
