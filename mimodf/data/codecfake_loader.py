"""Raw-audio Dataset/DataLoader utilities for CodecFake+."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from mimodf.data.codecfake_splits import SourceHoldoutRows
from mimodf.training.seeding import seed_dataloader_worker, torch_generator

LABEL_TO_INDEX = {"bonafide": 0, "spoof": 1}


@dataclass(frozen=True)
class CodecfakeLoaderSettings:
    batch_size: int
    eval_batch_size: int
    num_workers: int = 0
    sample_rate: int = 16_000
    cut: int | None = 64_600
    drop_last_train: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class CodecfakeLoaders:
    train_loader: Any
    val_loader: Any
    test_loader: Any
    label_to_index: dict[str, int]


class CodecfakeAudioDataset:
    """Map CodecFake protocol rows to fixed-length waveform tensors and binary labels."""

    def __init__(self, rows: tuple[dict[str, Any], ...], *, sample_rate: int, cut: int | None):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if cut is not None and cut <= 0:
            raise ValueError("cut must be positive when set")
        if not rows:
            raise ValueError("rows must not be empty")
        self.rows = rows
        self.sample_rate = sample_rate
        self.cut = cut

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        torch = _import_torch()
        row = self.rows[index]
        audio_path = row.get("audio_path")
        if not isinstance(audio_path, str):
            raise ValueError(f"row missing audio_path: {row.get('utterance_id')}")
        waveform = _read_audio(Path(audio_path), sample_rate=self.sample_rate)
        waveform = _fix_length(waveform, self.cut)
        label = row.get("label")
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"unsupported label: {label!r}")
        target = torch.tensor(LABEL_TO_INDEX[str(label)], dtype=torch.long)
        metadata = {
            "utterance_id": row.get("utterance_id"),
            "source_model": row.get("source_model"),
            "label": label,
            "audio_path": audio_path,
        }
        return waveform, target, metadata


def build_codecfake_loaders(
    split_rows: SourceHoldoutRows, settings: CodecfakeLoaderSettings
) -> CodecfakeLoaders:
    torch = _import_torch()
    train_set = CodecfakeAudioDataset(
        split_rows.train_rows, sample_rate=settings.sample_rate, cut=settings.cut
    )
    val_set = CodecfakeAudioDataset(
        split_rows.validation_rows, sample_rate=settings.sample_rate, cut=settings.cut
    )
    test_set = CodecfakeAudioDataset(
        split_rows.test_rows, sample_rate=settings.sample_rate, cut=settings.cut
    )
    generator = torch_generator(settings.seed) if settings.seed is not None else None
    worker_init_fn = (
        partial(seed_dataloader_worker, int(settings.seed)) if settings.seed is not None else None
    )
    return CodecfakeLoaders(
        train_loader=torch.utils.data.DataLoader(
            train_set,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=settings.num_workers,
            drop_last=settings.drop_last_train,
            generator=generator,
            worker_init_fn=worker_init_fn,
        ),
        val_loader=torch.utils.data.DataLoader(
            val_set,
            batch_size=settings.eval_batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            worker_init_fn=worker_init_fn,
        ),
        test_loader=torch.utils.data.DataLoader(
            test_set,
            batch_size=settings.eval_batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            worker_init_fn=worker_init_fn,
        ),
        label_to_index=dict(LABEL_TO_INDEX),
    )


def _read_audio(path: Path, *, sample_rate: int) -> Any:
    torch = _import_torch()
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError("soundfile is required for CodecFake audio loading") from exc

    if not path.is_file():
        raise FileNotFoundError(path)
    samples, read_sample_rate = sf.read(path, always_2d=False, dtype="float32")
    tensor = torch.as_tensor(samples, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.mean(dim=1)
    if int(read_sample_rate) != sample_rate:
        tensor = _resample(tensor, orig_freq=int(read_sample_rate), new_freq=sample_rate)
    return tensor.contiguous()


def _resample(waveform: Any, *, orig_freq: int, new_freq: int) -> Any:
    try:
        import torchaudio
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError("torchaudio is required when CodecFake audio must be resampled") from exc
    return torchaudio.functional.resample(waveform, orig_freq=orig_freq, new_freq=new_freq)


def _fix_length(waveform: Any, cut: int | None) -> Any:
    if cut is None:
        return waveform
    torch = _import_torch()
    if waveform.numel() >= cut:
        return waveform[:cut]
    padded = torch.zeros(cut, dtype=waveform.dtype)
    padded[: waveform.numel()] = waveform
    return padded


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Torch is required for CodecFake loaders") from exc
    return torch
