import json
import wave
from pathlib import Path

import pytest

from mimodf.data.codecfake_loader import (
    CodecfakeAudioDataset,
    CodecfakeLoaderSettings,
    build_codecfake_loaders,
)
from mimodf.data.codecfake_splits import build_source_holdout_rows


def _write_wav(path: Path, frames: int = 8, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"\x00\x00" * frames)


def _write_protocol(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rows(tmp_path: Path) -> list[dict[str, object]]:
    audio = tmp_path / "audio"
    audio.mkdir()
    rows: list[dict[str, object]] = []
    for source in ["A", "B", "C", "D"]:
        for label in ["bonafide", "spoof"]:
            for index in range(2):
                wav = audio / f"{source}_{label}_{index}.wav"
                _write_wav(wav)
                rows.append(
                    {
                        "subset": "CoSG",
                        "utterance_id": wav.stem,
                        "audio_path": str(wav),
                        "label": label,
                        "source_model": source,
                    }
                )
    return rows


def test_codecfake_audio_dataset_pads_and_maps_label(tmp_path):
    pytest.importorskip("torch")
    wav = tmp_path / "x.wav"
    _write_wav(wav, frames=4)
    dataset = CodecfakeAudioDataset(
        (
            {
                "utterance_id": "x",
                "audio_path": str(wav),
                "label": "spoof",
                "source_model": "S",
            },
        ),
        sample_rate=16_000,
        cut=10,
    )

    waveform, target, metadata = dataset[0]

    assert tuple(waveform.shape) == (10,)
    assert int(target) == 1
    assert metadata["utterance_id"] == "x"


def test_build_codecfake_loaders_batches(tmp_path):
    pytest.importorskip("torch")
    protocol = tmp_path / "protocol.jsonl"
    _write_protocol(protocol, _rows(tmp_path))
    split_rows = build_source_holdout_rows(
        protocol=protocol,
        heldout_source="A",
        validation_policy="stratified-row",
        validation_fraction=0.25,
    )

    loaders = build_codecfake_loaders(
        split_rows,
        CodecfakeLoaderSettings(batch_size=2, eval_batch_size=3, num_workers=0, cut=12),
    )
    inputs, targets, metadata = next(iter(loaders.train_loader))

    assert tuple(inputs.shape) == (2, 12)
    assert tuple(targets.shape) == (2,)
    assert set(loaders.label_to_index) == {"bonafide", "spoof"}
    assert "utterance_id" in metadata
