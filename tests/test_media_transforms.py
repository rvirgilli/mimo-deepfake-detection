import json
import math
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

import mimodf.cli as cli
from mimodf.transforms.media import (
    AddNoiseSettings,
    MediaTransformSettings,
    add_noise,
    generate_media_transform_smoke,
)


def _write_wav(path: Path, *, sample_rate: int = 16000, seconds: float = 0.1) -> None:
    samples = int(sample_rate * seconds)
    t = np.arange(samples) / sample_rate
    audio = 0.2 * np.sin(2.0 * math.pi * 440.0 * t)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def _write_protocol(path: Path, audio: Path) -> None:
    row = {
        "schema": "mimodf-protocol-record/v1",
        "utterance_id": "utt/1",
        "audio_path": str(audio),
        "label": "spoof",
        "source_model": "TESTSRC",
        "clip_id": "clip1",
        "caveats": [],
    }
    path.write_text(json.dumps(row) + "\n")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_add_noise_writes_wav(tmp_path):
    audio = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    _write_wav(audio)

    result = add_noise(AddNoiseSettings(input_path=audio, output_path=out, seed=7))

    assert result.output_path == out
    assert result.samples > 0
    assert out.is_file()
    with wave.open(str(out), "rb") as f:
        assert f.getframerate() == 16000
        assert f.getnchannels() == 1


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe unavailable"
)
def test_generate_media_transform_smoke_writes_manifest_and_protocol(tmp_path):
    audio = tmp_path / "in.wav"
    protocol = tmp_path / "protocol.jsonl"
    out_root = tmp_path / "media-smoke"
    _write_wav(audio)
    _write_protocol(protocol, audio)

    result = generate_media_transform_smoke(
        MediaTransformSettings(
            protocol=protocol,
            out_root=out_root,
            transforms=("noise_snr20",),
            seed=11,
        )
    )

    assert result.input_records == 1
    assert result.transformed_records == 1
    manifest = json.loads(result.manifest_path.read_text())
    records = [json.loads(line) for line in result.records_path.read_text().splitlines()]
    transformed = [json.loads(line) for line in result.protocol_path.read_text().splitlines()]
    assert manifest["transformed_records"] == 1
    assert records[0]["transform_id"] == "noise_snr20"
    assert records[0]["sha256_input"]
    assert records[0]["sha256_output"]
    assert transformed[0]["original_utterance_id"] == "utt/1"
    assert transformed[0]["label_policy"] == "inherited_for_stress_test_not_new_ground_truth"
    assert Path(transformed[0]["audio_path"]).is_file()


def test_media_smoke_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_generate(settings):
        calls.append(settings)
        return type(
            "Result",
            (),
            {
                "to_dict": lambda self: {
                    "manifest": "manifest.json",
                    "records_path": "records.jsonl",
                    "protocol_path": "transformed_protocol.jsonl",
                    "input_records": 1,
                    "transformed_records": 3,
                    "output_root": str(settings.out_root),
                }
            },
        )()

    monkeypatch.setattr(cli, "generate_media_transform_smoke", fake_generate)

    rc = cli.main(
        [
            "transforms",
            "media-smoke",
            "--protocol",
            "sample.jsonl",
            "--out-root",
            str(tmp_path / "out"),
            "--transforms",
            "noise_snr20",
            "--sample-rate",
            "16000",
            "--seed",
            "123",
            "--overwrite",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["transformed_records"] == 3
    assert calls[0].protocol == Path("sample.jsonl")
    assert calls[0].transforms == ("noise_snr20",)
    assert calls[0].seed == 123
    assert calls[0].overwrite is True


def test_add_noise_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_add_noise(settings):
        calls.append(settings)
        return type(
            "Result",
            (),
            {
                "to_dict": lambda self: {
                    "output_path": str(settings.output_path),
                    "sample_rate": settings.sample_rate,
                    "samples": 10,
                    "snr_db": settings.snr_db,
                }
            },
        )()

    monkeypatch.setattr(cli, "add_noise", fake_add_noise)

    rc = cli.main(
        [
            "transforms",
            "add-noise",
            "--input",
            "in.wav",
            "--output",
            str(tmp_path / "out.wav"),
            "--sample-rate",
            "8000",
            "--snr-db",
            "10",
            "--seed",
            "5",
            "--overwrite",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples"] == 10
    assert calls[0].input_path == Path("in.wav")
    assert calls[0].sample_rate == 8000
    assert calls[0].snr_db == 10
    assert calls[0].seed == 5
    assert calls[0].overwrite is True
