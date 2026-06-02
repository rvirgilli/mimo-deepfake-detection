import json
from pathlib import Path

import pytest

import mimodf.cli as cli
from mimodf.features.common import FeatureExtractionResult, load_audio_protocol
from mimodf.features.logmel import LogMelFeatureExtractionSettings, extract_logmel_features
from mimodf.features.mimo import selected_quantizers
from mimodf.features.wavlm import WavLMSmokeExtractionSettings, extract_wavlm_smoke_features


def test_selected_quantizers_groups():
    assert selected_quantizers("all", num_quantizers=5) == [0, 1, 2, 3, 4]
    assert selected_quantizers("early", num_quantizers=5) == [0, 1]
    assert selected_quantizers("late", num_quantizers=5) == [2, 3, 4]
    assert selected_quantizers("late", num_quantizers=2) == []


@pytest.mark.parametrize("count", [0, -1])
def test_selected_quantizers_rejects_invalid_count(count):
    with pytest.raises(ValueError, match="num_quantizers"):
        selected_quantizers("all", num_quantizers=count)


def test_load_audio_protocol_filters_missing_audio(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"not real wav")
    protocol = tmp_path / "protocol.jsonl"
    rows = [
        {"utterance_id": "x", "audio_path": str(audio), "label": "spoof"},
        {"utterance_id": "missing", "audio_path": str(tmp_path / "missing.wav"), "label": "spoof"},
        {"utterance_id": "labels-only", "audio_path": None, "label": "spoof"},
    ]
    protocol.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = load_audio_protocol(protocol)

    assert [record["utterance_id"] for record in loaded] == ["x"]


def test_load_audio_protocol_respects_max_items(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.wav"
        path.write_bytes(b"x")
        paths.append(path)
    protocol = tmp_path / "protocol.jsonl"
    protocol.write_text(
        "".join(
            json.dumps({"utterance_id": str(index), "audio_path": str(path)}) + "\n"
            for index, path in enumerate(paths)
        )
    )

    loaded = load_audio_protocol(protocol, max_items=2)

    assert [record["utterance_id"] for record in loaded] == ["0", "1"]


def test_mimo_extract_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_extract(settings):
        calls.append(settings)
        return FeatureExtractionResult(
            manifest_path=Path("manifest.json"),
            records_path=Path("records.jsonl"),
            records=2,
            output_dir=settings.out_dir,
        )

    monkeypatch.setattr(cli, "extract_mimo_features", fake_extract)

    rc = cli.main(
        [
            "features",
            "mimo-extract",
            "--protocol",
            "protocol.jsonl",
            "--out-dir",
            str(tmp_path / "features"),
            "--model-path",
            "model_weights",
            "--representation",
            "rvq_codes",
            "--quantizer-group",
            "late",
            "--max-items",
            "2",
            "--batch-size",
            "1",
            "--device",
            "cuda",
            "--no-bfloat16",
            "--overwrite",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 2
    assert calls[0].representation == "rvq_codes"
    assert calls[0].quantizer_group == "late"
    assert calls[0].use_bfloat16 is False
    assert calls[0].overwrite is True


def test_wavlm_smoke_rejects_unbounded_items(tmp_path):
    with pytest.raises(ValueError, match="max_items"):
        extract_wavlm_smoke_features(
            WavLMSmokeExtractionSettings(
                protocol=tmp_path / "protocol.jsonl",
                out_dir=tmp_path / "features",
                max_items=17,
            )
        )


def test_wav2vec2_extract_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_extract(settings):
        calls.append(settings)
        return FeatureExtractionResult(
            manifest_path=Path("manifest.json"),
            records_path=Path("records.jsonl"),
            records=2,
            output_dir=settings.out_dir,
        )

    monkeypatch.setattr(cli, "extract_wav2vec2_features", fake_extract)

    rc = cli.main(
        [
            "features",
            "wav2vec2-extract",
            "--protocol",
            "protocol.jsonl",
            "--out-dir",
            str(tmp_path / "features"),
            "--checkpoint",
            "xlsr2_300m.pt",
            "--max-items",
            "2",
            "--batch-size",
            "1",
            "--device",
            "cuda",
            "--overwrite",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 2
    assert calls[0].checkpoint == Path("xlsr2_300m.pt")
    assert calls[0].sample_rate == 16000
    assert calls[0].overwrite is True


def test_wavlm_extract_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_extract(settings):
        calls.append(settings)
        return FeatureExtractionResult(
            manifest_path=Path("manifest.json"),
            records_path=Path("records.jsonl"),
            records=1797,
            output_dir=settings.out_dir,
        )

    monkeypatch.setattr(cli, "extract_wavlm_features", fake_extract)

    rc = cli.main(
        [
            "features",
            "wavlm-extract",
            "--protocol",
            "protocol.jsonl",
            "--out-dir",
            str(tmp_path / "wavlm-full"),
            "--max-items",
            "1797",
            "--batch-size",
            "1",
            "--device",
            "cuda",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 1797
    assert calls[0].model_id == "microsoft/wavlm-base-plus"
    assert calls[0].max_items == 1797
    assert calls[0].batch_size == 1
    assert calls[0].overwrite is False


def test_logmel_extract_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_extract(settings):
        calls.append(settings)
        return FeatureExtractionResult(
            manifest_path=Path("manifest.json"),
            records_path=Path("records.jsonl"),
            records=3,
            output_dir=settings.out_dir,
        )

    monkeypatch.setattr(cli, "extract_logmel_features", fake_extract)

    rc = cli.main(
        [
            "features",
            "logmel-extract",
            "--protocol",
            "protocol.jsonl",
            "--out-dir",
            str(tmp_path / "logmel"),
            "--max-items",
            "3",
            "--sample-rate",
            "16000",
            "--n-mels",
            "64",
            "--overwrite",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 3
    assert calls[0].out_dir == tmp_path / "logmel"
    assert calls[0].max_items == 3
    assert calls[0].sample_rate == 16000
    assert calls[0].n_mels == 64
    assert calls[0].overwrite is True


def test_logmel_extract_rejects_invalid_max_items(tmp_path):
    with pytest.raises(ValueError, match="max_items"):
        extract_logmel_features(
            LogMelFeatureExtractionSettings(
                protocol=tmp_path / "protocol.jsonl",
                out_dir=tmp_path / "features",
                max_items=0,
            )
        )


def test_wavlm_smoke_extract_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_extract(settings):
        calls.append(settings)
        return FeatureExtractionResult(
            manifest_path=Path("manifest.json"),
            records_path=Path("records.jsonl"),
            records=2,
            output_dir=settings.out_dir,
        )

    monkeypatch.setattr(cli, "extract_wavlm_smoke_features", fake_extract)

    rc = cli.main(
        [
            "features",
            "wavlm-smoke-extract",
            "--protocol",
            "protocol.jsonl",
            "--out-dir",
            str(tmp_path / "wavlm-smoke"),
            "--model-id",
            "microsoft/wavlm-base-plus",
            "--revision",
            "b21194173c0af7e94822c1776d162e2659fd4761",
            "--max-items",
            "8",
            "--batch-size",
            "2",
            "--device",
            "cuda",
            "--local-files-only",
            "--overwrite",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 2
    assert calls[0].model_id == "microsoft/wavlm-base-plus"
    assert calls[0].revision == "b21194173c0af7e94822c1776d162e2659fd4761"
    assert calls[0].max_items == 8
    assert calls[0].batch_size == 2
    assert calls[0].local_files_only is True
    assert calls[0].overwrite is True
