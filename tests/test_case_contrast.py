import json
import wave
from pathlib import Path

import numpy as np

import mimodf.cli as cli
from mimodf.features.case_contrast import CaseContrastSettings, run_case_contrast
from mimodf.features.predictions import PredictionSource


def _write_wav(path: Path) -> None:
    samples = (np.sin(np.linspace(0, 10, 1600)) * 1000).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(samples.tobytes())


def _write_feature_cache(root: Path, ids: list[str]) -> None:
    arrays = root / "arrays"
    arrays.mkdir(parents=True)
    records = []
    for index, utterance_id in enumerate(ids):
        values = np.ones((2, 3), dtype=np.float32) * (index + 1)
        path = arrays / f"{utterance_id}.npz"
        np.savez_compressed(path, values=values)
        records.append(
            {"utterance_id": utterance_id, "array_path": str(path), "value_kind": "continuous"}
        )
    (root / "manifest.json").write_text(json.dumps({"schema": "mimodf-feature-manifest/v1"}))
    (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))


def _case(utterance_id: str, *, xlsr: bool, mimo: bool) -> dict[str, object]:
    return {
        "utterance_id": utterance_id,
        "target": "spoof",
        "duration_bucket": "short_lt3s",
        "systems": {
            "xlsr": {
                "correct": xlsr,
                "prediction": "spoof" if xlsr else "bonafide",
                "positive_probability": 0.4,
            },
            "mimo_cont": {
                "correct": mimo,
                "prediction": "spoof" if mimo else "bonafide",
                "positive_probability": 0.8,
            },
        },
        "audio": {"duration_sec": 1.0},
    }


def test_run_case_contrast(tmp_path):
    ids = ["a", "b", "c", "d"]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                _case("a", xlsr=False, mimo=True),
                _case("b", xlsr=True, mimo=False),
                _case("c", xlsr=True, mimo=True),
                _case("d", xlsr=False, mimo=False),
            ]
        )
    )
    audio = tmp_path / "a.wav"
    _write_wav(audio)
    protocol = tmp_path / "protocol.jsonl"
    protocol.write_text(
        "".join(
            json.dumps({"utterance_id": utterance_id, "audio_path": str(audio)}) + "\n"
            for utterance_id in ids
        )
    )
    features = tmp_path / "features"
    _write_feature_cache(features, ids)

    result = run_case_contrast(
        CaseContrastSettings(
            cases_path=cases_path,
            protocol=protocol,
            feature_sources=(PredictionSource("fake", features),),
            out_dir=tmp_path / "out",
            reference_system="xlsr",
            contrast_system="mimo_cont",
        )
    )

    summary = json.loads(result.summary_path.read_text())
    assert result.records == 4
    assert summary["groups"]["contrast_fixes_reference"]["records"] == 1
    assert summary["groups"]["reference_fixes_contrast"]["records"] == 1
    primary = summary["feature_contrasts"]["fake"][
        "contrast_fixes_reference__vs__reference_fixes_contrast"
    ]
    assert primary["left_count"] == 1
    assert primary["right_count"] == 1
    assert "CLAMTTS case-contrast" in result.report_path.read_text()


def test_case_contrast_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(settings):
        calls.append(settings)
        return type(
            "Result",
            (),
            {
                "to_dict": lambda self: {
                    "report": str(settings.out_dir / "report.md"),
                    "summary": str(settings.out_dir / "summary.json"),
                    "cases": str(settings.out_dir / "cases.jsonl"),
                    "records": 119,
                }
            },
        )()

    monkeypatch.setattr(cli, "run_case_contrast", fake_run)

    rc = cli.main(
        [
            "features",
            "case-contrast",
            "--cases",
            "cases.jsonl",
            "--protocol",
            "protocol.jsonl",
            "--features",
            "mimo=features/mimo",
            "xlsr=features/xlsr",
            "--out-dir",
            str(tmp_path / "out"),
            "--reference-system",
            "xlsr",
            "--contrast-system",
            "mimo_cont",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["records"] == 119
    assert calls[0].reference_system == "xlsr"
    assert calls[0].contrast_system == "mimo_cont"
    assert calls[0].feature_sources[0].name == "mimo"
    assert calls[0].overwrite is True
