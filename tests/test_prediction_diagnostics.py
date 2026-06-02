import json

import mimodf.cli as cli
from mimodf.features.diagnostics import PredictionDiagnosticSettings, run_prediction_diagnostics


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _prediction(utterance_id, target, prediction, source):
    return {
        "schema": "mimodf-feature-probe-prediction/v1",
        "utterance_id": utterance_id,
        "target": target,
        "label": target,
        "prediction": prediction,
        "probabilities": {"bonafide": 0.2, "spoof": 0.8},
        "source_model": source,
        "decoder_type": "Freq",
        "quantizer_type": "Mvq",
        "auxiliary_objective": "None",
    }


def _fixture(tmp_path):
    root = tmp_path / "predictions"
    protocol = tmp_path / "protocol.jsonl"
    rows = [
        {
            "utterance_id": "CLAMTTS_1",
            "source_model": "CLAMTTS",
            "label": "spoof",
            "audio_path": "missing.wav",
        },
        {
            "utterance_id": "CLAMTTS_2",
            "source_model": "CLAMTTS",
            "label": "bonafide",
            "audio_path": "missing.wav",
        },
        {
            "utterance_id": "NS2_1",
            "source_model": "NS2",
            "label": "spoof",
            "audio_path": "missing.wav",
        },
        {
            "utterance_id": "NS2_2",
            "source_model": "NS2",
            "label": "bonafide",
            "audio_path": "missing.wav",
        },
    ]
    _write_jsonl(protocol, rows)
    _write_jsonl(
        root / "CLAMTTS" / "ssl" / "predictions.jsonl",
        [
            _prediction("CLAMTTS_1", "spoof", "bonafide", "CLAMTTS"),
            _prediction("CLAMTTS_2", "bonafide", "bonafide", "CLAMTTS"),
        ],
    )
    _write_jsonl(
        root / "CLAMTTS" / "mimo" / "predictions.jsonl",
        [
            _prediction("CLAMTTS_1", "spoof", "spoof", "CLAMTTS"),
            _prediction("CLAMTTS_2", "bonafide", "bonafide", "CLAMTTS"),
        ],
    )
    _write_jsonl(
        root / "NS2" / "ssl" / "predictions.jsonl",
        [
            _prediction("NS2_1", "spoof", "spoof", "NS2"),
            _prediction("NS2_2", "bonafide", "bonafide", "NS2"),
        ],
    )
    _write_jsonl(
        root / "NS2" / "mimo" / "predictions.jsonl",
        [
            _prediction("NS2_1", "spoof", "bonafide", "NS2"),
            _prediction("NS2_2", "bonafide", "spoof", "NS2"),
        ],
    )
    return root, protocol


def test_prediction_diagnostics_writes_summary(tmp_path):
    root, protocol = _fixture(tmp_path)
    result = run_prediction_diagnostics(
        PredictionDiagnosticSettings(
            predictions_root=root,
            protocol=protocol,
            out_dir=tmp_path / "out",
            sources=("CLAMTTS", "NS2"),
            systems=("ssl", "mimo"),
        )
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.records == 4
    assert summary["source_summary"]["CLAMTTS"]["systems"]["mimo"]["wrong"] == 0
    assert summary["source_summary"]["NS2"]["systems"]["mimo"]["wrong"] == 2
    assert summary["contrast_summary"]["systems"]["mimo"]["best_source"] == "CLAMTTS"
    assert result.report_path.is_file()
    assert result.cases_path.is_file()


def test_prediction_diagnostics_cli(tmp_path, capsys):
    root, protocol = _fixture(tmp_path)

    rc = cli.main(
        [
            "features",
            "diagnose-predictions",
            "--predictions-root",
            str(root),
            "--protocol",
            str(protocol),
            "--out-dir",
            str(tmp_path / "out"),
            "--sources",
            "CLAMTTS",
            "NS2",
            "--systems",
            "ssl",
            "mimo",
            "--no-audio-metadata",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 4
    assert payload["sources"] == 2
