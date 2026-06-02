import json
from pathlib import Path

import mimodf.cli as cli
from mimodf.features.mechanism import MechanismAnalysisSettings, run_mechanism_analysis
from mimodf.features.predictions import PredictionSource


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _pred(utterance_id: str, target: str, prediction: str, score: float) -> dict[str, object]:
    return {
        "utterance_id": utterance_id,
        "target": target,
        "prediction": prediction,
        "probabilities": {"spoof": score, "bonafide": 1.0 - score},
        "label": target,
        "source_model": "CLAMTTS",
        "decoder_type": "Freq",
        "quantizer_type": "Mvq",
        "auxiliary_objective": "None",
    }


def test_run_mechanism_analysis_summarizes_reference_contrasts(tmp_path):
    xlsr = tmp_path / "xlsr.jsonl"
    mimo = tmp_path / "mimo.jsonl"
    protocol = tmp_path / "protocol.jsonl"
    _write_predictions(
        xlsr,
        [
            _pred("a", "spoof", "bonafide", 0.4),
            _pred("b", "bonafide", "bonafide", 0.2),
        ],
    )
    _write_predictions(
        mimo,
        [
            _pred("a", "spoof", "spoof", 0.9),
            _pred("b", "bonafide", "spoof", 0.8),
        ],
    )
    protocol.write_text(
        json.dumps({"utterance_id": "a", "source_model": "CLAMTTS", "audio_path": None})
        + "\n"
        + json.dumps({"utterance_id": "b", "source_model": "CLAMTTS", "audio_path": None})
        + "\n"
    )

    result = run_mechanism_analysis(
        MechanismAnalysisSettings(
            predictions=(PredictionSource("xlsr", xlsr), PredictionSource("mimo", mimo)),
            protocol=protocol,
            out_dir=tmp_path / "out",
            reference="xlsr",
            source_model="CLAMTTS",
        )
    )

    summary = json.loads(result.summary_path.read_text())
    assert summary["records"] == 2
    assert summary["systems"]["xlsr"]["wrong"] == 1
    assert summary["systems"]["mimo"]["wrong"] == 1
    assert summary["reference_contrasts"]["mimo"]["system_correct_reference_wrong"] == 1
    assert summary["reference_contrasts"]["mimo"]["reference_correct_system_wrong"] == 1
    assert "CLAMTTS mechanism analysis" in result.report_path.read_text()


def test_mechanism_analysis_cli_passes_settings(monkeypatch, tmp_path, capsys):
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

    monkeypatch.setattr(cli, "run_mechanism_analysis", fake_run)

    rc = cli.main(
        [
            "features",
            "mechanism-analysis",
            "--predictions",
            "xlsr=xlsr.jsonl",
            "mimo=mimo.jsonl",
            "--protocol",
            "protocol.jsonl",
            "--out-dir",
            str(tmp_path / "out"),
            "--reference",
            "xlsr",
            "--source-model",
            "CLAMTTS",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["records"] == 119
    assert calls[0].reference == "xlsr"
    assert calls[0].source_model == "CLAMTTS"
    assert calls[0].predictions[0].name == "xlsr"
    assert calls[0].overwrite is True
