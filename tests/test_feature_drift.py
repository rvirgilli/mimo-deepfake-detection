import json
from pathlib import Path

import numpy as np

import mimodf.cli as cli
from mimodf.features.drift import PairedDriftSettings, summarize_paired_feature_drift


def _write_cache(root: Path, rows: dict[str, np.ndarray]) -> None:
    arrays = root / "arrays"
    arrays.mkdir(parents=True)
    records = []
    for utterance_id, values in rows.items():
        array_path = arrays / f"{utterance_id}.npz"
        np.savez_compressed(array_path, values=values.astype(np.float32))
        records.append(
            {
                "utterance_id": utterance_id,
                "array_path": str(array_path),
                "value_kind": "continuous",
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"schema": "mimodf-feature-manifest/v1", "representation": "fake"})
    )
    (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))


def test_summarize_paired_feature_drift(tmp_path):
    clean = tmp_path / "clean"
    transformed = tmp_path / "transformed"
    _write_cache(clean, {"u1": np.array([[1.0, 2.0], [3.0, 4.0]])})
    _write_cache(transformed, {"u1__noise": np.array([[2.0, 2.0], [4.0, 4.0]])})
    transform_records = tmp_path / "transform_records.jsonl"
    transform_records.write_text(
        json.dumps(
            {
                "utterance_id": "u1__noise",
                "original_utterance_id": "u1",
                "transform_id": "noise_snr20",
                "transform_family": "additive_noise",
                "label": "spoof",
                "source_model": "S",
            }
        )
        + "\n"
    )

    result = summarize_paired_feature_drift(
        PairedDriftSettings(
            clean_feature_dir=clean,
            transformed_feature_dir=transformed,
            transform_records=transform_records,
            out_json=tmp_path / "summary.json",
            out_report=tmp_path / "report.md",
        )
    )

    summary = json.loads(result.summary_path.read_text())
    assert result.pairs == 1
    assert summary["pairs"] == 1
    assert summary["by_transform"]["noise_snr20"]["count"] == 1
    assert summary["rows"][0]["mean_abs_delta"] > 0
    assert "Paired feature drift" in result.report_path.read_text()


def test_paired_drift_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_drift(settings):
        calls.append(settings)
        return type(
            "Result",
            (),
            {
                "to_dict": lambda self: {
                    "summary": str(settings.out_json),
                    "report": str(settings.out_report),
                    "pairs": 72,
                }
            },
        )()

    monkeypatch.setattr(cli, "summarize_paired_feature_drift", fake_drift)

    rc = cli.main(
        [
            "features",
            "paired-drift",
            "--clean-feature-dir",
            "clean",
            "--transformed-feature-dir",
            "transformed",
            "--transform-records",
            "records.jsonl",
            "--out-json",
            str(tmp_path / "summary.json"),
            "--out-report",
            str(tmp_path / "report.md"),
            "--pooling",
            "continuous_mean_std",
            "--overwrite",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pairs"] == 72
    assert calls[0].clean_feature_dir == Path("clean")
    assert calls[0].transformed_feature_dir == Path("transformed")
    assert calls[0].overwrite is True
