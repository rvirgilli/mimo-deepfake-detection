import json
from pathlib import Path

import numpy as np

import mimodf.cli as cli
from mimodf.features.fusion import ProbeFusionSettings, run_probe_fusion
from mimodf.features.predictions import (
    PredictionComparisonSettings,
    PredictionSource,
    compare_predictions,
    parse_prediction_source,
)
from mimodf.features.probe import ProbeSettings, pool_feature_record, run_feature_probe


def _write_continuous_cache(root: Path) -> None:
    arrays = root / "arrays"
    arrays.mkdir(parents=True)
    manifest = {
        "schema": "mimodf-feature-manifest/v1",
        "component_id": "fixture/frontend",
        "representation": "continuous_50hz",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    for index in range(12):
        label = "spoof" if index % 2 else "bonafide"
        values = np.full((3, 2), index, dtype=np.float32)
        if label == "spoof":
            values += 10
        array_path = arrays / f"u{index}.npz"
        np.savez_compressed(array_path, values=values)
        rows.append(
            {
                "schema": "mimodf-feature-record/v1",
                "utterance_id": f"u{index}",
                "array_path": str(array_path),
                "value_kind": "continuous",
                "label": label,
                "source_model": "a" if index < 6 else "b",
            }
        )
    (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_run_feature_probe_numpy(tmp_path):
    feature_dir = tmp_path / "features"
    _write_continuous_cache(feature_dir)

    result = run_feature_probe(
        ProbeSettings(
            feature_dir=feature_dir,
            out_dir=tmp_path / "probe",
            task="label",
            backend="numpy",
            max_iter=50,
            overwrite=True,
        )
    )

    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["schema"] == "mimodf-feature-probe/v1"
    assert metrics["records"] == 12
    assert metrics["train_records"] > 0
    assert metrics["test_records"] > 0
    assert metrics["pooling"] == "continuous_mean_std"
    assert metrics["command_argv"]
    assert "eer" in metrics
    assert result.report_path.is_file()
    predictions = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
    assert predictions
    assert set(predictions[0]["probabilities"]) == {"bonafide", "spoof"}


def test_rvq_histogram_pooling(tmp_path):
    array_path = tmp_path / "codes.npz"
    np.savez_compressed(array_path, values=np.array([[0, 1], [1, 1], [1, 2]], dtype=np.int16))
    record = {
        "array_path": str(array_path),
        "value_kind": "rvq_codes",
        "selected_quantizers": [2, 3],
    }
    manifest = {"model_config": {"codebook_size": [1024, 1024, 3, 4]}}

    pooled = pool_feature_record(record, manifest, "auto")

    assert pooled.pooling == "rvq_hist"
    assert pooled.vector.shape == (7,)
    np.testing.assert_allclose(pooled.vector[:3], [1 / 3, 2 / 3, 0])
    np.testing.assert_allclose(pooled.vector[3:], [0, 2 / 3, 1 / 3, 0])


def test_feature_probe_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_probe(settings):
        calls.append(settings)
        from mimodf.features.probe import ProbeResult

        return ProbeResult(
            report_path=Path("report.md"),
            metrics_path=Path("metrics.json"),
            predictions_path=Path("predictions.jsonl"),
            records=10,
            train_records=8,
            test_records=2,
        )

    monkeypatch.setattr(cli, "run_feature_probe", fake_probe)

    rc = cli.main(
        [
            "features",
            "probe",
            "--feature-dir",
            "features",
            "--out-dir",
            str(tmp_path / "probe"),
            "--task",
            "label",
            "--split",
            "holdout-values",
            "--holdout-field",
            "source_model",
            "--holdout-values",
            "A",
            "B",
            "--backend",
            "numpy",
            "--overwrite",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 10
    assert calls[0].split == "holdout-values"
    assert calls[0].holdout_values == ("A", "B")
    assert calls[0].backend == "numpy"


def _write_predictions(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_run_probe_fusion(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    rows_left = [
        {
            "utterance_id": "a",
            "target": "spoof",
            "prediction": "spoof",
            "probabilities": {"bonafide": 0.2, "spoof": 0.8},
            "label": "spoof",
        },
        {
            "utterance_id": "b",
            "target": "bonafide",
            "prediction": "spoof",
            "probabilities": {"bonafide": 0.4, "spoof": 0.6},
            "label": "bonafide",
        },
    ]
    rows_right = [
        {
            "utterance_id": "a",
            "target": "spoof",
            "prediction": "bonafide",
            "probabilities": {"bonafide": 0.6, "spoof": 0.4},
            "label": "spoof",
        },
        {
            "utterance_id": "b",
            "target": "bonafide",
            "prediction": "bonafide",
            "probabilities": {"bonafide": 0.8, "spoof": 0.2},
            "label": "bonafide",
        },
    ]
    _write_predictions(left, rows_left)
    _write_predictions(right, rows_right)

    result = run_probe_fusion(
        ProbeFusionSettings(
            left_predictions=left, right_predictions=right, out_dir=tmp_path / "fusion"
        )
    )

    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["records"] == 2
    assert metrics["overlap"]["single_disagreements"] == 2
    assert result.predictions_path.is_file()


def test_probe_fusion_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_fusion(settings):
        calls.append(settings)
        from mimodf.features.fusion import ProbeFusionResult

        return ProbeFusionResult(
            report_path=Path("report.md"),
            metrics_path=Path("metrics.json"),
            predictions_path=Path("predictions.jsonl"),
            records=2,
        )

    monkeypatch.setattr(cli, "run_probe_fusion", fake_fusion)

    rc = cli.main(
        [
            "features",
            "fuse-probes",
            "--left-predictions",
            "left.jsonl",
            "--right-predictions",
            "right.jsonl",
            "--out-dir",
            str(tmp_path / "fusion"),
            "--left-weight",
            "0.7",
            "--right-weight",
            "0.3",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["records"] == 2
    assert calls[0].left_weight == 0.7
    assert calls[0].right_weight == 0.3


def test_compare_predictions(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_predictions(
        left,
        [
            {
                "utterance_id": "a",
                "target": "spoof",
                "prediction": "spoof",
                "probabilities": {"bonafide": 0.2, "spoof": 0.8},
                "label": "spoof",
                "source_model": "x",
            },
            {
                "utterance_id": "b",
                "target": "bonafide",
                "prediction": "spoof",
                "probabilities": {"bonafide": 0.4, "spoof": 0.6},
                "label": "bonafide",
                "source_model": "x",
            },
        ],
    )
    _write_predictions(
        right,
        [
            {
                "utterance_id": "a",
                "target": "spoof",
                "prediction": "bonafide",
                "probabilities": {"bonafide": 0.6, "spoof": 0.4},
                "label": "spoof",
                "source_model": "x",
            },
            {
                "utterance_id": "b",
                "target": "bonafide",
                "prediction": "bonafide",
                "probabilities": {"bonafide": 0.8, "spoof": 0.2},
                "label": "bonafide",
                "source_model": "x",
            },
        ],
    )

    result = compare_predictions(
        PredictionComparisonSettings(
            sources=(PredictionSource("left", left), PredictionSource("right", right)),
            out_dir=tmp_path / "compare",
        )
    )

    summary = json.loads(result.summary_path.read_text())
    assert summary["records"] == 2
    assert summary["system_summary"]["left"]["wrong"] == 1
    assert summary["pair_summary"]["left vs right"]["disagreements"] == 2
    assert result.cases_path.is_file()


def test_parse_prediction_source():
    source = parse_prediction_source("ssl=/tmp/predictions.jsonl")
    assert source.name == "ssl"
    assert source.path == Path("/tmp/predictions.jsonl")


def test_compare_predictions_cli_passes_settings(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_compare(settings):
        calls.append(settings)
        from mimodf.features.predictions import PredictionComparisonResult

        return PredictionComparisonResult(
            report_path=Path("report.md"),
            summary_path=Path("summary.json"),
            cases_path=Path("cases.jsonl"),
            records=2,
        )

    monkeypatch.setattr(cli, "compare_predictions", fake_compare)

    rc = cli.main(
        [
            "features",
            "compare-predictions",
            "--predictions",
            "ssl=ssl.jsonl",
            "mimo=mimo.jsonl",
            "--out-dir",
            str(tmp_path / "compare"),
            "--overwrite",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["records"] == 2
    assert [source.name for source in calls[0].sources] == ["ssl", "mimo"]
