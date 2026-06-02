import json
import wave
from pathlib import Path

import pytest

from mimodf.cli import main
from mimodf.data.codecfake_splits import build_source_holdout_plan, render_source_holdout_plan_json
from mimodf.training.codecfake import (
    CodecfakeXlsrPlanSettings,
    build_codecfake_xlsr_dry_run_plan,
    check_codecfake_xlsr_loaders,
)
from mimodf.training.codecfake_metrics import build_prediction_metrics


def _write_protocol(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_wav(path: Path, frames: int = 8, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"\x00\x00" * frames)


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


def _split_plan(tmp_path: Path) -> tuple[Path, Path]:
    protocol = tmp_path / "protocol.jsonl"
    split_plan = tmp_path / "split_plan.json"
    _write_protocol(protocol, _rows(tmp_path))
    plan = build_source_holdout_plan(
        protocol=protocol,
        min_per_label=2,
        validation_policy="stratified-row",
        validation_fraction=0.25,
    )
    split_plan.write_text(render_source_holdout_plan_json(plan))
    return protocol, split_plan


def test_codecfake_xlsr_dry_run_plan_counts_and_paths(tmp_path):
    protocol, split_plan = _split_plan(tmp_path)

    plan = build_codecfake_xlsr_dry_run_plan(
        CodecfakeXlsrPlanSettings(
            split_plan=split_plan,
            protocol=protocol,
            fold="A",
            condition="xlsr_frozen_backend",
            seed=42,
            out_dir=tmp_path / "runs",
        )
    )
    payload = plan.to_dict()

    assert payload["implementation_status"] == "dry_run_only_no_training"
    assert payload["counts"]["train"]["records"] == 6
    assert payload["counts"]["validation"]["records"] == 6
    assert payload["counts"]["test"]["records"] == 4
    assert payload["counts"]["test"]["sources"] == {"A": 4}
    assert payload["condition_metadata"]["adaptation"] == "adaptation:frozen/v1"
    assert payload["output_paths"]["run_dir"].endswith("xlsr_frozen_backend/seed_42/A")


def test_codecfake_xlsr_loader_check_reports_batch_shapes(tmp_path):
    pytest.importorskip("torch")
    protocol, split_plan = _split_plan(tmp_path)

    check = check_codecfake_xlsr_loaders(
        CodecfakeXlsrPlanSettings(
            split_plan=split_plan,
            protocol=protocol,
            fold="A",
            condition="xlsr_frozen_backend",
            seed=42,
            out_dir=tmp_path / "runs",
        ),
        batch_size=2,
        eval_batch_size=3,
        cut=12,
    )

    assert check["train"]["input_shape"] == [2, 12]
    assert check["validation"]["input_shape"] == [3, 12]
    assert check["test"]["input_shape"] == [3, 12]


def test_codecfake_xlsr_cli_requires_dry_run_or_model_smoke(tmp_path, capsys):
    protocol, split_plan = _split_plan(tmp_path)

    rc = main(
        [
            "train",
            "codecfake-xlsr",
            "--split-plan",
            str(split_plan),
            "--protocol",
            str(protocol),
            "--fold",
            "A",
            "--condition",
            "xlsr_frozen_backend",
            "--seed",
            "42",
            "--out",
            str(tmp_path / "runs"),
        ]
    )

    assert rc == 1
    assert "--dry-run, --model-smoke, and/or --train-run" in capsys.readouterr().out


def test_build_prediction_metrics_includes_label_convention_and_per_source():
    records = [
        {
            "utterance_id": "a",
            "target": "bonafide",
            "prediction": "bonafide",
            "probabilities": {"bonafide": 0.9, "spoof": 0.1},
            "source_model": "S1",
        },
        {
            "utterance_id": "b",
            "target": "spoof",
            "prediction": "spoof",
            "probabilities": {"bonafide": 0.2, "spoof": 0.8},
            "source_model": "S1",
        },
        {
            "utterance_id": "c",
            "target": "spoof",
            "prediction": "bonafide",
            "probabilities": {"bonafide": 0.6, "spoof": 0.4},
            "source_model": "S2",
        },
    ]

    metrics = build_prediction_metrics(records)

    assert metrics["records"] == 3
    assert metrics["label_convention"] == {"bonafide": 0, "spoof": 1}
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_source"]["S1"]["accuracy"] == pytest.approx(1.0)
    assert metrics["per_source"]["S2"]["skipped"] == "single_class_support"
    assert metrics["score_summary_by_label"]["spoof"]["records"] == 2


def test_prediction_metrics_mark_binary_metrics_undefined_for_single_class():
    metrics = build_prediction_metrics(
        [
            {
                "utterance_id": "a",
                "target": "spoof",
                "prediction": "bonafide",
                "probabilities": {"bonafide": 0.6, "spoof": 0.4},
                "source_model": "S",
            }
        ]
    )

    assert metrics["auroc"] is None
    assert metrics["eer"] is None
    assert metrics["binary_metric_status"] == "undefined_single_class_support"


def test_codecfake_xlsr_cli_prints_plan(tmp_path, capsys):
    protocol, split_plan = _split_plan(tmp_path)

    rc = main(
        [
            "train",
            "codecfake-xlsr",
            "--split-plan",
            str(split_plan),
            "--protocol",
            str(protocol),
            "--fold",
            "A",
            "--condition",
            "xlsr_frozen_backend",
            "--seed",
            "42",
            "--out",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["test"]["records"] == 4
    assert payload["caveats"]
