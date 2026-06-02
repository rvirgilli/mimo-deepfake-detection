import json
from pathlib import Path

import pytest

from mimodf.cli import main
from mimodf.data.codecfake_splits import build_source_holdout_plan, build_source_holdout_rows


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
                wav.write_bytes(b"x")
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


def test_source_holdout_plan_uses_heldout_only_for_test(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_protocol(protocol, _rows(tmp_path))

    plan = build_source_holdout_plan(
        protocol=protocol,
        min_per_label=2,
        validation_source_count=1,
        seed=42,
    )

    assert plan.eligible_sources == ("A", "B", "C", "D")
    assert len(plan.folds) == 4
    for fold in plan.folds:
        assert fold.heldout_source not in fold.validation_sources
        assert fold.heldout_source not in fold.train_sources
        assert set(fold.validation_sources).isdisjoint(fold.train_sources)
        assert fold.heldout_labels == {"bonafide": 2, "spoof": 2}
        assert fold.validation_labels == {"bonafide": 2, "spoof": 2}
        assert fold.train_labels == {"bonafide": 4, "spoof": 4}


def test_source_holdout_plan_is_deterministic(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_protocol(protocol, _rows(tmp_path))

    left = build_source_holdout_plan(protocol=protocol, min_per_label=2, seed=7)
    right = build_source_holdout_plan(protocol=protocol, min_per_label=2, seed=7)

    assert left.to_dict() == right.to_dict()


def test_source_holdout_plan_stratified_row_policy_keeps_heldout_clean(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_protocol(protocol, _rows(tmp_path))

    plan = build_source_holdout_plan(
        protocol=protocol,
        min_per_label=2,
        validation_policy="stratified-row",
        validation_fraction=0.25,
    )

    fold = next(item for item in plan.folds if item.heldout_source == "A")
    assert fold.validation_sources == ()
    assert "A" not in fold.train_sources
    assert fold.heldout_labels == {"bonafide": 2, "spoof": 2}
    assert fold.validation_labels == {"bonafide": 3, "spoof": 3}
    assert fold.train_labels == {"bonafide": 3, "spoof": 3}


def test_source_holdout_plan_prefers_small_validation_source(tmp_path):
    rows = _rows(tmp_path)
    audio = tmp_path / "audio"
    for index in range(10):
        wav = audio / f"D_extra_spoof_{index}.wav"
        wav.write_bytes(b"x")
        rows.append(
            {
                "subset": "CoSG",
                "utterance_id": wav.stem,
                "audio_path": str(wav),
                "label": "spoof",
                "source_model": "D",
            }
        )
    protocol = tmp_path / "protocol.jsonl"
    _write_protocol(protocol, rows)

    plan = build_source_holdout_plan(protocol=protocol, min_per_label=2, seed=42)
    fold = next(item for item in plan.folds if item.heldout_source == "A")

    assert fold.validation_sources != ("D",)


def test_source_holdout_plan_requires_enough_sources(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    rows = [row for row in _rows(tmp_path) if row["source_model"] in {"A", "B"}]
    _write_protocol(protocol, rows)

    with pytest.raises(ValueError, match="not enough eligible sources"):
        build_source_holdout_plan(protocol=protocol, min_per_label=2)


def test_source_holdout_rows_materializes_train_validation_test(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_protocol(protocol, _rows(tmp_path))

    rows = build_source_holdout_rows(
        protocol=protocol,
        heldout_source="A",
        validation_policy="stratified-row",
        validation_fraction=0.25,
    )

    assert {row["source_model"] for row in rows.test_rows} == {"A"}
    assert {row["source_model"] for row in rows.train_rows}.isdisjoint({"A"})
    assert {row["source_model"] for row in rows.validation_rows}.isdisjoint({"A"})
    assert len(rows.train_rows) == 6
    assert len(rows.validation_rows) == 6
    assert len(rows.test_rows) == 4


def test_source_holdout_plan_cli_writes_json_summary(tmp_path, capsys):
    protocol = tmp_path / "protocol.jsonl"
    summary = tmp_path / "summary.json"
    _write_protocol(protocol, _rows(tmp_path))

    rc = main(
        [
            "data",
            "codecfake-source-holdout-plan",
            "--protocol",
            str(protocol),
            "--min-per-label",
            "2",
            "--summary-out",
            str(summary),
            "--format",
            "json",
        ]
    )

    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(summary.read_text())
    assert printed == written
    assert printed["eligible_sources"] == ["A", "B", "C", "D"]
    assert len(printed["folds"]) == 4
