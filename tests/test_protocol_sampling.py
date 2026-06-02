import json
from pathlib import Path

from mimodf.cli import main
from mimodf.data.protocol import ProtocolSampleSettings, sample_protocol


def _write_protocol(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_sample_protocol_groups_and_requires_audio(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for name in ["a.wav", "b.wav", "c.wav", "d.wav"]:
        (audio_dir / name).write_bytes(b"x")
    rows = [
        {
            "utterance_id": "a",
            "audio_path": str(audio_dir / "a.wav"),
            "label": "spoof",
            "source_model": "A",
        },
        {
            "utterance_id": "b",
            "audio_path": str(audio_dir / "b.wav"),
            "label": "spoof",
            "source_model": "A",
        },
        {
            "utterance_id": "c",
            "audio_path": str(audio_dir / "c.wav"),
            "label": "bonafide",
            "source_model": "A",
        },
        {
            "utterance_id": "d",
            "audio_path": str(audio_dir / "d.wav"),
            "label": "spoof",
            "source_model": "B",
        },
        {
            "utterance_id": "missing",
            "audio_path": str(audio_dir / "missing.wav"),
            "label": "spoof",
            "source_model": "C",
        },
    ]
    protocol = tmp_path / "protocol.jsonl"
    out = tmp_path / "sample.jsonl"
    _write_protocol(protocol, rows)

    summary = sample_protocol(
        ProtocolSampleSettings(
            input_path=protocol,
            out_path=out,
            group_by=("label", "source_model"),
            max_per_group=1,
            max_records=16,
            seed=7,
        )
    )

    sampled = [json.loads(line) for line in out.read_text().splitlines()]
    groups = {(row["label"], row["source_model"]) for row in sampled}
    assert summary.records_read == 5
    assert summary.records_eligible == 4
    assert summary.skipped_missing_audio == 1
    assert summary.records_written == 3
    assert groups == {("spoof", "A"), ("bonafide", "A"), ("spoof", "B")}


def test_sample_protocol_is_deterministic_for_seed(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for index in range(10):
        path = audio_dir / f"{index}.wav"
        path.write_bytes(b"x")
        rows.append(
            {
                "utterance_id": str(index),
                "audio_path": str(path),
                "label": "spoof" if index % 2 else "bonafide",
                "source_model": f"S{index % 4}",
            }
        )
    protocol = tmp_path / "protocol.jsonl"
    _write_protocol(protocol, rows)

    settings = {
        "input_path": protocol,
        "group_by": ("label", "source_model"),
        "max_per_group": 1,
        "max_records": 6,
        "seed": 123,
    }
    sample_protocol(ProtocolSampleSettings(out_path=tmp_path / "left.jsonl", **settings))
    sample_protocol(ProtocolSampleSettings(out_path=tmp_path / "right.jsonl", **settings))

    assert (tmp_path / "left.jsonl").read_text() == (tmp_path / "right.jsonl").read_text()


def test_sample_protocol_cli_writes_json_summary(tmp_path, capsys):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"x")
    protocol = tmp_path / "protocol.jsonl"
    out = tmp_path / "sample.jsonl"
    summary_out = tmp_path / "summary.json"
    _write_protocol(
        protocol,
        [
            {
                "utterance_id": "x",
                "audio_path": str(audio),
                "label": "spoof",
                "source_model": "A",
            }
        ],
    )

    rc = main(
        [
            "data",
            "sample-protocol",
            "--input",
            str(protocol),
            "--out",
            str(out),
            "--group-by",
            "label",
            "source_model",
            "--max-per-group",
            "1",
            "--max-records",
            "4",
            "--summary-out",
            str(summary_out),
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["records_written"] == 1
    assert json.loads(summary_out.read_text())["records_written"] == 1
    assert json.loads(out.read_text())["utterance_id"] == "x"
