import json

from mimodf.cli import main
from mimodf.data.codecfake import (
    build_codecfake_plus_index,
    load_cors_labels,
    load_cosg_labels,
    render_codecfake_summary_markdown,
)


def test_load_cosg_labels_with_audio_paths(tmp_path):
    labels = tmp_path / "CoSG_labels.txt"
    labels.write_text(
        "CLAMTTS CLAMTTS_1 Mvq None Freq Spoof\nCLAMTTS CLAMTTS_24 Real Real Real bonafide\n"
    )
    audio = tmp_path / "CoSG"
    audio.mkdir()
    (audio / "CLAMTTS_1.wav").write_bytes(b"fake")
    (audio / "CLAMTTS_24.wav").write_bytes(b"real")

    records = load_cosg_labels(labels, audio_root=audio)

    assert [record.label for record in records] == ["spoof", "bonafide"]
    assert records[0].source_model == "CLAMTTS"
    assert records[0].quantizer_type == "Mvq"
    assert records[0].auxiliary_objective == "None"
    assert records[0].decoder_type == "Freq"
    assert records[0].archive_member == "CoSG/CLAMTTS_1.wav"
    assert records[1].quantizer_type is None
    assert records[1].audio_path == str(audio / "CLAMTTS_24.wav")


def test_load_cors_labels_parses_codec_and_split_with_file(tmp_path):
    labels = tmp_path / "CoRS_labels.txt"
    labels.write_text(
        "p225 p225_001.wav bonafide\n"
        "p226 p226_002_Encodec_24b24k.wav spoof\n"
        "p228 p228_003_bigcodec.wav spoof\n"
    )

    records = load_cors_labels(labels)

    assert records[0].clip_id == "p225_001"
    assert records[0].codec_name is None
    assert records[0].split_hint == "train"
    assert records[1].clip_id == "p226_002"
    assert records[1].codec_name == "Encodec_24b24k"
    assert records[1].split_hint == "validation"
    assert records[2].split_hint == "evaluation"
    assert records[1].caveats == ("CoRS taxonomy fields require an explicit codec-name mapping",)


def test_build_codecfake_plus_index_writes_jsonl_and_summary(tmp_path):
    cosg = tmp_path / "CoSG_labels.txt"
    cosg.write_text(
        "MASKGCT MASKGCT_1 Rvq None Time Spoof\nMASKGCT MASKGCT_2 Real Real Real bonafide\n"
    )
    cors = tmp_path / "CoRS_labels.txt"
    cors.write_text("p225 p225_001.wav bonafide\n")
    audio = tmp_path / "CoSG"
    audio.mkdir()
    (audio / "MASKGCT_1.wav").write_bytes(b"x")
    out = tmp_path / "index.jsonl"

    summary = build_codecfake_plus_index(
        cosg_labels=cosg,
        cors_labels=cors,
        cosg_audio_root=audio,
        out=out,
    )

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["schema"] == "mimodf-protocol-record/v1"
    assert rows[0]["dataset_id"] == "codecfake_plus"
    assert summary.records == 3
    assert summary.subsets == {"CoRS": 1, "CoSG": 2}
    assert summary.labels == {"bonafide": 2, "spoof": 1}
    assert summary.missing_audio == 1
    assert summary.inputs["cosg_labels"]["sha256"]


def test_render_codecfake_summary_markdown(tmp_path):
    labels = tmp_path / "CoSG_labels.txt"
    labels.write_text("A A_1 Mvq None Freq Spoof\n")
    summary = build_codecfake_plus_index(cosg_labels=labels, out=tmp_path / "index.jsonl")

    rendered = render_codecfake_summary_markdown(summary)

    assert "# CodecFake+ protocol index summary" in rendered
    assert "- records: 1" in rendered
    assert "`spoof`: 1" in rendered


def test_codecfake_plus_index_cli(tmp_path, capsys):
    labels = tmp_path / "CoSG_labels.txt"
    labels.write_text("A A_1 Mvq None Freq Spoof\n")
    out = tmp_path / "index.jsonl"
    summary_out = tmp_path / "summary.json"

    rc = main(
        [
            "data",
            "codecfake-plus-index",
            "--cosg-labels",
            str(labels),
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["records"] == 1
    assert json.loads(summary_out.read_text())["records"] == 1
    assert json.loads(out.read_text())["utterance_id"] == "A_1"
