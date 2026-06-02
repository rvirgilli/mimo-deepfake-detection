import json

from mimodf.cli import main

EXAMPLE_SPEC = "docs/current/examples/experiment_spec_v1_minimal.yaml"


def test_experiment_validate_cli_json(capsys):
    assert main(["experiment", "validate", EXAMPLE_SPEC, "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["spec_hash"].startswith("sha256:")


def test_experiment_init_cli_writes_run_layout(tmp_path, capsys):
    assert (
        main(
            [
                "experiment",
                "init",
                EXAMPLE_SPEC,
                "--seed",
                "42",
                "--root",
                str(tmp_path),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["spec_hash"].startswith("sha256:")
    assert payload["run_dir"].startswith(str(tmp_path))
    assert payload["resolved_spec"].endswith("resolved_spec.yaml")
    assert payload["manifest"].endswith("manifest.json")


def test_report_index_cli_writes_jsonl(tmp_path, capsys):
    assert (
        main(
            [
                "experiment",
                "init",
                EXAMPLE_SPEC,
                "--seed",
                "42",
                "--root",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    out = tmp_path / "index.jsonl"

    assert main(["report", "index", str(tmp_path / "runs"), "--out", str(out)]) == 0

    line = out.read_text().splitlines()[0]
    assert json.loads(line)["source_type"] == "new_run"


def test_report_aggregate_and_compare_cli(tmp_path, capsys):
    assert (
        main(
            [
                "experiment",
                "init",
                EXAMPLE_SPEC,
                "--seed",
                "42",
                "--root",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    index = tmp_path / "index.jsonl"
    assert main(["report", "index", str(tmp_path / "runs"), "--out", str(index)]) == 0

    assert main(["report", "aggregate", "--index", str(index)]) == 0
    assert "controlled_mimo_wav2vec2_v1" in capsys.readouterr().out

    rc = main(
        [
            "report",
            "compare",
            "--index",
            str(index),
            "--experiments",
            "controlled_mimo_wav2vec2_v1",
            "missing",
            "--strict",
        ]
    )
    assert rc == 1
    assert "missing" in capsys.readouterr().out


def test_experiment_resolve_and_inspect_cli(tmp_path, capsys):
    resolved = tmp_path / "resolved.yaml"
    assert main(["experiment", "resolve", EXAMPLE_SPEC, "--out", str(resolved)]) == 0
    assert resolved.is_file()
    capsys.readouterr()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert main(["experiment", "inspect", str(run_dir)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "manifest_present": False,
        "resolved_spec_present": False,
        "run_dir": str(run_dir),
    }
