import json

import mimodf.cli as cli
from mimodf.logs.execution import summarize_log, validate_log


def _write_log(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(run_id="run-1", status="completed"):
    return {
        "schema": "mimodf-research-execution-log/v1",
        "run_id": run_id,
        "wave": "wave-test",
        "kind": "feature_probe",
        "status": status,
        "cwd": "/tmp/repo",
        "environment": "test",
        "command": "python -m thing",
        "inputs": ["in"],
        "planned_outputs": ["out"],
        "outputs": ["out"],
        "git_revision_at_plan": "abc",
        "git_revision_at_run": "abc",
        "result_summary": {"records": 1},
    }


def test_validate_log_accepts_complete_log(tmp_path):
    path = tmp_path / "log.jsonl"
    _write_log(path, [_row()])

    result = validate_log(path)

    assert result.passed
    assert result.rows == 1
    assert result.warnings == ()


def test_validate_log_rejects_duplicate_run_id(tmp_path):
    path = tmp_path / "log.jsonl"
    _write_log(path, [_row("same"), _row("same")])

    result = validate_log(path)

    assert not result.passed
    assert "duplicate run_id" in result.errors[0].message


def test_validate_log_warns_for_planned_rows_by_default(tmp_path):
    path = tmp_path / "log.jsonl"
    row = _row(status="planned")
    row.pop("outputs")
    row.pop("result_summary")
    row.pop("git_revision_at_run")
    _write_log(path, [row])

    result = validate_log(path)

    assert result.passed
    assert any("planned row" in warning.message for warning in result.warnings)


def test_validate_log_strict_fails_for_incomplete_completed_row(tmp_path):
    path = tmp_path / "log.jsonl"
    row = _row()
    row.pop("outputs")
    _write_log(path, [row])

    result = validate_log(path, strict=True)

    assert not result.passed
    assert any("outputs" in error.message for error in result.errors)


def test_summarize_log_counts_terminal_statuses(tmp_path):
    path = tmp_path / "log.jsonl"
    _write_log(path, [_row("ok"), _row("bad", "failed"), _row("stopped", "interrupted")])

    summary = summarize_log(path)

    terminal_rows = sum(
        summary.by_status.get(status, 0) for status in ("completed", "failed", "interrupted")
    )
    assert summary.rows == 3
    assert terminal_rows == summary.rows
    assert summary.by_status["completed"] == 1
    assert summary.by_status["failed"] == 1
    assert summary.by_status["interrupted"] == 1


def test_log_validate_cli_json(tmp_path, capsys):
    path = tmp_path / "log.jsonl"
    _write_log(path, [_row()])

    rc = cli.main(["log", "validate", str(path), "--format", "json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["rows"] == 1


def test_log_summary_cli_markdown(tmp_path, capsys):
    path = tmp_path / "log.jsonl"
    _write_log(path, [_row()])

    rc = cli.main(["log", "summary", str(path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Research execution log summary" in out
    assert "By kind" in out
