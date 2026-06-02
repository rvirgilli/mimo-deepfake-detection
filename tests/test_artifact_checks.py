from pathlib import Path

from mimodf.audit.artifacts import check_artifacts_from_file, render_artifact_checks_json


def test_artifact_checker_reports_present_missing_and_declared_absent(tmp_path):
    existing = tmp_path / "score.txt"
    existing.write_text("abc")
    provenance = tmp_path / "provenance.yaml"
    provenance.write_text(
        """
version: 1
rows:
  - id: row1
    model: M
    strategy: S
    n_source: 2 local
    status: partial
    notes: artifact check fixture
    tdcf_note: fixture has one tDCF only
    seeds:
      - id: 1
        source: local
        status: partial
        metrics: {la_eer: 1.0, df_eer: 2.0, la_tdcf: 0.1}
        artifacts:
          score: score.txt
          checkpoint: missing
      - id: 2
        source: local
        status: partial
        metrics: {la_eer: 3.0, df_eer: 4.0}
        artifacts:
          score: missing-score.txt
""".strip()
    )

    checks = check_artifacts_from_file(provenance, root=tmp_path)
    by_kind_value = {(check.kind, check.value): check for check in checks}

    assert by_kind_value[("score", "score.txt")].status == "present"
    assert by_kind_value[("score", "score.txt")].sha256 is not None
    assert by_kind_value[("checkpoint", "missing")].status == "declared_absent"
    assert by_kind_value[("score", "missing-score.txt")].status == "missing"


def test_artifact_checker_marks_undeclared_seed_artifacts(tmp_path):
    provenance = tmp_path / "provenance.yaml"
    provenance.write_text(
        """
version: 1
rows:
  - id: row1
    model: M
    strategy: S
    n_source: 1 local
    status: partial
    notes: no artifacts fixture
    seeds:
      - id: 1
        source: local
        status: partial
        metrics: {la_eer: 1.0, df_eer: 2.0, la_tdcf: 0.1}
""".strip()
    )

    checks = check_artifacts_from_file(provenance, root=tmp_path)

    assert len(checks) == 1
    assert checks[0].status == "not_declared"
    assert checks[0].kind == "artifacts"


def test_artifact_checks_have_json_output():
    checks = check_artifacts_from_file(Path("docs/current/main_table_provenance.yaml"))
    payload = render_artifact_checks_json(checks)

    assert '"row_id": "wav2vec2_frozen"' in payload
    assert '"status": "present"' in payload
    assert '"status": "missing"' in payload
    assert '"status": "declared_absent"' in payload


def test_current_checkpoint_gaps_are_machine_detected():
    checks = check_artifacts_from_file(Path("docs/current/main_table_provenance.yaml"))
    missing_checkpoints = {
        (check.row_id, check.seed_id)
        for check in checks
        if check.kind == "checkpoint" and check.status == "missing"
    }

    assert missing_checkpoints == {
        ("wav2vec2_adapter", "42"),
        ("wav2vec2_adapter", "789"),
        ("mimo_frozen", "123"),
        ("mimo_frozen", "789"),
        ("mimo_frozen", "1234"),
        ("mimo_adapter", "2024"),
        ("mimo_full", "42"),
        ("mimo_full", "789"),
        ("mimo_full", "1234"),
    }
