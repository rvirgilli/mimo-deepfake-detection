import json
import subprocess

from mimodf.audit.release_gate import (
    build_release_gate_report,
    render_release_gate_json,
    render_release_gate_markdown,
)
from mimodf.cli import main


def write_provenance(path, *, artifact_value="score.txt"):
    path.write_text(
        f"""
version: 1
rows:
  - id: row1
    model: M
    strategy: S
    n_source: 1 local
    status: partial
    notes: release gate fixture
    seeds:
      - id: 1
        source: local
        status: partial
        metrics: {{la_eer: 1.0, df_eer: 2.0, la_tdcf: 0.1}}
        artifacts:
          score: {artifact_value}
""".strip()
    )


def write_dependency_spec(path):
    path.write_text(
        """
dependencies:
  - name: tiny-dep
    path: tiny-dep
    policy: fixture dependency
    expected_git_remote: https://example.test/tiny.git
    expected_git_head: "1111111111111111111111111111111111111111"
    required_paths:
      - path: required.txt
        kind: file
        expected_sha256: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
""".strip()
    )


def make_git_dep(path):
    dep = path / "tiny-dep"
    dep.mkdir()
    (dep / "required.txt").write_text("hello")
    subprocess.run(["git", "init"], cwd=dep, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=dep, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=dep, check=True)
    subprocess.run(["git", "add", "."], cwd=dep, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=dep, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.test/tiny.git"], cwd=dep, check=True
    )
    return dep


def test_release_gate_reports_blockers_and_warnings(tmp_path):
    provenance = tmp_path / "provenance.yaml"
    spec = tmp_path / "deps.yaml"
    write_provenance(provenance, artifact_value="missing-score.txt")
    write_dependency_spec(spec)

    report = build_release_gate_report(
        provenance,
        root=tmp_path,
        dependency_spec_path=spec,
        dependency_local_spec_path=None,
    )

    assert report.verdict == "fail"
    assert report.artifact_status_counts["missing"] == 1
    assert {issue.code for issue in report.blockers} == {
        "artifact_missing",
        "dependency_missing",
        "dependency_required_path_missing",
    }
    assert any(issue.code == "dependency_hashes_not_checked" for issue in report.warnings)


def test_release_gate_hash_strict_and_renderers(tmp_path, capsys):
    provenance = tmp_path / "provenance.yaml"
    spec = tmp_path / "deps.yaml"
    (tmp_path / "score.txt").write_text("score")
    write_provenance(provenance)
    write_dependency_spec(spec)
    make_git_dep(tmp_path)

    report = build_release_gate_report(
        provenance,
        root=tmp_path,
        dependency_spec_path=spec,
        dependency_local_spec_path=None,
        hash_files=True,
    )

    assert report.verdict == "fail"
    assert "dependency_revision_mismatch" in {issue.code for issue in report.blockers}
    assert report.dependency_counts["hash_mismatch"] == 0
    assert "# Release gate" in render_release_gate_markdown(report)
    assert json.loads(render_release_gate_json(report))["verdict"] == "fail"

    rc = main(
        [
            "audit",
            "release-gate",
            str(provenance),
            "--root",
            str(tmp_path),
            "--dependency-spec",
            str(spec),
            "--format",
            "json",
            "--dependency-local-spec",
            "none",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "fail"

    strict_rc = main(
        [
            "audit",
            "release-gate",
            str(provenance),
            "--root",
            str(tmp_path),
            "--dependency-spec",
            str(spec),
            "--dependency-local-spec",
            "none",
            "--strict",
        ]
    )
    assert strict_rc == 1


def test_release_gate_uses_local_dependency_override_for_clean_clone(tmp_path):
    provenance = tmp_path / "provenance.yaml"
    spec = tmp_path / "deps.yaml"
    local = tmp_path / "deps.local.yaml"
    (tmp_path / "score.txt").write_text("score")
    write_provenance(provenance)
    spec.write_text(
        """
dependencies:
  - name: scorer
    path: dirty-scorer
    policy: fixture scorer
    required_paths:
      - path: required.txt
        kind: file
""".strip()
    )
    dirty = tmp_path / "dirty-scorer"
    dirty.mkdir()
    (dirty / "required.txt").write_text("ok")
    subprocess.run(["git", "init"], cwd=dirty, check=True, capture_output=True)
    (dirty / "untracked.txt").write_text("dirty\n")
    clean = tmp_path / "clean-scorer"
    clean.mkdir()
    (clean / "required.txt").write_text("ok")
    local.write_text(
        """
dependencies:
  - name: scorer
    path: clean-scorer
""".strip()
    )

    report = build_release_gate_report(
        provenance,
        root=tmp_path,
        dependency_spec_path=spec,
        dependency_local_spec_path=local,
    )

    assert report.verdict == "pass"
    assert not report.blockers


def test_release_gate_can_allow_explicit_known_artifact_gaps(tmp_path):
    provenance = tmp_path / "provenance.yaml"
    spec = tmp_path / "deps.yaml"
    gap_policy = tmp_path / "gaps.yaml"
    write_provenance(provenance, artifact_value="missing-score.txt")
    spec.write_text("dependencies: []\n")
    gap_policy.write_text(
        """
version: 1
policy: fixture known gap policy
gaps:
  - row_id: row1
    seed_id: "1"
    kind: score
    value: missing-score.txt
    decision: known_missing_score_only
    reason: fixture
    action: fixture
""".strip()
    )

    report = build_release_gate_report(
        provenance,
        root=tmp_path,
        dependency_spec_path=spec,
        dependency_local_spec_path=None,
        artifact_gap_policy_path=gap_policy,
        allow_known_artifact_gaps=True,
    )

    assert report.verdict == "pass"
    assert not report.blockers
    assert any(issue.code == "artifact_known_gap_allowed" for issue in report.warnings)


def test_current_release_gate_is_truthful_about_known_blockers():
    report = build_release_gate_report(
        "docs/current/main_table_provenance.yaml",
        dependency_local_spec_path=None,
    )
    blocker_codes = {issue.code for issue in report.blockers}

    assert report.verdict == "fail"
    assert "artifact_missing" in blocker_codes
    assert "dependency_dirty" in blocker_codes
    assert report.artifact_status_counts["missing"] == 11


def test_current_release_gate_can_be_reduced_to_known_artifact_gap_warnings(capsys):
    report = build_release_gate_report(
        "docs/current/main_table_provenance.yaml",
        allow_known_artifact_gaps=True,
    )
    blocker_codes = {issue.code for issue in report.blockers}
    warning_codes = {issue.code for issue in report.warnings}

    assert "artifact_missing" not in blocker_codes
    assert "artifact_known_gap_allowed" in warning_codes

    rc = main(["audit", "release-gate", "--system-profile", "--strict", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "pass"
