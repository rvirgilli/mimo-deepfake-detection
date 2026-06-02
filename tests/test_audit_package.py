import json
from pathlib import Path

from mimodf.audit.package import write_audit_package

PROVENANCE = Path("docs/current/main_table_provenance.yaml")


def test_write_audit_package_outputs_expected_files(tmp_path):
    package = write_audit_package(PROVENANCE, tmp_path, dependency_local_spec_path=None)

    expected = {
        "main_table_markdown",
        "artifact_checks_markdown",
        "artifact_checks_json",
        "artifact_gaps_markdown",
        "artifact_gaps_json",
        "artifact_gap_policy_yaml",
        "dependency_checks_markdown",
        "dependency_checks_json",
        "dependency_spec_yaml",
        "summary_json",
        "provenance_yaml",
        "tdcf_summary_markdown",
        "tdcf_summary_json",
        "tdcf_values_yaml",
    }
    assert set(package.files) == expected
    for path in package.files.values():
        assert Path(path).is_file()

    main_table = Path(package.files["main_table_markdown"]).read_text()
    assert "| wav2vec2 | Frozen | 5 local | 8.05 ± 0.73 |" in main_table

    checks = json.loads(Path(package.files["artifact_checks_json"]).read_text())
    assert any(check["status"] == "missing" for check in checks)
    assert package.artifact_status_counts["missing"] == 11

    gaps = json.loads(Path(package.files["artifact_gaps_json"]).read_text())
    assert len(gaps["gaps"]) == 11

    dependencies = json.loads(Path(package.files["dependency_checks_json"]).read_text())
    assert {dependency["name"] for dependency in dependencies} >= {
        "SSL_Anti-spoofing",
        "MiMo-Audio-Tokenizer",
    }

    tdcf_summary = json.loads(Path(package.files["tdcf_summary_json"]).read_text())
    assert any(
        row["row_id"] == "mimo_full" and row["mean"] == 0.34956 for row in tdcf_summary["rows"]
    )

    summary = json.loads(Path(package.files["summary_json"]).read_text())
    assert summary["artifact_status_counts"] == package.artifact_status_counts


def test_write_audit_package_can_omit_provenance_copy(tmp_path):
    package = write_audit_package(
        PROVENANCE,
        tmp_path,
        include_provenance_copy=False,
        dependency_local_spec_path=None,
    )

    assert "provenance_yaml" not in package.files
    assert not (tmp_path / "main_table_provenance.yaml").exists()
