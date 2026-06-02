import json
import subprocess

from mimodf.audit.dependencies import (
    audit_external_dependencies,
    load_dependency_specs,
    render_dependency_checks_json,
    render_dependency_checks_markdown,
)
from mimodf.cli import main


def test_dependency_specs_load_pinned_revisions():
    specs = {spec.name: spec for spec in load_dependency_specs(local_path=None)}

    assert specs["SSL_Anti-spoofing"].expected_git_remote == (
        "https://github.com/TakHemlata/SSL_Anti-spoofing.git"
    )
    assert specs["SSL_Anti-spoofing"].expected_git_head == (
        "4acaa61dcef5f7610f43aa4d0b29c4559b970cd2"
    )
    assert specs["MiMo-Audio-Tokenizer"].expected_git_head == (
        "b62b59922979bf9f389b373169298a251587653f"
    )
    xlsr = next(
        item for item in specs["SSL_Anti-spoofing"].required_paths if item.path == "xlsr2_300m.pt"
    )
    assert xlsr.expected_sha256 == (
        "b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9"
    )


def test_dependency_local_override_changes_path_without_mutating_base(tmp_path):
    base = tmp_path / "base.yaml"
    local = tmp_path / "local.yaml"
    base.write_text(
        """
dependencies:
  - name: dep
    path: dirty-dep
    policy: base policy
    expected_git_remote: https://example.test/dep.git
    expected_git_head: "1111111111111111111111111111111111111111"
    required_paths:
      - path: required.txt
        kind: file
""".strip()
    )
    local.write_text(
        """
dependencies:
  - name: dep
    path: clean-dep
    notes:
      - use clean local clone
""".strip()
    )

    specs = {spec.name: spec for spec in load_dependency_specs(base, local_path=local)}

    assert specs["dep"].path == "clean-dep"
    assert specs["dep"].expected_git_head == "1111111111111111111111111111111111111111"
    assert specs["dep"].required_paths[0].path == "required.txt"
    assert specs["dep"].spec_source == f"local:{local}"


def test_dependency_audit_reports_missing_local_dependencies(tmp_path):
    checks = audit_external_dependencies(root=tmp_path, local_spec_path=None)

    by_name = {check.name: check for check in checks}

    assert by_name["SSL_Anti-spoofing"].present is False
    assert by_name["SSL_Anti-spoofing"].git_head is None
    assert any(not item.present for item in by_name["SSL_Anti-spoofing"].required_paths)
    assert by_name["MiMo-Audio-Tokenizer"].present is False


def test_dependency_audit_reports_git_revision_dirty_state_and_required_paths(tmp_path):
    dep = tmp_path / "SSL_Anti-spoofing"
    dep.mkdir()
    (dep / "evaluate_2021_LA.py").write_text("print('score')\n")
    (dep / "eval_metric_LA.py").write_text("# metric\n")
    (dep / "RawBoost.py").write_text("# rawboost\n")
    (dep / "xlsr2_300m.pt").write_text("weights placeholder\n")
    subprocess.run(["git", "init"], cwd=dep, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=dep, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=dep, check=True)
    subprocess.run(["git", "add", "."], cwd=dep, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=dep, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.test/ssl.git"], cwd=dep, check=True
    )
    (dep / "untracked.txt").write_text("dirty\n")

    checks = audit_external_dependencies(root=tmp_path, local_spec_path=None)
    ssl = next(check for check in checks if check.name == "SSL_Anti-spoofing")

    assert ssl.present is True
    assert ssl.git_head is not None
    assert ssl.git_remote == "https://example.test/ssl.git"
    assert ssl.git_remote_matches_expected is False
    assert ssl.git_head_matches_expected is False
    assert ssl.git_dirty is True
    assert "?? untracked.txt" in ssl.git_status_lines
    assert all(item.present for item in ssl.required_paths)
    xlsr = next(item for item in ssl.required_paths if item.path == "xlsr2_300m.pt")
    assert xlsr.size_bytes == len("weights placeholder\n")
    assert xlsr.sha256 is None
    assert xlsr.sha256_matches_expected is None


def test_dependency_audit_can_hash_required_files(tmp_path):
    dep = tmp_path / "SSL_Anti-spoofing"
    dep.mkdir()
    (dep / "evaluate_2021_LA.py").write_text("print('score')\n")
    (dep / "eval_metric_LA.py").write_text("# metric\n")
    (dep / "RawBoost.py").write_text("# rawboost\n")
    (dep / "xlsr2_300m.pt").write_text("weights placeholder\n")

    checks = audit_external_dependencies(root=tmp_path, local_spec_path=None, hash_files=True)
    ssl = next(check for check in checks if check.name == "SSL_Anti-spoofing")
    xlsr = next(item for item in ssl.required_paths if item.path == "xlsr2_300m.pt")

    assert xlsr.sha256 == "f2c293e6c3ea00ec232e826bc01e3b837a6426174277db8249d874dd88b3872a"
    assert xlsr.sha256_matches_expected is False


def test_dependency_renderers_and_cli(tmp_path, capsys):
    (tmp_path / "MiMo-Audio-Tokenizer" / "mimo_audio_tokenizer").mkdir(parents=True)
    (tmp_path / "MiMo-Audio-Tokenizer" / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "MiMo-Audio-Tokenizer" / "mimo_audio_tokenizer" / "__init__.py").write_text("")
    checks = audit_external_dependencies(root=tmp_path, local_spec_path=None)

    rendered_json = json.loads(render_dependency_checks_json(checks))
    rendered_markdown = render_dependency_checks_markdown(checks)

    assert rendered_json[0]["name"] == "SSL_Anti-spoofing"
    assert "# External dependency audit" in rendered_markdown
    assert "MiMo-Audio-Tokenizer" in rendered_markdown

    rc = main(
        [
            "audit",
            "dependencies",
            "--root",
            str(tmp_path),
            "--format",
            "json",
            "--local-spec",
            "none",
        ]
    )
    assert rc == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert any(item["name"] == "MiMo-Audio-Tokenizer" for item in cli_json)
