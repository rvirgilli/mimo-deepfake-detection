import json

from mimodf.cli import main
from mimodf.scoring.compare import compare_score_files, render_score_comparison_markdown


def test_compare_score_files_reports_overlap_and_diffs(tmp_path):
    candidate = tmp_path / "candidate.txt"
    reference = tmp_path / "reference.txt"
    candidate.write_text("utt_a 1.0\nutt_b 2.5\n")
    reference.write_text("utt_a 1.1\nutt_c 9.0\n")

    comparison = compare_score_files(candidate, reference, tolerance=0.2)

    assert comparison.passed is True
    assert comparison.candidate_count == 2
    assert comparison.reference_count == 2
    assert comparison.common_count == 1
    assert comparison.candidate_only_count == 1
    assert comparison.reference_only_count == 1
    assert comparison.max_abs_diff == pytest_approx(0.1)
    assert "Score-file comparison" in render_score_comparison_markdown(comparison)


def test_compare_score_files_fails_when_tolerance_exceeded(tmp_path):
    candidate = tmp_path / "candidate.txt"
    reference = tmp_path / "reference.txt"
    candidate.write_text("utt_a 1.0\n")
    reference.write_text("utt_a 2.0\n")

    comparison = compare_score_files(candidate, reference, tolerance=0.5)

    assert comparison.passed is False


def test_compare_score_files_cli_json_and_strict(tmp_path, capsys):
    candidate = tmp_path / "candidate.txt"
    reference = tmp_path / "reference.txt"
    candidate.write_text("utt_a 1.0\n")
    reference.write_text("utt_a 1.0\n")

    rc = main(
        [
            "score",
            "compare-files",
            str(candidate),
            str(reference),
            "--tolerance",
            "0.0",
            "--format",
            "json",
            "--strict",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def pytest_approx(value):
    import pytest

    return pytest.approx(value)
