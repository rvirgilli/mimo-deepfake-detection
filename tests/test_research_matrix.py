import json

import pytest

import mimodf.cli as cli
from mimodf.research.matrix import (
    MatrixValidationError,
    load_matrix,
    matrix_counts,
    validate_matrix,
)


def _minimal_matrix():
    return {
        "schema_version": "representation-transfer-matrix/v1",
        "matrix_id": "fixture_matrix",
        "intent": "diagnostic",
        "purpose": "test matrix",
        "owner": "tests",
        "created": "2026-05-27",
        "frontends": [
            {"id": "ssl", "family": "ssl", "status": "ready", "purpose": "baseline"},
            {"id": "tok", "family": "tokenizer", "status": "planned", "purpose": "contrast"},
        ],
        "datasets": [{"id": "d", "purpose": "dataset"}],
        "shift_axes": [{"id": "source_holdout", "purpose": "source shift"}],
        "rows": [
            {
                "row_id": "r1",
                "kind": "feature_probe",
                "status": "planned",
                "dataset": "d",
                "shift_axis": "source_holdout",
                "frontends": ["ssl", "tok"],
                "question": "does it transfer?",
                "metrics": ["eer"],
                "approval_required": False,
            }
        ],
        "decision_policy": {"promote_if": ["signal"], "kill_if": ["no signal"]},
    }


def test_validate_minimal_matrix():
    data = _minimal_matrix()

    validate_matrix(data)

    assert matrix_counts(data)["rows"] == 1


def test_validate_matrix_rejects_unknown_frontend():
    data = _minimal_matrix()
    data["rows"][0]["frontends"] = ["missing"]

    with pytest.raises(MatrixValidationError, match="unknown frontend"):
        validate_matrix(data)


def test_real_representation_transfer_matrix_loads():
    matrix = load_matrix("docs/current/representation_transfer_matrix.yaml")

    counts = matrix_counts(matrix)
    assert counts["rows"] >= 5
    assert counts["rows_by_status"]["completed"] >= 1


def test_research_matrix_cli_validate_json(capsys):
    rc = cli.main(
        [
            "research",
            "validate-matrix",
            "docs/current/representation_transfer_matrix.yaml",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["matrix_id"] == "representation_transfer_shift_map_v1"


def test_research_matrix_cli_summary(capsys):
    rc = cli.main(
        ["research", "matrix-summary", "docs/current/representation_transfer_matrix.yaml"]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Representation-transfer matrix" in out
    assert "wave1_closed_codecfake_cosg_source_holdout" in out
