"""Validation and rendering for representation-transfer research matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

MATRIX_SCHEMA = "representation-transfer-matrix/v1"
INTENTS = {"diagnostic", "exploratory", "confirmatory"}
STATUSES = {"planned", "ready", "blocked", "running", "completed", "dropped"}
FRONTEND_FAMILIES = {"ssl", "tokenizer", "codec", "acoustic", "spectral", "other"}
ROW_KINDS = {"feature_probe", "eval_only", "training", "indexing", "analysis"}


class MatrixValidationError(ValueError):
    """Raised when a representation-transfer matrix violates the contract."""


def load_matrix(path: str | Path) -> dict[str, Any]:
    matrix_path = Path(path)
    data = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MatrixValidationError("matrix must be a YAML mapping")
    validate_matrix(data)
    return data


def validate_matrix(data: dict[str, Any]) -> None:
    if data.get("schema_version") != MATRIX_SCHEMA:
        raise MatrixValidationError(f"schema_version must be {MATRIX_SCHEMA}")
    _require_str(data, "matrix_id")
    intent = _require_str(data, "intent")
    if intent not in INTENTS:
        raise MatrixValidationError(f"intent must be one of {sorted(INTENTS)}")
    _require_str(data, "purpose")
    _require_str(data, "owner")
    _require_str(data, "created")

    frontends = _require_list(data, "frontends")
    frontend_ids = _validate_frontends(frontends)
    datasets = _require_list(data, "datasets")
    dataset_ids = _validate_named_items(datasets, "datasets")
    shift_axes = _require_list(data, "shift_axes")
    shift_axis_ids = _validate_named_items(shift_axes, "shift_axes")
    rows = _require_list(data, "rows")
    _validate_rows(
        rows, frontend_ids=frontend_ids, dataset_ids=dataset_ids, shift_axis_ids=shift_axis_ids
    )

    decisions = _require_mapping(data, "decision_policy")
    _require_list(decisions, "promote_if", display="decision_policy.promote_if")
    _require_list(decisions, "kill_if", display="decision_policy.kill_if")


def render_matrix_summary(data: dict[str, Any]) -> str:
    validate_matrix(data)
    rows = data["rows"]
    lines = [
        f"# Representation-transfer matrix: {data['matrix_id']}",
        "",
        f"Intent: `{data['intent']}`",
        "",
        data["purpose"].strip(),
        "",
        "## Frontends",
        "",
        "| ID | Family | Status | Purpose |",
        "|---|---|---|---|",
    ]
    for item in data["frontends"]:
        lines.append(
            f"| `{item['id']}` | {item['family']} | {item['status']} | {item['purpose']} |"
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Row | Kind | Dataset | Shift axis | Status | Approval |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        approval = "yes" if row.get("approval_required", False) else "no"
        lines.append(
            f"| `{row['row_id']}` | {row['kind']} | {row['dataset']} | "
            f"{row['shift_axis']} | {row['status']} | {approval} |"
        )
    counts = _counts(data)
    lines.extend(
        [
            "",
            "## Counts",
            "",
            "```json",
            json.dumps(counts, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def matrix_counts(data: dict[str, Any]) -> dict[str, object]:
    validate_matrix(data)
    return _counts(data)


def _counts(data: dict[str, Any]) -> dict[str, object]:
    rows = data["rows"]
    return {
        "frontends": len(data["frontends"]),
        "datasets": len(data["datasets"]),
        "shift_axes": len(data["shift_axes"]),
        "rows": len(rows),
        "rows_by_kind": _count_by(rows, "kind"),
        "rows_by_status": _count_by(rows, "status"),
        "rows_requiring_approval": sum(1 for row in rows if row.get("approval_required", False)),
    }


def _validate_frontends(frontends: list[Any]) -> set[str]:
    ids: set[str] = set()
    for index, item in enumerate(frontends):
        if not isinstance(item, dict):
            raise MatrixValidationError(f"frontends[{index}] must be a mapping")
        item_id = _require_str(item, "id", display=f"frontends[{index}].id")
        if item_id in ids:
            raise MatrixValidationError(f"duplicate frontend id: {item_id}")
        ids.add(item_id)
        family = _require_str(item, "family", display=f"frontends[{index}].family")
        if family not in FRONTEND_FAMILIES:
            raise MatrixValidationError(
                f"frontends[{index}].family must be one of {sorted(FRONTEND_FAMILIES)}"
            )
        status = _require_str(item, "status", display=f"frontends[{index}].status")
        if status not in STATUSES:
            raise MatrixValidationError(
                f"frontends[{index}].status must be one of {sorted(STATUSES)}"
            )
        _require_str(item, "purpose", display=f"frontends[{index}].purpose")
    return ids


def _validate_named_items(items: list[Any], field: str) -> set[str]:
    ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise MatrixValidationError(f"{field}[{index}] must be a mapping")
        item_id = _require_str(item, "id", display=f"{field}[{index}].id")
        if item_id in ids:
            raise MatrixValidationError(f"duplicate {field} id: {item_id}")
        ids.add(item_id)
        _require_str(item, "purpose", display=f"{field}[{index}].purpose")
    return ids


def _validate_rows(
    rows: list[Any], *, frontend_ids: set[str], dataset_ids: set[str], shift_axis_ids: set[str]
) -> None:
    row_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MatrixValidationError(f"rows[{index}] must be a mapping")
        row_id = _require_str(row, "row_id", display=f"rows[{index}].row_id")
        if row_id in row_ids:
            raise MatrixValidationError(f"duplicate row id: {row_id}")
        row_ids.add(row_id)
        kind = _require_str(row, "kind", display=f"rows[{index}].kind")
        if kind not in ROW_KINDS:
            raise MatrixValidationError(f"rows[{index}].kind must be one of {sorted(ROW_KINDS)}")
        status = _require_str(row, "status", display=f"rows[{index}].status")
        if status not in STATUSES:
            raise MatrixValidationError(f"rows[{index}].status must be one of {sorted(STATUSES)}")
        dataset = _require_str(row, "dataset", display=f"rows[{index}].dataset")
        if dataset not in dataset_ids:
            raise MatrixValidationError(
                f"rows[{index}].dataset references unknown dataset: {dataset}"
            )
        shift_axis = _require_str(row, "shift_axis", display=f"rows[{index}].shift_axis")
        if shift_axis not in shift_axis_ids:
            raise MatrixValidationError(
                f"rows[{index}].shift_axis references unknown shift axis: {shift_axis}"
            )
        row_frontends = _require_list(row, "frontends", display=f"rows[{index}].frontends")
        if not row_frontends:
            raise MatrixValidationError(f"rows[{index}].frontends must not be empty")
        for frontend in row_frontends:
            if frontend not in frontend_ids:
                raise MatrixValidationError(
                    f"rows[{index}].frontends references unknown frontend: {frontend}"
                )
        _require_str(row, "question", display=f"rows[{index}].question")
        _require_list(row, "metrics", display=f"rows[{index}].metrics")
        if not isinstance(row.get("approval_required"), bool):
            raise MatrixValidationError(f"rows[{index}].approval_required must be boolean")


def _require_mapping(
    data: dict[str, Any], key: str, *, display: str | None = None
) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise MatrixValidationError(f"{display or key} must be a mapping")
    return value


def _require_list(data: dict[str, Any], key: str, *, display: str | None = None) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise MatrixValidationError(f"{display or key} must be a list")
    return value


def _require_str(data: dict[str, Any], key: str, *, display: str | None = None) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MatrixValidationError(f"{display or key} must be a non-empty string")
    return value


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[field])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
