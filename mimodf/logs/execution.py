"""Validation and lightweight summaries for the research execution log."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_RESEARCH_LOG = Path("docs/current/research_execution_log.jsonl")
VALID_STATUSES = {"planned", "completed", "failed", "interrupted"}
CORE_FIELDS = (
    "schema",
    "run_id",
    "wave",
    "kind",
    "status",
    "cwd",
    "environment",
    "command",
)
COMPLETED_FIELDS = ("outputs", "git_revision_at_run", "result_summary")
PLANNED_FIELDS = ("inputs", "planned_outputs", "git_revision_at_plan")


@dataclass(frozen=True)
class LogIssue:
    severity: str
    line: int | None
    run_id: str | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LogValidationResult:
    path: str
    rows: int
    errors: tuple[LogIssue, ...]
    warnings: tuple[LogIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "rows": self.rows,
            "passed": self.passed,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class LogSummary:
    path: str
    rows: int
    by_status: dict[str, int]
    by_wave: dict[str, int]
    by_kind: dict[str, int]
    missing_fields: dict[str, int]
    planned_run_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_log_records(path: str | Path = DEFAULT_RESEARCH_LOG) -> list[tuple[int, dict[str, Any]]]:
    log_path = Path(path)
    records: list[tuple[int, dict[str, Any]]] = []
    with log_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            records.append((line_number, data))
    return records


def validate_log(
    path: str | Path = DEFAULT_RESEARCH_LOG, *, strict: bool = False
) -> LogValidationResult:
    log_path = Path(path)
    errors: list[LogIssue] = []
    warnings: list[LogIssue] = []
    records: list[tuple[int, dict[str, Any]]] = []
    try:
        records = load_log_records(log_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return LogValidationResult(
            path=str(log_path),
            rows=0,
            errors=(LogIssue("error", None, None, str(exc)),),
            warnings=(),
        )

    seen: dict[str, int] = {}
    for line_number, record in records:
        run_id = _string_or_none(record.get("run_id"))
        _validate_record(record, line_number, run_id, errors, warnings, strict=strict)
        if run_id is None:
            continue
        if run_id in seen:
            errors.append(
                LogIssue(
                    "error",
                    line_number,
                    run_id,
                    f"duplicate run_id first seen on line {seen[run_id]}",
                )
            )
        else:
            seen[run_id] = line_number

    return LogValidationResult(
        path=str(log_path),
        rows=len(records),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def summarize_log(path: str | Path = DEFAULT_RESEARCH_LOG) -> LogSummary:
    log_path = Path(path)
    records = load_log_records(log_path)
    missing: Counter[str] = Counter()
    status: Counter[str] = Counter()
    wave: Counter[str] = Counter()
    kind: Counter[str] = Counter()
    planned: list[str] = []
    for _, record in records:
        status[str(record.get("status"))] += 1
        wave[str(record.get("wave"))] += 1
        kind[str(record.get("kind"))] += 1
        if record.get("status") == "planned" and isinstance(record.get("run_id"), str):
            planned.append(record["run_id"])
        for field in (*CORE_FIELDS, *PLANNED_FIELDS, *COMPLETED_FIELDS):
            if field not in record:
                missing[field] += 1
    return LogSummary(
        path=str(log_path),
        rows=len(records),
        by_status=_sorted_counts(status),
        by_wave=_sorted_counts(wave),
        by_kind=_sorted_counts(kind),
        missing_fields=_sorted_counts(missing),
        planned_run_ids=tuple(sorted(planned)),
    )


def render_validation_text(result: LogValidationResult) -> str:
    lines = [
        "# Research execution log validation",
        "",
        f"Path: `{result.path}`",
        f"Rows: {result.rows}",
        f"Status: **{'pass' if result.passed else 'fail'}**",
        "",
    ]
    lines.extend(_issue_section("Errors", result.errors))
    lines.extend(_issue_section("Warnings", result.warnings))
    return "\n".join(lines).rstrip() + "\n"


def render_summary_markdown(summary: LogSummary) -> str:
    lines = [
        "# Research execution log summary",
        "",
        f"Path: `{summary.path}`",
        f"Rows: {summary.rows}",
        "",
        "## By status",
        "",
        _table(summary.by_status),
        "",
        "## By wave",
        "",
        _table(summary.by_wave),
        "",
        "## By kind",
        "",
        _table(summary.by_kind),
        "",
        "## Missing fields",
        "",
        _table(summary.missing_fields),
        "",
        "## Planned runs",
        "",
    ]
    if summary.planned_run_ids:
        lines.extend(f"- `{run_id}`" for run_id in summary.planned_run_ids)
    else:
        lines.append("None.")
    return "\n".join(lines).rstrip() + "\n"


def _validate_record(
    record: dict[str, Any],
    line_number: int,
    run_id: str | None,
    errors: list[LogIssue],
    warnings: list[LogIssue],
    *,
    strict: bool,
) -> None:
    for field in CORE_FIELDS:
        if field not in record:
            errors.append(LogIssue("error", line_number, run_id, f"missing core field: {field}"))
    status = record.get("status")
    if status not in VALID_STATUSES:
        errors.append(LogIssue("error", line_number, run_id, f"invalid status: {status}"))
    if status == "planned":
        _add_missing_fields(
            record, line_number, run_id, PLANNED_FIELDS, errors, warnings, strict=strict
        )
        _add_issue(
            line_number, run_id, "planned row remains in log", errors, warnings, strict=strict
        )
    if status in {"completed", "failed", "interrupted"}:
        _add_missing_fields(
            record, line_number, run_id, COMPLETED_FIELDS, errors, warnings, strict=strict
        )
    if "outputs" in record and not isinstance(record["outputs"], list):
        errors.append(LogIssue("error", line_number, run_id, "outputs must be a list"))
    if "inputs" in record and not isinstance(record["inputs"], list):
        errors.append(LogIssue("error", line_number, run_id, "inputs must be a list"))


def _add_missing_fields(
    record: dict[str, Any],
    line_number: int,
    run_id: str | None,
    fields: tuple[str, ...],
    errors: list[LogIssue],
    warnings: list[LogIssue],
    *,
    strict: bool,
) -> None:
    for field in fields:
        if field not in record:
            _add_issue(
                line_number, run_id, f"missing field: {field}", errors, warnings, strict=strict
            )


def _add_issue(
    line_number: int,
    run_id: str | None,
    message: str,
    errors: list[LogIssue],
    warnings: list[LogIssue],
    *,
    strict: bool,
) -> None:
    issue = LogIssue("error" if strict else "warning", line_number, run_id, message)
    if strict:
        errors.append(issue)
    else:
        warnings.append(issue)


def _issue_section(title: str, issues: tuple[LogIssue, ...]) -> list[str]:
    lines = [f"## {title}", ""]
    if not issues:
        return [*lines, "None.", ""]
    for issue in issues:
        location = "unknown" if issue.line is None else f"line {issue.line}"
        run_id = "unknown" if issue.run_id is None else issue.run_id
        lines.append(f"- {location} `{run_id}`: {issue.message}")
    lines.append("")
    return lines


def _table(counts: dict[str, int]) -> str:
    if not counts:
        return "None."
    lines = ["| Value | Count |", "|---|---:|"]
    for value, count in counts.items():
        lines.append(f"| `{value}` | {count} |")
    return "\n".join(lines)


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
