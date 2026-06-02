"""Strict comparison checks for indexed experiment records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from mimodf.report.index import RunIndexRecord


@dataclass(frozen=True)
class ComparisonCheck:
    name: str
    passed: bool
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonReport:
    experiments: list[str]
    strict: bool
    passed: bool
    checks: list[ComparisonCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiments": self.experiments,
            "strict": self.strict,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


def compare_experiments(
    records: Iterable[RunIndexRecord],
    experiment_ids: list[str],
    *,
    strict: bool = False,
) -> ComparisonReport:
    if len(experiment_ids) < 2:
        raise ValueError("compare requires at least two experiment ids")
    grouped = _select(records, experiment_ids)
    checks: list[ComparisonCheck] = []
    checks.extend(_existence_checks(grouped, experiment_ids))
    existing = {experiment_id: items for experiment_id, items in grouped.items() if items}
    if len(existing) >= 2:
        checks.append(_seed_set_check(existing, strict=strict))
        checks.append(_protocol_check(existing, strict=strict))
        checks.append(_intent_check(existing, strict=strict))
        checks.append(_batch_sensitive_check(existing, strict=strict))
    passed = all(check.passed or check.severity != "error" for check in checks)
    return ComparisonReport(
        experiments=experiment_ids,
        strict=strict,
        passed=passed,
        checks=checks,
    )


def render_comparison_json(report: ComparisonReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_comparison_markdown(report: ComparisonReport) -> str:
    verdict = "passed" if report.passed else "failed"
    lines = [
        f"# Comparison report: {verdict}",
        "",
        f"Experiments: {', '.join(report.experiments)}",
        f"Strict: {str(report.strict).lower()}",
        "",
        "| Check | Severity | Passed | Message |",
        "|---|---|---:|---|",
    ]
    for check in report.checks:
        lines.append(f"| {check.name} | {check.severity} | {check.passed} | {check.message} |")
    return "\n".join(lines) + "\n"


def _select(
    records: Iterable[RunIndexRecord],
    experiment_ids: list[str],
) -> dict[str, list[RunIndexRecord]]:
    selected = {experiment_id: [] for experiment_id in experiment_ids}
    for record in records:
        if record.experiment_id in selected:
            selected[record.experiment_id].append(record)
    return selected


def _existence_checks(
    grouped: dict[str, list[RunIndexRecord]],
    experiment_ids: list[str],
) -> list[ComparisonCheck]:
    checks: list[ComparisonCheck] = []
    for experiment_id in experiment_ids:
        count = len(grouped.get(experiment_id, []))
        checks.append(
            ComparisonCheck(
                name=f"records:{experiment_id}",
                passed=count > 0,
                severity="error",
                message=f"{count} record(s) found",
            )
        )
    return checks


def _seed_set_check(
    grouped: dict[str, list[RunIndexRecord]],
    *,
    strict: bool,
) -> ComparisonCheck:
    seed_sets = {
        experiment_id: {record.seed for record in records}
        for experiment_id, records in grouped.items()
    }
    unique = {_freeze(seed_set) for seed_set in seed_sets.values()}
    passed = len(unique) == 1
    return ComparisonCheck(
        name="same_seed_set",
        passed=passed,
        severity="error" if strict else "warning",
        message="; ".join(
            f"{key}={sorted(value, key=str)}" for key, value in sorted(seed_sets.items())
        ),
    )


def _protocol_check(
    grouped: dict[str, list[RunIndexRecord]],
    *,
    strict: bool,
) -> ComparisonCheck:
    protocol_sets = {
        experiment_id: {_freeze_mapping(record.protocol_ids) for record in records}
        for experiment_id, records in grouped.items()
    }
    missing = [
        experiment_id
        for experiment_id, protocols in protocol_sets.items()
        if not protocols or protocols == {()}
    ]
    passed = not missing and len({_freeze(protocols) for protocols in protocol_sets.values()}) == 1
    message = "; ".join(
        f"{experiment_id}={_format_protocol_set(protocols)}"
        for experiment_id, protocols in sorted(protocol_sets.items())
    )
    if missing:
        message += f"; missing protocol ids for {missing}"
    return ComparisonCheck(
        name="same_protocol",
        passed=passed,
        severity="error" if strict else "warning",
        message=message,
    )


def _intent_check(
    grouped: dict[str, list[RunIndexRecord]],
    *,
    strict: bool,
) -> ComparisonCheck:
    intents = {record.intent for records in grouped.values() for record in records}
    exploratory = "exploratory" in intents
    passed = not (strict and exploratory)
    return ComparisonCheck(
        name="confirmatory_intent",
        passed=passed,
        severity="error" if strict else "warning",
        message=f"intents={sorted(intents)}",
    )


def _batch_sensitive_check(
    grouped: dict[str, list[RunIndexRecord]],
    *,
    strict: bool,
) -> ComparisonCheck:
    sensitive_experiments = []
    missing_batch = []
    for experiment_id, records in grouped.items():
        sensitive = any(
            any("batch_size_sensitive" in warning for warning in record.warnings)
            for record in records
        )
        if sensitive:
            sensitive_experiments.append(experiment_id)
            if any("evaluation_batch_size" not in record.protocol_ids for record in records):
                missing_batch.append(experiment_id)
    passed = not missing_batch
    return ComparisonCheck(
        name="batch_size_policy",
        passed=passed,
        severity="error" if strict else "warning",
        message=(
            f"batch-sensitive experiments={sensitive_experiments}; "
            f"missing evaluation_batch_size={missing_batch}"
        ),
    )


def _freeze(values: set[Any]) -> tuple[Any, ...]:
    return tuple(sorted(values, key=str))


def _freeze_mapping(mapping: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(mapping.items()))


def _format_protocol_set(protocols: set[tuple[tuple[str, str], ...]]) -> str:
    if not protocols:
        return "[]"
    rendered = []
    for protocol in sorted(protocols):
        rendered.append("{" + ", ".join(f"{key}: {value}" for key, value in protocol) + "}")
    return "[" + ", ".join(rendered) + "]"
