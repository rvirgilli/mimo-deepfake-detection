"""Summaries for audited official LA tDCF values."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


class TDCFValidationError(ValueError):
    """Raised when audited tDCF data is inconsistent."""


@dataclass(frozen=True)
class TDCFRowSummary:
    row_id: str
    n: int
    mean: float
    display: str
    status: str
    note: str


@dataclass(frozen=True)
class TDCFSummary:
    rows: list[TDCFRowSummary]
    wrong_scale_examples: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [asdict(row) for row in self.rows],
            "wrong_scale_examples": self.wrong_scale_examples,
        }


def load_tdcf_summary(path: str | Path) -> TDCFSummary:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise TDCFValidationError("tDCF summary root must be a mapping")
    rows = [_summarize_row(row) for row in data.get("rows", [])]
    if not rows:
        raise TDCFValidationError("tDCF summary must contain rows")
    return TDCFSummary(rows=rows, wrong_scale_examples=list(data.get("wrong_scale_examples", [])))


def render_tdcf_summary_markdown(summary: TDCFSummary) -> str:
    lines = [
        "| Row | n | Mean LA min-tDCF | Display | Status | Note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in summary.rows:
        lines.append(
            f"| {row.row_id} | {row.n} | {row.mean:.4f} | {row.display} | {row.status} | {row.note} |"
        )
    return "\n".join(lines) + "\n"


def render_tdcf_summary_json(summary: TDCFSummary) -> str:
    return json.dumps(summary.to_dict(), indent=2) + "\n"


def _summarize_row(row: Any) -> TDCFRowSummary:
    if not isinstance(row, dict):
        raise TDCFValidationError("each tDCF row must be a mapping")
    values = row.get("values")
    if not isinstance(values, list) or not values:
        raise TDCFValidationError(f"{row.get('row_id')}: missing values")
    tdcfs = [float(item["tdcf"]) for item in values]
    computed = mean(tdcfs)
    reported = float(row["reported_mean"])
    if abs(computed - reported) > 5e-5:
        raise TDCFValidationError(
            f"{row.get('row_id')}: reported_mean {reported:.4f} != computed {computed:.4f}"
        )
    return TDCFRowSummary(
        row_id=str(row["row_id"]),
        n=len(tdcfs),
        mean=computed,
        display=str(row["display"]),
        status=str(row["status"]),
        note=str(row.get("note", "")),
    )
