"""Structured provenance for audited paper tables.

This module is deliberately small and dependency-light. It turns the audited
YAML ledger into validated Python objects, then leaves presentation to table
code. Missing artifacts are valid data; silent seed-set mismatches are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ProvenanceError(ValueError):
    """Raised when provenance data is malformed or scientifically unsafe."""


@dataclass(frozen=True)
class SeedMetrics:
    la_eer: float | None = None
    df_eer: float | None = None
    la_tdcf: float | None = None


@dataclass(frozen=True)
class SeedResult:
    id: str
    source: str
    status: str
    metrics: SeedMetrics
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TableRow:
    id: str
    model: str
    strategy: str
    n_source: str
    status: str
    notes: str
    seeds: tuple[SeedResult, ...]
    tdcf_note: str | None = None
    tdcf_marker: str = ""
    exploratory: bool = False

    def values(self, metric: str) -> list[float]:
        vals: list[float] = []
        for seed in self.seeds:
            value = getattr(seed.metrics, metric)
            if value is not None:
                vals.append(float(value))
        return vals

    @property
    def la_eer_values(self) -> list[float]:
        return self.values("la_eer")

    @property
    def df_eer_values(self) -> list[float]:
        return self.values("df_eer")

    @property
    def la_tdcf_values(self) -> list[float]:
        return self.values("la_tdcf")


@dataclass(frozen=True)
class MainTableProvenance:
    version: int
    policy: dict[str, Any]
    rows: tuple[TableRow, ...]

    def row(self, row_id: str) -> TableRow:
        for row in self.rows:
            if row.id == row_id:
                return row
        raise KeyError(row_id)


def load_main_table_provenance(path: str | Path) -> MainTableProvenance:
    """Load and validate a main-table provenance YAML file."""

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ProvenanceError("provenance root must be a mapping")

    rows_data = data.get("rows")
    if not isinstance(rows_data, list) or not rows_data:
        raise ProvenanceError("provenance must contain non-empty 'rows'")

    rows = tuple(_parse_row(item) for item in rows_data)
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ProvenanceError("row ids must be unique")

    provenance = MainTableProvenance(
        version=int(data.get("version", 0)),
        policy=dict(data.get("policy") or {}),
        rows=rows,
    )
    validate_main_table_provenance(provenance)
    return provenance


def validate_main_table_provenance(provenance: MainTableProvenance) -> None:
    """Validate cross-row scientific invariants."""

    if provenance.version != 1:
        raise ProvenanceError(f"unsupported provenance version: {provenance.version}")

    for row in provenance.rows:
        if not row.la_eer_values:
            raise ProvenanceError(f"{row.id}: missing LA EER values")
        if not row.df_eer_values:
            raise ProvenanceError(f"{row.id}: missing DF EER values")
        if len(row.la_eer_values) != len(row.df_eer_values):
            raise ProvenanceError(f"{row.id}: LA/DF EER seed counts differ")

        if row.la_tdcf_values and len(row.la_tdcf_values) != len(row.la_eer_values):
            if not row.tdcf_note:
                raise ProvenanceError(f"{row.id}: tDCF seed count differs from EER; add tdcf_note")

        if row.exploratory and "exploratory" not in row.status.lower():
            raise ProvenanceError(f"{row.id}: exploratory rows must say so in status")


def _parse_row(data: Any) -> TableRow:
    if not isinstance(data, dict):
        raise ProvenanceError("each row must be a mapping")

    required = ["id", "model", "strategy", "n_source", "status", "notes", "seeds"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ProvenanceError(f"row missing keys: {', '.join(missing)}")

    seeds_data = data["seeds"]
    if not isinstance(seeds_data, list) or not seeds_data:
        raise ProvenanceError(f"{data['id']}: seeds must be non-empty list")

    return TableRow(
        id=str(data["id"]),
        model=str(data["model"]),
        strategy=str(data["strategy"]),
        n_source=str(data["n_source"]),
        status=str(data["status"]),
        notes=str(data["notes"]),
        seeds=tuple(_parse_seed(data["id"], seed) for seed in seeds_data),
        tdcf_note=_optional_str(data.get("tdcf_note")),
        tdcf_marker=str(data.get("tdcf_marker") or ""),
        exploratory=bool(data.get("exploratory", False)),
    )


def _parse_seed(row_id: str, data: Any) -> SeedResult:
    if not isinstance(data, dict):
        raise ProvenanceError(f"{row_id}: each seed must be a mapping")

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        raise ProvenanceError(f"{row_id}: seed {data.get('id')} missing metrics")

    return SeedResult(
        id=str(data.get("id")),
        source=str(data.get("source")),
        status=str(data.get("status")),
        metrics=SeedMetrics(
            la_eer=_optional_float(metrics.get("la_eer")),
            df_eer=_optional_float(metrics.get("df_eer")),
            la_tdcf=_optional_float(metrics.get("la_tdcf")),
        ),
        artifacts={str(k): str(v) for k, v in dict(data.get("artifacts") or {}).items()},
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
