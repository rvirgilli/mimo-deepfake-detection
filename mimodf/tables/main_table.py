"""Generate the audited main-results table from structured provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from mimodf.provenance import MainTableProvenance, TableRow, load_main_table_provenance


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mean: float
    sample_std: float | None = None

    @classmethod
    def from_values(cls, values: list[float], *, with_std: bool) -> MetricSummary:
        if not values:
            raise ValueError("cannot summarize empty values")
        return cls(
            n=len(values),
            mean=mean(values),
            sample_std=stdev(values) if with_std and len(values) > 1 else None,
        )


def summarize_eer(values: list[float]) -> MetricSummary:
    """Summarize EER with sample standard deviation."""

    return MetricSummary.from_values(values, with_std=True)


def summarize_tdcf(values: list[float]) -> MetricSummary | None:
    """Summarize tDCF as mean only for the paper table."""

    if not values:
        return None
    return MetricSummary.from_values(values, with_std=False)


def render_main_table(provenance: MainTableProvenance) -> str:
    """Render the recommended artifact-backed table as Markdown."""

    lines = [
        "| Model | Strategy | n/source | LA EER (%) | LA min-tDCF | DF EER (%) | Status | Notes |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in provenance.rows:
        lines.append(_render_row(row))
    return "\n".join(lines) + "\n"


def render_main_table_from_file(path: str | Path) -> str:
    return render_main_table(load_main_table_provenance(path))


def _render_row(row: TableRow) -> str:
    la_eer = summarize_eer(row.la_eer_values)
    df_eer = summarize_eer(row.df_eer_values)
    la_tdcf = summarize_tdcf(row.la_tdcf_values)

    tdcf_text = "---"
    if la_tdcf is not None:
        tdcf_text = f"{la_tdcf.mean:.3f}{row.tdcf_marker}"

    return (
        f"| {row.model} | {row.strategy} | {row.n_source} | "
        f"{_format_eer(la_eer)} | {tdcf_text} | {_format_eer(df_eer)} | "
        f"{row.status} | {row.notes} |"
    )


def _format_eer(summary: MetricSummary) -> str:
    if summary.sample_std is None:
        return f"{summary.mean:.2f}"
    return f"{summary.mean:.2f} ± {summary.sample_std:.2f}"
