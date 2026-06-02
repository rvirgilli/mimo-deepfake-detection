"""Aggregate numeric metrics from run-index records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import mean, stdev
from typing import Any

from mimodf.report.index import RunIndexRecord


@dataclass(frozen=True)
class MetricAggregate:
    metric: str
    n: int
    mean: float
    sample_std: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentAggregate:
    experiment_id: str
    seeds: list[int | str]
    statuses: dict[str, int]
    intents: dict[str, int]
    source_types: dict[str, int]
    min_reproducibility_tier: int
    metrics: list[MetricAggregate]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = [metric.to_dict() for metric in self.metrics]
        return data


def aggregate_records(records: Iterable[RunIndexRecord]) -> list[ExperimentAggregate]:
    grouped: dict[str, list[RunIndexRecord]] = {}
    for record in records:
        grouped.setdefault(record.experiment_id, []).append(record)
    return [
        aggregate_experiment(experiment_id, grouped[experiment_id])
        for experiment_id in sorted(grouped)
    ]


def aggregate_experiment(
    experiment_id: str,
    records: Iterable[RunIndexRecord],
) -> ExperimentAggregate:
    items = list(records)
    if not items:
        raise ValueError("cannot aggregate empty record set")
    metric_names = sorted(
        {name for record in items for name, value in record.metrics.items() if _numeric(value)}
    )
    metrics = [_aggregate_metric(name, items) for name in metric_names]
    return ExperimentAggregate(
        experiment_id=experiment_id,
        seeds=sorted((record.seed for record in items), key=str),
        statuses=_counts(record.status for record in items),
        intents=_counts(record.intent for record in items),
        source_types=_counts(record.source_type for record in items),
        min_reproducibility_tier=min(record.reproducibility_tier for record in items),
        metrics=metrics,
        warnings=sorted({warning for record in items for warning in record.warnings}),
    )


def render_aggregates_json(aggregates: Iterable[ExperimentAggregate]) -> str:
    return (
        json.dumps([aggregate.to_dict() for aggregate in aggregates], indent=2, sort_keys=True)
        + "\n"
    )


def render_aggregates_markdown(aggregates: Iterable[ExperimentAggregate]) -> str:
    lines = [
        "| Experiment | Seeds | Tier | Statuses | Intents | Sources | Metrics | Warnings |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for aggregate in aggregates:
        metric_text = ", ".join(_format_metric(metric) for metric in aggregate.metrics)
        lines.append(
            f"| {aggregate.experiment_id} | {len(aggregate.seeds)} | "
            f"{aggregate.min_reproducibility_tier} | {_format_counts(aggregate.statuses)} | "
            f"{_format_counts(aggregate.intents)} | {_format_counts(aggregate.source_types)} | "
            f"{metric_text or '---'} | {'; '.join(aggregate.warnings) or '---'} |"
        )
    return "\n".join(lines) + "\n"


def _aggregate_metric(metric: str, records: list[RunIndexRecord]) -> MetricAggregate:
    values = [
        float(record.metrics[metric]) for record in records if _numeric(record.metrics.get(metric))
    ]
    return MetricAggregate(
        metric=metric,
        n=len(values),
        mean=mean(values),
        sample_std=stdev(values) if len(values) > 1 else None,
    )


def _numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items()) or "---"


def _format_metric(metric: MetricAggregate) -> str:
    if metric.sample_std is None:
        return f"{metric.metric}={metric.mean:.4g} (n={metric.n})"
    return f"{metric.metric}={metric.mean:.4g} ± {metric.sample_std:.4g} (n={metric.n})"
