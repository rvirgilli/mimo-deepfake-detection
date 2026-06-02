"""Small JSONL protocol sampling helpers for bounded smoke runs."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProtocolSampleSettings:
    input_path: Path
    out_path: Path
    group_by: tuple[str, ...]
    max_per_group: int
    max_records: int | None = None
    seed: int = 42
    require_audio: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ProtocolSampleSummary:
    input_path: str
    output_path: str
    records_read: int
    records_eligible: int
    records_written: int
    skipped_missing_audio: int
    group_by: tuple[str, ...]
    max_per_group: int
    max_records: int | None
    seed: int
    require_audio: bool
    labels: dict[str, int]
    source_models: dict[str, int]
    groups_input: int
    groups_written: int
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["group_by"] = list(self.group_by)
        data["caveats"] = list(self.caveats)
        return data


def sample_protocol(settings: ProtocolSampleSettings) -> ProtocolSampleSummary:
    _validate_settings(settings)
    rows, skipped_missing_audio = _load_rows(
        settings.input_path, require_audio=settings.require_audio
    )
    grouped = _group_rows(rows, settings.group_by)
    sampled = _sample_groups(
        grouped, settings.max_per_group, settings.max_records, seed=settings.seed
    )

    if settings.out_path.exists() and not settings.overwrite:
        raise FileExistsError(settings.out_path)
    settings.out_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.out_path.open("w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    return _summary(settings, rows, sampled, skipped_missing_audio, grouped)


def render_protocol_sample_json(summary: ProtocolSampleSummary) -> str:
    return json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n"


def render_protocol_sample_markdown(summary: ProtocolSampleSummary) -> str:
    lines = [
        "# Protocol sample summary",
        "",
        f"- input: `{summary.input_path}`",
        f"- output: `{summary.output_path}`",
        f"- records read: {summary.records_read}",
        f"- records eligible: {summary.records_eligible}",
        f"- records written: {summary.records_written}",
        f"- skipped missing audio: {summary.skipped_missing_audio}",
        f"- group by: `{', '.join(summary.group_by)}`",
        f"- max per group: {summary.max_per_group}",
        f"- max records: {summary.max_records}",
        f"- seed: {summary.seed}",
        "",
        "## Labels",
        *_bullet_counts(summary.labels),
        "",
        "## Source models",
        *_bullet_counts(summary.source_models),
        "",
        "## Caveats",
        *(f"- {item}" for item in summary.caveats),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _validate_settings(settings: ProtocolSampleSettings) -> None:
    if not settings.group_by:
        raise ValueError("group_by must not be empty")
    if any(not item.strip() for item in settings.group_by):
        raise ValueError("group_by fields must be non-empty")
    if settings.max_per_group <= 0:
        raise ValueError("max_per_group must be positive")
    if settings.max_records is not None and settings.max_records <= 0:
        raise ValueError("max_records must be positive")
    if not settings.input_path.is_file():
        raise FileNotFoundError(settings.input_path)


def _load_rows(path: Path, *, require_audio: bool) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_missing_audio = 0
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            audio_path = row.get("audio_path")
            if require_audio and (not audio_path or not Path(str(audio_path)).is_file()):
                skipped_missing_audio += 1
                continue
            rows.append(row)
    return rows, skipped_missing_audio


def _group_rows(
    rows: list[dict[str, Any]], group_by: tuple[str, ...]
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(_field_value(row, field) for field in group_by)
        grouped.setdefault(key, []).append(row)
    return grouped


def _sample_groups(
    grouped: dict[tuple[str, ...], list[dict[str, Any]]],
    max_per_group: int,
    max_records: int | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    keys = sorted(grouped)
    rng.shuffle(keys)
    per_group: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for key in keys:
        rows = list(grouped[key])
        rng.shuffle(rows)
        per_group[key] = rows[:max_per_group]

    output: list[dict[str, Any]] = []
    for depth in range(max_per_group):
        for key in keys:
            rows = per_group[key]
            if depth < len(rows):
                output.append(rows[depth])
                if max_records is not None and len(output) >= max_records:
                    return output
    return output


def _summary(
    settings: ProtocolSampleSettings,
    eligible_rows: list[dict[str, Any]],
    sampled: list[dict[str, Any]],
    skipped_missing_audio: int,
    grouped: dict[tuple[str, ...], list[dict[str, Any]]],
) -> ProtocolSampleSummary:
    return ProtocolSampleSummary(
        input_path=str(settings.input_path),
        output_path=str(settings.out_path),
        records_read=len(eligible_rows) + skipped_missing_audio,
        records_eligible=len(eligible_rows),
        records_written=len(sampled),
        skipped_missing_audio=skipped_missing_audio,
        group_by=settings.group_by,
        max_per_group=settings.max_per_group,
        max_records=settings.max_records,
        seed=settings.seed,
        require_audio=settings.require_audio,
        labels=_counter(sampled, "label"),
        source_models=_counter(sampled, "source_model"),
        groups_input=len(grouped),
        groups_written=len(
            {tuple(_field_value(row, field) for field in settings.group_by) for row in sampled}
        ),
        caveats=(
            "smoke protocol only; not a train/test split",
            "sampling is deterministic for the declared seed and input order",
            "rows without existing audio_path are excluded when require_audio is true",
        ),
    )


def _field_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        return "<missing>"
    return str(value)


def _counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(_field_value(row, field) for row in rows).items()))


def _bullet_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- `{key}`: {value}" for key, value in counts.items()]
