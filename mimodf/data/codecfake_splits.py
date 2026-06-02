"""Deterministic CodecFake+ source-holdout split planning."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("bonafide", "spoof")


@dataclass(frozen=True)
class SourceHoldoutFold:
    heldout_source: str
    validation_sources: tuple[str, ...]
    train_sources: tuple[str, ...]
    heldout_records: int
    validation_records: int
    train_records: int
    heldout_labels: dict[str, int]
    validation_labels: dict[str, int]
    train_labels: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "heldout_source": self.heldout_source,
            "validation_sources": list(self.validation_sources),
            "train_sources": list(self.train_sources),
            "heldout_records": self.heldout_records,
            "validation_records": self.validation_records,
            "train_records": self.train_records,
            "heldout_labels": self.heldout_labels,
            "validation_labels": self.validation_labels,
            "train_labels": self.train_labels,
        }


@dataclass(frozen=True)
class SourceHoldoutRows:
    heldout_source: str
    train_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]
    test_rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "heldout_source": self.heldout_source,
            "train_rows": list(self.train_rows),
            "validation_rows": list(self.validation_rows),
            "test_rows": list(self.test_rows),
        }


@dataclass(frozen=True)
class SourceHoldoutPlan:
    protocol: str
    subset: str
    seed: int
    min_per_label: int
    validation_source_count: int
    validation_policy: str
    validation_fraction: float
    total_records: int
    eligible_sources: tuple[str, ...]
    source_labels: dict[str, dict[str, int]]
    folds: tuple[SourceHoldoutFold, ...]
    skipped_missing_audio: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "subset": self.subset,
            "seed": self.seed,
            "min_per_label": self.min_per_label,
            "validation_source_count": self.validation_source_count,
            "validation_policy": self.validation_policy,
            "validation_fraction": self.validation_fraction,
            "total_records": self.total_records,
            "eligible_sources": list(self.eligible_sources),
            "source_labels": self.source_labels,
            "folds": [fold.to_dict() for fold in self.folds],
            "skipped_missing_audio": self.skipped_missing_audio,
        }


def build_source_holdout_plan(
    *,
    protocol: str | Path,
    subset: str = "CoSG",
    min_per_label: int = 10,
    validation_source_count: int = 1,
    validation_policy: str = "source",
    validation_fraction: float = 0.15,
    seed: int = 42,
    require_audio: bool = True,
) -> SourceHoldoutPlan:
    """Build deterministic source-holdout fold counts without writing row splits."""

    if min_per_label <= 0:
        raise ValueError("min_per_label must be positive")
    if validation_source_count <= 0:
        raise ValueError("validation_source_count must be positive")
    if validation_policy not in {"source", "stratified-row"}:
        raise ValueError("validation_policy must be 'source' or 'stratified-row'")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be > 0 and < 0.5")

    protocol_path = Path(protocol)
    rows_by_source, skipped_missing_audio = _load_rows_by_source(
        protocol_path, subset=subset, require_audio=require_audio
    )
    total_records = sum(len(rows) for rows in rows_by_source.values())

    if not rows_by_source:
        raise ValueError("no eligible rows found")

    source_labels = {source: _label_counts(rows) for source, rows in sorted(rows_by_source.items())}
    eligible_sources = tuple(
        source
        for source, counts in source_labels.items()
        if all(counts.get(label, 0) >= min_per_label for label in LABELS)
    )
    if len(eligible_sources) < validation_source_count + 2:
        raise ValueError(
            "not enough eligible sources for holdout plus validation plus training: "
            f"eligible={len(eligible_sources)}, validation_source_count={validation_source_count}"
        )

    folds: list[SourceHoldoutFold] = []
    for heldout in eligible_sources:
        heldout_rows = rows_by_source[heldout]
        if validation_policy == "source":
            validation_sources = _select_validation_sources(
                heldout=heldout,
                candidates=tuple(source for source in eligible_sources if source != heldout),
                source_sizes={source: len(rows) for source, rows in rows_by_source.items()},
                count=validation_source_count,
                seed=seed,
            )
            train_sources = tuple(
                source
                for source in sorted(rows_by_source)
                if source != heldout and source not in validation_sources
            )
            validation_rows = _rows_for_sources(rows_by_source, validation_sources)
            train_rows = _rows_for_sources(rows_by_source, train_sources)
        else:
            validation_sources = ()
            train_sources = tuple(source for source in sorted(rows_by_source) if source != heldout)
            train_rows, validation_rows = _split_train_validation_rows(
                _rows_for_sources(rows_by_source, train_sources),
                heldout=heldout,
                seed=seed,
                validation_fraction=validation_fraction,
            )
        _require_both_labels(train_rows, f"train fold for {heldout}")
        _require_both_labels(validation_rows, f"validation fold for {heldout}")
        _require_both_labels(heldout_rows, f"heldout fold for {heldout}")
        folds.append(
            SourceHoldoutFold(
                heldout_source=heldout,
                validation_sources=validation_sources,
                train_sources=train_sources,
                heldout_records=len(heldout_rows),
                validation_records=len(validation_rows),
                train_records=len(train_rows),
                heldout_labels=_label_counts(heldout_rows),
                validation_labels=_label_counts(validation_rows),
                train_labels=_label_counts(train_rows),
            )
        )

    return SourceHoldoutPlan(
        protocol=str(protocol_path),
        subset=subset,
        seed=seed,
        min_per_label=min_per_label,
        validation_source_count=validation_source_count,
        validation_policy=validation_policy,
        validation_fraction=validation_fraction,
        total_records=total_records,
        eligible_sources=eligible_sources,
        source_labels=source_labels,
        folds=tuple(folds),
        skipped_missing_audio=skipped_missing_audio,
    )


def build_source_holdout_rows(
    *,
    protocol: str | Path,
    heldout_source: str,
    subset: str = "CoSG",
    validation_policy: str = "source",
    validation_source_count: int = 1,
    validation_fraction: float = 0.15,
    seed: int = 42,
    require_audio: bool = True,
) -> SourceHoldoutRows:
    """Materialize train/validation/test rows for one source-holdout fold."""

    if validation_policy not in {"source", "stratified-row"}:
        raise ValueError("validation_policy must be 'source' or 'stratified-row'")
    rows_by_source, _ = _load_rows_by_source(
        Path(protocol), subset=subset, require_audio=require_audio
    )
    if heldout_source not in rows_by_source:
        raise ValueError(f"heldout source not found: {heldout_source}")

    heldout_rows = rows_by_source[heldout_source]
    if validation_policy == "source":
        candidates = tuple(source for source in sorted(rows_by_source) if source != heldout_source)
        validation_sources = _select_validation_sources(
            heldout=heldout_source,
            candidates=candidates,
            source_sizes={source: len(rows) for source, rows in rows_by_source.items()},
            count=validation_source_count,
            seed=seed,
        )
        train_sources = tuple(
            source
            for source in sorted(rows_by_source)
            if source != heldout_source and source not in validation_sources
        )
        train_rows = _rows_for_sources(rows_by_source, train_sources)
        validation_rows = _rows_for_sources(rows_by_source, validation_sources)
    else:
        train_sources = tuple(
            source for source in sorted(rows_by_source) if source != heldout_source
        )
        train_rows, validation_rows = _split_train_validation_rows(
            _rows_for_sources(rows_by_source, train_sources),
            heldout=heldout_source,
            seed=seed,
            validation_fraction=validation_fraction,
        )

    _require_both_labels(train_rows, f"train fold for {heldout_source}")
    _require_both_labels(validation_rows, f"validation fold for {heldout_source}")
    _require_both_labels(heldout_rows, f"heldout fold for {heldout_source}")
    return SourceHoldoutRows(
        heldout_source=heldout_source,
        train_rows=tuple(train_rows),
        validation_rows=tuple(validation_rows),
        test_rows=tuple(heldout_rows),
    )


def render_source_holdout_plan_json(plan: SourceHoldoutPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"


def render_source_holdout_plan_markdown(plan: SourceHoldoutPlan) -> str:
    lines = [
        "# CodecFake+ source-holdout plan",
        "",
        f"- protocol: `{plan.protocol}`",
        f"- subset: `{plan.subset}`",
        f"- seed: `{plan.seed}`",
        f"- min per label: `{plan.min_per_label}`",
        f"- validation source count: `{plan.validation_source_count}`",
        f"- validation policy: `{plan.validation_policy}`",
        f"- validation fraction: `{plan.validation_fraction}`",
        f"- total records: `{plan.total_records}`",
        f"- eligible sources: `{len(plan.eligible_sources)}`",
        f"- skipped missing audio: `{plan.skipped_missing_audio}`",
        "",
        "## Folds",
        "",
        "| Held-out | Validation sources | Train records | Validation records | Held-out records | Held-out labels |",
        "|---|---|---:|---:|---:|---|",
    ]
    for fold in plan.folds:
        lines.append(
            "| "
            f"`{fold.heldout_source}` | "
            f"`{', '.join(fold.validation_sources)}` | "
            f"{fold.train_records} | {fold.validation_records} | {fold.heldout_records} | "
            f"`{json.dumps(fold.heldout_labels, sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is a split/count plan only; it does not train or score models.",
            "- Held-out sources are never used as validation sources in their fold.",
            "- Validation source choice is deterministic from seed and held-out source.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_rows_by_source(
    protocol_path: Path, *, subset: str, require_audio: bool
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_missing_audio = 0
    with protocol_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            if row.get("subset") != subset:
                continue
            source = row.get("source_model")
            label = row.get("label")
            if not isinstance(source, str) or not source:
                raise ValueError(f"{protocol_path}:{line_number}: source_model is required")
            if label not in LABELS:
                raise ValueError(f"{protocol_path}:{line_number}: unsupported label {label!r}")
            audio_path = row.get("audio_path")
            if require_audio and (
                not isinstance(audio_path, str) or not Path(audio_path).is_file()
            ):
                skipped_missing_audio += 1
                continue
            rows_by_source[source].append(row)
    return rows_by_source, skipped_missing_audio


def _split_train_validation_rows(
    rows: list[dict[str, Any]],
    *,
    heldout: str,
    seed: int,
    validation_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source_model"]), str(row["label"]))].append(row)

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for (source, label), group_rows in sorted(groups.items()):
        ranked = sorted(
            group_rows,
            key=lambda row: _stable_rank(seed, heldout, f"{source}:{label}:{row['utterance_id']}"),
        )
        if len(ranked) < 2:
            train_rows.extend(ranked)
            continue
        validation_count = min(len(ranked) - 1, max(1, int(len(ranked) * validation_fraction)))
        validation_rows.extend(ranked[:validation_count])
        train_rows.extend(ranked[validation_count:])
    return train_rows, validation_rows


def _select_validation_sources(
    *,
    heldout: str,
    candidates: tuple[str, ...],
    source_sizes: dict[str, int],
    count: int,
    seed: int,
) -> tuple[str, ...]:
    ranked = sorted(
        candidates,
        key=lambda source: (source_sizes[source], _stable_rank(seed, heldout, source)),
    )
    selected = tuple(sorted(ranked[:count]))
    if len(selected) != count:
        raise ValueError(f"could not select {count} validation sources for {heldout}")
    return selected


def _stable_rank(seed: int, heldout: str, source: str) -> str:
    return hashlib.sha256(f"{seed}:{heldout}:{source}".encode()).hexdigest()


def _rows_for_sources(
    rows_by_source: dict[str, list[dict[str, Any]]], sources: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        rows.extend(rows_by_source[source])
    return rows


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["label"]) for row in rows)
    return {label: counts.get(label, 0) for label in LABELS}


def _require_both_labels(rows: list[dict[str, Any]], name: str) -> None:
    counts = _label_counts(rows)
    missing = [label for label in LABELS if counts.get(label, 0) <= 0]
    if missing:
        raise ValueError(f"{name} is missing labels: {', '.join(missing)}")
