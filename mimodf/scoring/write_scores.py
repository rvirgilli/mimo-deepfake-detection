"""ASVspoof score-file writing primitives.

This is the model-independent contract for future evaluation code: inference code
produces utterance ids and scalar spoof scores; this module writes the exact
text format consumed by ASVspoof evaluators.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScoreRecord:
    utterance_id: str
    score: float


def write_score_file(
    records: Iterable[ScoreRecord],
    path: str | Path,
    *,
    sort_by_utterance_id: bool = True,
) -> Path:
    """Write ASVspoof-compatible score file.

    Format: one record per line, ``<utterance_id> <score>``.
    Duplicate utterance ids are rejected because they make scoring ambiguous.
    Sorting is enabled by default so output is deterministic even if dataloader
    order changes.
    """

    materialized = list(records)
    _validate_records(materialized)
    if sort_by_utterance_id:
        materialized = sorted(materialized, key=lambda item: item.utterance_id)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for record in materialized:
            f.write(f"{record.utterance_id} {_format_score(record.score)}\n")
    return out


def read_score_file(path: str | Path) -> list[ScoreRecord]:
    """Read a two-column ASVspoof score file."""

    records: list[ScoreRecord] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected '<utterance_id> <score>'")
        records.append(ScoreRecord(parts[0], float(parts[1])))
    _validate_records(records)
    return records


def _validate_records(records: list[ScoreRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if not record.utterance_id:
            raise ValueError("utterance_id must be non-empty")
        if record.utterance_id in seen:
            raise ValueError(f"duplicate utterance_id: {record.utterance_id}")
        seen.add(record.utterance_id)


def _format_score(score: float) -> str:
    # 10 significant digits is enough for evaluator stability without noisy text.
    return f"{float(score):.10g}"
