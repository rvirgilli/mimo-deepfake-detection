"""Minimal evaluation path for writing ASVspoof score files.

This module is intentionally framework-agnostic. Real Torch/MiMo code can sit
behind the predictor callable; tests can use fake predictors. The invariant we
care about here is the public contract: one score per utterance, no duplicate
ids, deterministic score-file output.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from mimodf.scoring.write_scores import ScoreRecord, write_score_file

T = TypeVar("T")


@dataclass(frozen=True)
class EvaluationItem[T]:
    utterance_id: str
    input: T


@dataclass(frozen=True)
class EvaluationBatch[T]:
    items: tuple[EvaluationItem[T], ...]

    @classmethod
    def from_items(cls, items: Iterable[EvaluationItem[T]]) -> EvaluationBatch[T]:
        materialized = tuple(items)
        if not materialized:
            raise ValueError("evaluation batch must not be empty")
        return cls(materialized)

    @property
    def utterance_ids(self) -> list[str]:
        return [item.utterance_id for item in self.items]

    @property
    def inputs(self) -> list[T]:
        return [item.input for item in self.items]


PredictBatch = Callable[[Sequence[T]], Sequence[float]]


def score_batches(
    batches: Iterable[EvaluationBatch[T]],
    predict_batch: PredictBatch[T],
) -> list[ScoreRecord]:
    """Run a predictor over batches and return score records.

    The predictor receives only model inputs and must return one scalar score per
    input, in the same order. Any framework-specific eval/no_grad/device logic
    belongs inside that callable.
    """

    records: list[ScoreRecord] = []
    seen: set[str] = set()

    for batch in batches:
        _reject_duplicate_ids(batch.utterance_ids, seen)
        scores = list(predict_batch(batch.inputs))
        if len(scores) != len(batch.items):
            raise ValueError(
                f"predictor returned {len(scores)} scores for {len(batch.items)} inputs"
            )
        records.extend(
            ScoreRecord(item.utterance_id, float(score))
            for item, score in zip(batch.items, scores, strict=True)
        )

    if not records:
        raise ValueError("no evaluation items provided")
    return records


def write_scores_from_batches(
    batches: Iterable[EvaluationBatch[T]],
    predict_batch: PredictBatch[T],
    path: str | Path,
    *,
    sort_by_utterance_id: bool = True,
) -> Path:
    """Score batches and write an ASVspoof-compatible score file."""

    return write_score_file(
        score_batches(batches, predict_batch),
        path,
        sort_by_utterance_id=sort_by_utterance_id,
    )


def _reject_duplicate_ids(ids: Sequence[str], seen: set[str]) -> None:
    for utterance_id in ids:
        if not utterance_id:
            raise ValueError("utterance_id must be non-empty")
        if utterance_id in seen:
            raise ValueError(f"duplicate utterance_id: {utterance_id}")
        seen.add(utterance_id)
