"""Checkpoint selection and retention primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

StateT = TypeVar("StateT")
SaveFn = Callable[[StateT, Path], None]
DeleteFn = Callable[[Path], None]


@dataclass(frozen=True)
class CheckpointRecord:
    metric: float
    val_loss: float
    epoch: int
    path: Path


@dataclass
class TopKCheckpointTracker[StateT]:
    """Retain the best k checkpoints by lower metric."""

    directory: Path
    k: int
    save_fn: SaveFn[StateT]
    delete_fn: DeleteFn = lambda path: path.unlink(missing_ok=True)
    records: list[CheckpointRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def best(self) -> CheckpointRecord | None:
        return self.records[0] if self.records else None

    def consider(
        self,
        *,
        state: StateT,
        epoch: int,
        metric: float,
        val_loss: float,
        metric_name: str,
    ) -> CheckpointRecord | None:
        """Save checkpoint if it belongs in the retained top-k set.

        Returns the saved record, or `None` when the candidate was worse than
        all retained records.
        """

        candidate_path = self.directory / checkpoint_filename(epoch, metric_name, metric)
        candidate = CheckpointRecord(
            metric=float(metric), val_loss=float(val_loss), epoch=epoch, path=candidate_path
        )

        if len(self.records) < self.k:
            self._save(candidate, state)
            return candidate

        worst = self.records[-1]
        if candidate.metric >= worst.metric:
            return None

        self.records.pop()
        self.delete_fn(worst.path)
        self._save(candidate, state)
        return candidate

    def _save(self, record: CheckpointRecord, state: StateT) -> None:
        self.save_fn(state, record.path)
        self.records.append(record)
        self.records.sort(key=lambda item: (item.metric, item.epoch))


def checkpoint_filename(epoch: int, metric_name: str, metric: float) -> str:
    safe_metric = metric_name.strip().lower().replace(" ", "_")
    if not safe_metric:
        raise ValueError("metric_name must be non-empty")
    return f"epoch_{epoch}_{safe_metric}_{metric:.4f}.pth"
