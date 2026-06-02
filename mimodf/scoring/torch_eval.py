"""Torch adapter for the framework-agnostic evaluation pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class TorchEvaluationError(RuntimeError):
    """Raised when Torch evaluation cannot produce one score per input."""


ScoreFn = Callable[[Any], Any]


@dataclass(frozen=True)
class TorchBatchPredictor:
    """Callable adapter that turns a Torch model into a batch score predictor.

    The model-specific interpretation of outputs is deliberately explicit via
    ``score_fn``. This avoids baking AASIST/MiMo shape assumptions into the
    generic evaluation layer.
    """

    model: Any
    score_fn: ScoreFn
    device: str | None = None

    def __call__(self, inputs: Sequence[Any]) -> list[float]:
        torch = _import_torch()
        if not inputs:
            return []

        self.model.eval()
        batch = torch.stack(list(inputs))
        if self.device is not None:
            batch = batch.to(self.device)
            self.model.to(self.device)

        with torch.inference_mode():
            output = self.model(batch)
            scores = self.score_fn(output)

        return _scores_to_floats(scores, expected=len(inputs), torch=torch)


def _scores_to_floats(scores: Any, *, expected: int, torch: Any) -> list[float]:
    if not torch.is_tensor(scores):
        scores = torch.as_tensor(scores)
    scores = scores.detach().cpu().reshape(-1)
    if scores.numel() != expected:
        raise TorchEvaluationError(
            f"score_fn returned {scores.numel()} scores for {expected} inputs"
        )
    return [float(value) for value in scores.tolist()]


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise TorchEvaluationError("Torch is required for TorchBatchPredictor") from exc
    return torch
