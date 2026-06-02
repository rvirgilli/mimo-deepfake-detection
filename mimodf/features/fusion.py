"""Score fusion and error-overlap reports for feature probes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mimodf.features.common import command_argv, git_revision
from mimodf.features.probe import compute_metrics

FUSION_REPORT_SCHEMA = "mimodf-feature-probe-fusion/v1"


@dataclass(frozen=True)
class ProbeFusionSettings:
    left_predictions: Path
    right_predictions: Path
    out_dir: Path
    left_weight: float = 0.5
    right_weight: float = 0.5
    positive_label: str = "spoof"
    overwrite: bool = False


@dataclass(frozen=True)
class ProbeFusionResult:
    report_path: Path
    metrics_path: Path
    predictions_path: Path
    records: int

    def to_dict(self) -> dict[str, object]:
        return {
            "report": str(self.report_path),
            "metrics": str(self.metrics_path),
            "predictions": str(self.predictions_path),
            "records": self.records,
        }


def run_probe_fusion(settings: ProbeFusionSettings) -> ProbeFusionResult:
    _validate_settings(settings)
    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    left = _load_predictions(settings.left_predictions)
    right = _load_predictions(settings.right_predictions)
    left_by_id = {str(record["utterance_id"]): record for record in left}
    right_by_id = {str(record["utterance_id"]): record for record in right}
    shared_ids = sorted(set(left_by_id) & set(right_by_id))
    if not shared_ids:
        raise ValueError("prediction files have no overlapping utterance_id values")

    classes = _classes(left_by_id[shared_ids[0]])
    if classes != _classes(right_by_id[shared_ids[0]]):
        raise ValueError("prediction files use different class sets")
    if settings.positive_label not in classes and len(classes) == 2:
        positive_label: str | None = classes[1]
    else:
        positive_label = settings.positive_label if settings.positive_label in classes else None
    class_to_index = {label: index for index, label in enumerate(classes)}

    y_true: list[int] = []
    y_left: list[int] = []
    y_right: list[int] = []
    y_fused: list[int] = []
    probabilities: list[list[float]] = []
    prediction_rows: list[dict[str, object]] = []
    for utterance_id in shared_ids:
        left_record = left_by_id[utterance_id]
        right_record = right_by_id[utterance_id]
        if left_record["target"] != right_record["target"]:
            raise ValueError(f"target mismatch for {utterance_id}")
        left_probs = _probability_vector(left_record, classes)
        right_probs = _probability_vector(right_record, classes)
        fused = settings.left_weight * left_probs + settings.right_weight * right_probs
        fused = fused / fused.sum()
        target_index = class_to_index[str(left_record["target"])]
        fused_index = int(fused.argmax())
        y_true.append(target_index)
        y_left.append(class_to_index[str(left_record["prediction"])])
        y_right.append(class_to_index[str(right_record["prediction"])])
        y_fused.append(fused_index)
        probabilities.append(fused.tolist())
        prediction_rows.append(
            {
                "schema": "mimodf-feature-probe-fusion-prediction/v1",
                "utterance_id": utterance_id,
                "target": left_record["target"],
                "left_prediction": left_record["prediction"],
                "right_prediction": right_record["prediction"],
                "fusion_prediction": classes[fused_index],
                "fusion_probabilities": {
                    label: float(fused[class_index]) for class_index, label in enumerate(classes)
                },
                "label": left_record.get("label"),
                "source_model": left_record.get("source_model"),
            }
        )

    y_true_np = np.array(y_true, dtype=np.int64)
    y_fused_np = np.array(y_fused, dtype=np.int64)
    prob_np = np.array(probabilities, dtype=np.float64)
    metrics = compute_metrics(
        y_true_np, y_fused_np, prob_np, classes=classes, positive_label=positive_label
    )
    metrics.update(
        {
            "schema": FUSION_REPORT_SCHEMA,
            "left_predictions": str(settings.left_predictions),
            "right_predictions": str(settings.right_predictions),
            "left_weight": settings.left_weight,
            "right_weight": settings.right_weight,
            "records": len(shared_ids),
            "classes": classes,
            "overlap": _error_overlap(y_true_np, np.array(y_left), np.array(y_right), y_fused_np),
            "score_correlation": _score_correlation(
                left_by_id, right_by_id, shared_ids, classes, positive_label
            ),
            "started_unix": started,
            "finished_unix": time.time(),
            "git_revision": git_revision(),
            "command_argv": command_argv(),
            "caveats": [
                "score-level average over existing feature-probe probabilities",
                "diagnostic only; not a calibrated deployment fusion",
            ],
        }
    )

    metrics_path = settings.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path = settings.out_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    report_path = settings.out_dir / "report.md"
    report_path.write_text(render_fusion_report(metrics), encoding="utf-8")
    return ProbeFusionResult(report_path, metrics_path, predictions_path, len(shared_ids))


def render_fusion_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Feature probe fusion report",
        "",
        f"Left predictions: `{metrics['left_predictions']}`",
        f"Right predictions: `{metrics['right_predictions']}`",
        f"Weights: {metrics['left_weight']} / {metrics['right_weight']}",
        "",
        "## Metrics",
        "",
        f"- records: {metrics['records']}",
        f"- balanced_accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- macro_f1: {metrics['macro_f1']:.4f}",
    ]
    if "auroc" in metrics:
        lines.extend([f"- auroc: {metrics['auroc']:.4f}", f"- eer: {metrics['eer']:.4f}"])
    lines.extend(
        [
            "",
            "## Error overlap",
            "",
            f"`{json.dumps(metrics['overlap'], sort_keys=True)}`",
            "",
            f"Score correlation: `{metrics['score_correlation']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_settings(settings: ProbeFusionSettings) -> None:
    if settings.left_weight < 0 or settings.right_weight < 0:
        raise ValueError("fusion weights must be non-negative")
    if settings.left_weight + settings.right_weight <= 0:
        raise ValueError("at least one fusion weight must be positive")
    if not settings.left_predictions.is_file():
        raise FileNotFoundError(settings.left_predictions)
    if not settings.right_predictions.is_file():
        raise FileNotFoundError(settings.right_predictions)


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _classes(record: dict[str, Any]) -> list[str]:
    return sorted(record["probabilities"].keys())


def _probability_vector(record: dict[str, Any], classes: list[str]) -> np.ndarray:
    return np.array([float(record["probabilities"][label]) for label in classes], dtype=np.float64)


def _error_overlap(
    y_true: np.ndarray, y_left: np.ndarray, y_right: np.ndarray, y_fused: np.ndarray
) -> dict[str, int]:
    left_correct = y_left == y_true
    right_correct = y_right == y_true
    fused_correct = y_fused == y_true
    return {
        "both_single_correct": int((left_correct & right_correct).sum()),
        "left_only_correct": int((left_correct & ~right_correct).sum()),
        "right_only_correct": int((~left_correct & right_correct).sum()),
        "both_single_wrong": int((~left_correct & ~right_correct).sum()),
        "single_disagreements": int((y_left != y_right).sum()),
        "fusion_correct": int(fused_correct.sum()),
        "fusion_wrong": int((~fused_correct).sum()),
    }


def _score_correlation(
    left_by_id: dict[str, dict[str, Any]],
    right_by_id: dict[str, dict[str, Any]],
    shared_ids: list[str],
    classes: list[str],
    positive_label: str | None,
) -> float | None:
    if positive_label is None or positive_label not in classes or len(shared_ids) < 2:
        return None
    left_scores = np.array(
        [left_by_id[item]["probabilities"][positive_label] for item in shared_ids], dtype=np.float64
    )
    right_scores = np.array(
        [right_by_id[item]["probabilities"][positive_label] for item in shared_ids],
        dtype=np.float64,
    )
    if float(left_scores.std()) == 0.0 or float(right_scores.std()) == 0.0:
        return None
    return float(np.corrcoef(left_scores, right_scores)[0, 1])
