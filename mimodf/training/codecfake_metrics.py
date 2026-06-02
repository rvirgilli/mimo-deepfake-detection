"""Prediction metrics for CodecFake+ trained diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mimodf.features.probe import compute_metrics
from mimodf.training.codecfake import CLASSES


def build_prediction_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("prediction records must not be empty")
    metrics = build_prediction_metrics_no_groups(records)
    metrics["records"] = len(records)
    metrics["label_convention"] = {label: index for index, label in enumerate(CLASSES)}
    metrics["score_summary_by_label"] = _score_summary_by_label(records)
    metrics["per_source"] = _per_group_metrics(records, "source_model")
    return _json_safe(metrics)


def build_prediction_metrics_no_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    target = np.array([CLASSES.index(str(row["target"])) for row in records], dtype=np.int64)
    pred = np.array([CLASSES.index(str(row["prediction"])) for row in records], dtype=np.int64)
    probabilities = np.array(
        [[float(row["probabilities"][label]) for label in CLASSES] for row in records],
        dtype=np.float64,
    )
    metrics = compute_metrics(target, pred, probabilities, classes=CLASSES, positive_label="spoof")
    if len(set(target.tolist())) < 2:
        metrics["auroc"] = math.nan
        metrics["eer"] = math.nan
        metrics["binary_metric_status"] = "undefined_single_class_support"
    return _json_safe(metrics)


def _score_summary_by_label(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label in CLASSES:
        scores = [_spoof_score(row) for row in records if row.get("target") == label]
        if not scores:
            output[label] = {"records": 0}
            continue
        arr = np.array(scores, dtype=np.float64)
        output[label] = {
            "records": len(scores),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return output


def _spoof_score(row: dict[str, Any]) -> float:
    if "score" in row:
        return float(row["score"])
    return float(row["probabilities"]["spoof"])


def _per_group_metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get(field)), []).append(record)
    output: dict[str, Any] = {}
    for value, rows in sorted(groups.items()):
        if len({row["target"] for row in rows}) < 2:
            output[value] = {"records": len(rows), "skipped": "single_class_support"}
        else:
            item = build_prediction_metrics_no_groups(rows)
            item["records"] = len(rows)
            output[value] = item
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
