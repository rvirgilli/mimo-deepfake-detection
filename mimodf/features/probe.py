"""Frozen-feature linear probes for Wave 1 research diagnostics.

This is deliberately small and file-based. It consumes feature-cache manifests and
records, pools utterance-level vectors, trains one simple linear classifier, and
writes metrics. It is not a training framework.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mimodf.features.common import command_argv, git_revision

TaskName = Literal["label", "source_model", "quantizer_type", "auxiliary_objective", "decoder_type"]
PoolingMode = Literal["auto", "continuous_mean_std", "continuous_mean", "rvq_hist"]
SplitMode = Literal["random-stratified", "holdout-values"]
Backend = Literal["auto", "sklearn", "numpy"]

PROBE_REPORT_SCHEMA = "mimodf-feature-probe/v1"


@dataclass(frozen=True)
class ProbeSettings:
    feature_dir: Path
    out_dir: Path
    task: TaskName
    split: SplitMode = "random-stratified"
    seed: int = 42
    test_fraction: float = 0.2
    holdout_field: str | None = None
    holdout_values: tuple[str, ...] = ()
    pooling: PoolingMode = "auto"
    backend: Backend = "auto"
    l2: float = 1.0
    max_iter: int = 500
    drop_missing_target: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ProbeResult:
    report_path: Path
    metrics_path: Path
    predictions_path: Path
    records: int
    train_records: int
    test_records: int

    def to_dict(self) -> dict[str, object]:
        return {
            "report": str(self.report_path),
            "metrics": str(self.metrics_path),
            "predictions": str(self.predictions_path),
            "records": self.records,
            "train_records": self.train_records,
            "test_records": self.test_records,
        }


def run_feature_probe(settings: ProbeSettings) -> ProbeResult:
    _validate_settings(settings)
    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    manifest = _load_json(settings.feature_dir / "manifest.json")
    records = _load_jsonl(settings.feature_dir / "records.jsonl")
    usable_records = [
        record for record in records if _target_value(record, settings.task) is not None
    ]
    if not settings.drop_missing_target and len(usable_records) != len(records):
        raise ValueError(f"task {settings.task} has missing target values")
    if len(usable_records) < 4:
        raise ValueError("not enough records with usable targets")

    pooled = [pool_feature_record(record, manifest, settings.pooling) for record in usable_records]
    x = np.vstack([item.vector for item in pooled]).astype(np.float32, copy=False)
    targets = [str(_target_value(record, settings.task)) for record in usable_records]
    classes = sorted(set(targets))
    if len(classes) < 2:
        raise ValueError(f"task {settings.task} has fewer than two target classes")
    class_to_index = {label: index for index, label in enumerate(classes)}
    y = np.array([class_to_index[target] for target in targets], dtype=np.int64)

    train_idx, test_idx = make_split(
        usable_records,
        y,
        split=settings.split,
        seed=settings.seed,
        test_fraction=settings.test_fraction,
        holdout_field=settings.holdout_field,
        holdout_values=settings.holdout_values,
    )
    _validate_split(y, train_idx, test_idx)

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    x_train, x_test, standardization = _standardize_train_only(x_train, x_test)

    model_payload = fit_linear_classifier(
        x_train,
        y_train,
        num_classes=len(classes),
        backend=settings.backend,
        l2=settings.l2,
        max_iter=settings.max_iter,
        seed=settings.seed,
    )
    probabilities = model_payload["predict_proba"](x_test)
    predictions = probabilities.argmax(axis=1)
    prediction_records = _prediction_records(
        usable_records, test_idx, y_test, predictions, probabilities, classes
    )

    metrics = compute_metrics(
        y_test, predictions, probabilities, classes=classes, positive_label=_positive_label(classes)
    )
    metrics.update(
        {
            "schema": PROBE_REPORT_SCHEMA,
            "feature_dir": str(settings.feature_dir),
            "feature_manifest": str(settings.feature_dir / "manifest.json"),
            "component_id": manifest.get("component_id"),
            "representation": manifest.get("representation"),
            "task": settings.task,
            "pooling": pooled[0].pooling,
            "feature_dim": int(x.shape[1]),
            "records": int(len(usable_records)),
            "dropped_missing_target": int(len(records) - len(usable_records)),
            "train_records": int(len(train_idx)),
            "test_records": int(len(test_idx)),
            "split": {
                "mode": settings.split,
                "seed": settings.seed,
                "test_fraction": settings.test_fraction,
                "holdout_field": settings.holdout_field,
                "holdout_values": list(settings.holdout_values),
            },
            "classes": classes,
            "train_support": _support(y_train, classes),
            "test_support": _support(y_test, classes),
            "backend": model_payload["backend"],
            "standardization": standardization,
            "l2": settings.l2,
            "max_iter": settings.max_iter,
            "started_unix": started,
            "finished_unix": time.time(),
            "git_revision": git_revision(),
            "command_argv": command_argv(),
            "caveats": _caveats(settings),
        }
    )

    metrics_path = settings.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path = settings.out_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for record in prediction_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    report_path = settings.out_dir / "report.md"
    report_path.write_text(render_probe_report(metrics), encoding="utf-8")
    return ProbeResult(
        report_path=report_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        records=len(usable_records),
        train_records=len(train_idx),
        test_records=len(test_idx),
    )


@dataclass(frozen=True)
class PooledFeature:
    vector: np.ndarray
    pooling: str


def pool_feature_record(
    record: dict[str, Any], manifest: dict[str, Any], pooling: PoolingMode = "auto"
) -> PooledFeature:
    values = np.load(record["array_path"])["values"]
    value_kind = record.get("value_kind")
    selected_pooling = pooling
    if selected_pooling == "auto":
        selected_pooling = "rvq_hist" if value_kind == "rvq_codes" else "continuous_mean_std"

    if selected_pooling == "continuous_mean":
        vector = values.astype(np.float32, copy=False).mean(axis=0)
    elif selected_pooling == "continuous_mean_std":
        floats = values.astype(np.float32, copy=False)
        vector = np.concatenate([floats.mean(axis=0), floats.std(axis=0)]).astype(
            np.float32, copy=False
        )
    elif selected_pooling == "rvq_hist":
        vector = _rvq_histogram(values, record, manifest)
    else:
        raise ValueError(f"unsupported pooling: {pooling}")
    return PooledFeature(vector=np.asarray(vector, dtype=np.float32), pooling=selected_pooling)


def make_split(
    records: list[dict[str, Any]],
    y: np.ndarray,
    *,
    split: SplitMode,
    seed: int,
    test_fraction: float,
    holdout_field: str | None,
    holdout_values: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if split == "random-stratified":
        return _random_stratified_split(y, seed=seed, test_fraction=test_fraction)
    if split == "holdout-values":
        if not holdout_field or not holdout_values:
            raise ValueError("holdout-values split requires holdout_field and holdout_values")
        holdouts = set(holdout_values)
        test_idx = np.array(
            [
                index
                for index, record in enumerate(records)
                if str(record.get(holdout_field)) in holdouts
            ],
            dtype=np.int64,
        )
        train_idx = np.array(
            [
                index
                for index, record in enumerate(records)
                if str(record.get(holdout_field)) not in holdouts
            ],
            dtype=np.int64,
        )
        return train_idx, test_idx
    raise ValueError(f"unsupported split: {split}")


def fit_linear_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    num_classes: int,
    backend: Backend,
    l2: float,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    if backend in {"auto", "sklearn"}:
        try:
            return _fit_sklearn_logistic(
                x_train, y_train, backend=backend, l2=l2, max_iter=max_iter, seed=seed
            )
        except ImportError:
            if backend == "sklearn":
                raise
    return _fit_numpy_softmax(
        x_train, y_train, num_classes=num_classes, l2=l2, max_iter=max_iter, seed=seed
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: list[str],
    positive_label: str | None,
) -> dict[str, Any]:
    confusion = _confusion_matrix(y_true, y_pred, len(classes))
    per_class_recall = []
    per_class_f1 = []
    for index in range(len(classes)):
        tp = float(confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        recall = _safe_div(tp, tp + fn)
        precision = _safe_div(tp, tp + fp)
        per_class_recall.append(recall)
        per_class_f1.append(_safe_div(2 * precision * recall, precision + recall))

    payload: dict[str, Any] = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean(per_class_recall)),
        "macro_f1": float(np.mean(per_class_f1)),
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": {
            label: float(value) for label, value in zip(classes, per_class_recall, strict=True)
        },
        "per_class_f1": {
            label: float(value) for label, value in zip(classes, per_class_f1, strict=True)
        },
    }
    if len(classes) == 2 and positive_label is not None:
        pos_index = classes.index(positive_label)
        positive_scores = probabilities[:, pos_index]
        binary_true = (y_true == pos_index).astype(np.int64)
        payload["positive_label"] = positive_label
        payload["auroc"] = float(_binary_auroc(binary_true, positive_scores))
        payload["eer"] = float(_binary_eer(binary_true, positive_scores))
    return payload


def _prediction_records(
    records: list[dict[str, Any]],
    test_idx: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for local_index, record_index in enumerate(test_idx.tolist()):
        record = records[record_index]
        output.append(
            {
                "schema": "mimodf-feature-probe-prediction/v1",
                "utterance_id": record.get("utterance_id"),
                "target": classes[int(y_test[local_index])],
                "prediction": classes[int(predictions[local_index])],
                "probabilities": {
                    label: float(probabilities[local_index, class_index])
                    for class_index, label in enumerate(classes)
                },
                "label": record.get("label"),
                "source_model": record.get("source_model"),
                "quantizer_type": record.get("quantizer_type"),
                "auxiliary_objective": record.get("auxiliary_objective"),
                "decoder_type": record.get("decoder_type"),
            }
        )
    return output


def render_probe_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Feature probe report",
        "",
        f"Feature dir: `{metrics['feature_dir']}`",
        f"Task: `{metrics['task']}`",
        f"Representation: `{metrics.get('representation')}`",
        f"Pooling: `{metrics['pooling']}`",
        f"Split: `{metrics['split']['mode']}`",
        f"Backend: `{metrics['backend']}`",
        "",
        "## Metrics",
        "",
        f"- accuracy: {metrics['accuracy']:.4f}",
        f"- balanced_accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- macro_f1: {metrics['macro_f1']:.4f}",
    ]
    if "auroc" in metrics:
        lines.extend([f"- auroc: {metrics['auroc']:.4f}", f"- eer: {metrics['eer']:.4f}"])
    lines.extend(
        [
            "",
            "## Supports",
            "",
            f"Train: `{json.dumps(metrics['train_support'], sort_keys=True)}`",
            f"Test: `{json.dumps(metrics['test_support'], sort_keys=True)}`",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in metrics["caveats"])
    lines.append("")
    return "\n".join(lines)


def _validate_settings(settings: ProbeSettings) -> None:
    if not (0.0 < settings.test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1")
    if settings.l2 < 0:
        raise ValueError("l2 must be non-negative")
    if settings.max_iter <= 0:
        raise ValueError("max_iter must be positive")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _target_value(record: dict[str, Any], task: TaskName) -> object | None:
    return record.get(task)


def _rvq_histogram(
    values: np.ndarray, record: dict[str, Any], manifest: dict[str, Any]
) -> np.ndarray:
    selected = record.get("selected_quantizers")
    if selected is None:
        selected = list(range(values.shape[1]))
    codebook_sizes = manifest.get("model_config", {}).get("codebook_size") or []
    parts: list[np.ndarray] = []
    for local_index, quantizer_index in enumerate(selected):
        if quantizer_index < len(codebook_sizes):
            bins = int(codebook_sizes[quantizer_index])
        else:
            bins = int(values[:, local_index].max()) + 1
        counts = np.bincount(values[:, local_index].astype(np.int64), minlength=bins).astype(
            np.float32
        )
        total = float(max(1, values.shape[0]))
        parts.append(counts / total)
    return np.concatenate(parts).astype(np.float32, copy=False)


def _random_stratified_split(
    y: np.ndarray, *, seed: int, test_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for class_index in sorted(set(y.tolist())):
        indices = np.flatnonzero(y == class_index)
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_fraction)))
        if test_count >= len(indices):
            test_count = len(indices) - 1
        test_parts.append(indices[:test_count])
        train_parts.append(indices[test_count:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def _validate_split(y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("split produced empty train or test set")
    if set(y[train_idx].tolist()) != set(y.tolist()):
        raise ValueError("train split is missing one or more target classes")
    if set(y[test_idx].tolist()) != set(y.tolist()):
        raise ValueError("test split is missing one or more target classes")


def _standardize_train_only(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, {"mode": "train_mean_std"}


def _fit_sklearn_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    backend: Backend,
    l2: float,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        raise
    c_value = 1e6 if l2 == 0 else 1.0 / l2
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(x_train, y_train)
    return {"backend": "sklearn_logistic_regression", "predict_proba": model.predict_proba}


def _fit_numpy_softmax(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    num_classes: int,
    l2: float,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n, d = x_train.shape
    weights = rng.normal(0.0, 0.01, size=(d, num_classes)).astype(np.float32)
    bias = np.zeros(num_classes, dtype=np.float32)
    targets = np.zeros((n, num_classes), dtype=np.float32)
    targets[np.arange(n), y_train] = 1.0
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = n / (num_classes * np.maximum(class_counts, 1.0))
    sample_weights = class_weights[y_train][:, None]
    lr = 0.1
    for _ in range(max_iter):
        logits = x_train @ weights + bias
        probs = _softmax(logits)
        grad = (probs - targets) * sample_weights / n
        weights -= lr * (x_train.T @ grad + l2 * weights / n)
        bias -= lr * grad.sum(axis=0)

    def predict_proba(x: np.ndarray) -> np.ndarray:
        return _softmax(x @ weights + bias)

    return {"backend": "numpy_softmax_regression", "predict_proba": predict_proba}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred, strict=True):
        confusion[int(truth), int(pred)] += 1
    return confusion


def _support(y: np.ndarray, classes: list[str]) -> dict[str, int]:
    counts = np.bincount(y, minlength=len(classes))
    return {label: int(counts[index]) for index, label in enumerate(classes)}


def _positive_label(classes: list[str]) -> str | None:
    if "spoof" in classes:
        return "spoof"
    if len(classes) == 2:
        return classes[1]
    return None


def _binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[y_true == 1]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _binary_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.unique(scores)
    best = (1.0, 1.0)
    positives = max(1, int((y_true == 1).sum()))
    negatives = max(1, int((y_true == 0).sum()))
    for threshold in thresholds:
        pred = scores >= threshold
        frr = float(((y_true == 1) & ~pred).sum() / positives)
        far = float(((y_true == 0) & pred).sum() / negatives)
        gap = abs(frr - far)
        if gap < best[0]:
            best = (gap, (frr + far) / 2)
    return float(best[1])


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _caveats(settings: ProbeSettings) -> list[str]:
    caveats = ["feature-only linear probe; not a full model training/evaluation claim"]
    if settings.split == "random-stratified":
        caveats.append(
            "random row split is diagnostic only; do not treat as the main leakage-safe result"
        )
    if settings.split == "holdout-values":
        caveats.append("holdout split is only as good as the declared holdout field/value policy")
    return caveats
