"""Paired feature-drift summaries for tiny transform smokes."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mimodf.features.common import command_argv, git_revision
from mimodf.features.probe import pool_feature_record

DRIFT_REPORT_SCHEMA = "mimodf-paired-feature-drift/v1"


@dataclass(frozen=True)
class PairedDriftSettings:
    clean_feature_dir: Path
    transformed_feature_dir: Path
    transform_records: Path
    out_json: Path
    out_report: Path
    pooling: str = "continuous_mean_std"
    overwrite: bool = False


@dataclass(frozen=True)
class PairedDriftResult:
    summary_path: Path
    report_path: Path
    pairs: int

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": str(self.summary_path),
            "report": str(self.report_path),
            "pairs": self.pairs,
        }


def summarize_paired_feature_drift(settings: PairedDriftSettings) -> PairedDriftResult:
    _validate_settings(settings)
    if (settings.out_json.exists() or settings.out_report.exists()) and not settings.overwrite:
        raise FileExistsError("output exists")
    settings.out_json.parent.mkdir(parents=True, exist_ok=True)
    settings.out_report.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    clean_manifest = _load_json(settings.clean_feature_dir / "manifest.json")
    transformed_manifest = _load_json(settings.transformed_feature_dir / "manifest.json")
    clean_records = _records_by_utterance(settings.clean_feature_dir / "records.jsonl")
    transformed_records = _records_by_utterance(settings.transformed_feature_dir / "records.jsonl")
    transform_rows = _load_jsonl(settings.transform_records)

    rows: list[dict[str, object]] = []
    for transform in transform_rows:
        original_id = str(transform["original_utterance_id"])
        transformed_id = str(transform["utterance_id"])
        clean_record = clean_records[original_id]
        transformed_record = transformed_records[transformed_id]
        clean_vec = pool_feature_record(clean_record, clean_manifest, settings.pooling).vector
        transformed_vec = pool_feature_record(
            transformed_record, transformed_manifest, settings.pooling
        ).vector
        if clean_vec.shape != transformed_vec.shape:
            raise ValueError(
                f"feature shape mismatch for {transformed_id}: "
                f"{clean_vec.shape} != {transformed_vec.shape}"
            )
        delta = transformed_vec - clean_vec
        rows.append(
            {
                "utterance_id": transformed_id,
                "original_utterance_id": original_id,
                "transform_id": transform.get("transform_id"),
                "transform_family": transform.get("transform_family"),
                "label": transform.get("label"),
                "source_model": transform.get("source_model"),
                "cosine_similarity": _cosine(clean_vec, transformed_vec),
                "l2_delta": float(np.linalg.norm(delta)),
                "mean_abs_delta": float(np.mean(np.abs(delta))),
            }
        )

    summary = {
        "schema": DRIFT_REPORT_SCHEMA,
        "clean_feature_dir": str(settings.clean_feature_dir),
        "transformed_feature_dir": str(settings.transformed_feature_dir),
        "transform_records": str(settings.transform_records),
        "pooling": settings.pooling,
        "pairs": len(rows),
        "by_transform": _aggregate(rows, "transform_id"),
        "by_family": _aggregate(rows, "transform_family"),
        "by_label": _aggregate(rows, "label"),
        "rows": rows,
        "started_unix": started,
        "finished_unix": time.time(),
        "git_revision": git_revision(),
        "command_argv": command_argv(),
        "caveats": [
            "tiny paired feature-drift smoke only; not robustness metric evidence",
            "no classifier scoring or binary EER is computed",
            "transformed labels are inherited stress-test metadata, not new ground truth",
        ],
    }
    settings.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    settings.out_report.write_text(render_paired_drift_report(summary), encoding="utf-8")
    return PairedDriftResult(
        summary_path=settings.out_json,
        report_path=settings.out_report,
        pairs=len(rows),
    )


def render_paired_drift_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Paired feature drift summary",
        "",
        f"Pairs: {summary['pairs']}",
        f"Pooling: `{summary['pooling']}`",
        "",
        "## By transform",
        "",
        _aggregate_table(summary["by_transform"]),
        "",
        "## By label",
        "",
        _aggregate_table(summary["by_label"]),
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["caveats"])
    return "\n".join(lines).rstrip() + "\n"


def _validate_settings(settings: PairedDriftSettings) -> None:
    for path in (
        settings.clean_feature_dir / "manifest.json",
        settings.clean_feature_dir / "records.jsonl",
        settings.transformed_feature_dir / "manifest.json",
        settings.transformed_feature_dir / "records.jsonl",
        settings.transform_records,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected object")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(data)
    return rows


def _records_by_utterance(path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        utterance_id = str(row.get("utterance_id"))
        if utterance_id in result:
            raise ValueError(f"duplicate utterance_id in {path}: {utterance_id}")
        result[utterance_id] = row
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _aggregate(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field))].append(row)
    return {key: _aggregate_group(values) for key, values in sorted(grouped.items())}


def _aggregate_group(rows: list[dict[str, object]]) -> dict[str, float | int]:
    cosine = np.array([float(row["cosine_similarity"]) for row in rows], dtype=np.float64)
    l2 = np.array([float(row["l2_delta"]) for row in rows], dtype=np.float64)
    mad = np.array([float(row["mean_abs_delta"]) for row in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
        "l2_delta_mean": float(l2.mean()),
        "l2_delta_max": float(l2.max()),
        "mean_abs_delta_mean": float(mad.mean()),
    }


def _aggregate_table(groups: dict[str, dict[str, float | int]]) -> str:
    lines = [
        "| Group | Count | Cosine mean | Cosine min | L2 mean | L2 max | Mean abs delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in groups.items():
        lines.append(
            f"| {key} | {values['count']} | {values['cosine_mean']:.4f} | "
            f"{values['cosine_min']:.4f} | {values['l2_delta_mean']:.4f} | "
            f"{values['l2_delta_max']:.4f} | {values['mean_abs_delta_mean']:.4f} |"
        )
    return "\n".join(lines)
