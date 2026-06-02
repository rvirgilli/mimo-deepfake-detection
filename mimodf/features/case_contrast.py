"""Contrast case groups from an existing mechanism-analysis run."""

from __future__ import annotations

import json
import time
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from mimodf.features.common import command_argv, git_revision
from mimodf.features.predictions import PredictionSource, parse_prediction_source
from mimodf.features.probe import pool_feature_record

CASE_CONTRAST_SCHEMA = "mimodf-case-contrast/v1"


@dataclass(frozen=True)
class CaseContrastSettings:
    cases_path: Path
    protocol: Path
    feature_sources: tuple[PredictionSource, ...]
    out_dir: Path
    reference_system: str
    contrast_system: str
    positive_label: str = "spoof"
    overwrite: bool = False


@dataclass(frozen=True)
class CaseContrastResult:
    report_path: Path
    summary_path: Path
    cases_path: Path
    records: int

    def to_dict(self) -> dict[str, object]:
        return {
            "report": str(self.report_path),
            "summary": str(self.summary_path),
            "cases": str(self.cases_path),
            "records": self.records,
        }


def parse_feature_source(value: str) -> PredictionSource:
    return parse_prediction_source(value)


def run_case_contrast(settings: CaseContrastSettings) -> CaseContrastResult:
    _validate_settings(settings)
    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    cases = _load_jsonl(settings.cases_path)
    protocol = _load_protocol(settings.protocol)
    enriched = [_enrich_case(case, protocol) for case in cases]
    for case in enriched:
        case["contrast_group"] = _contrast_group(
            case, reference=settings.reference_system, contrast=settings.contrast_system
        )

    features = {
        source.name: _feature_vectors(source.path, [case["utterance_id"] for case in enriched])
        for source in settings.feature_sources
    }
    summary = {
        "schema": CASE_CONTRAST_SCHEMA,
        "cases_path": str(settings.cases_path),
        "protocol": str(settings.protocol),
        "feature_sources": {source.name: str(source.path) for source in settings.feature_sources},
        "reference_system": settings.reference_system,
        "contrast_system": settings.contrast_system,
        "positive_label": settings.positive_label,
        "records": len(enriched),
        "groups": _group_summaries(enriched, positive_label=settings.positive_label),
        "score_contrasts": _score_contrasts(enriched, positive_label=settings.positive_label),
        "audio_contrasts": _audio_contrasts(enriched),
        "feature_contrasts": {
            name: _feature_contrast(enriched, vectors) for name, vectors in features.items()
        },
        "rvq_entropy": {
            name: _rvq_entropy_summary(source.path, enriched)
            for name, source in ((source.name, source) for source in settings.feature_sources)
            if _is_rvq_feature_dir(source.path)
        },
        "started_unix": started,
        "finished_unix": time.time(),
        "git_revision": git_revision(),
        "command_argv": command_argv(),
        "caveats": [
            "case contrast over existing predictions/features only",
            "no model training, feature extraction, HPO, or external evaluation is run",
            "small contrast group sizes, especially reference-fixes-contrast; treat as mechanism diagnostics",
        ],
    }

    cases_out = settings.out_dir / "cases.jsonl"
    with cases_out.open("w", encoding="utf-8") as f:
        for case in enriched:
            f.write(json.dumps(case, sort_keys=True) + "\n")
    summary_path = settings.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_path = settings.out_dir / "report.md"
    report_path.write_text(render_case_contrast_report(summary), encoding="utf-8")
    return CaseContrastResult(report_path, summary_path, cases_out, len(enriched))


def render_case_contrast_report(summary: dict[str, Any]) -> str:
    ref = summary["reference_system"]
    contrast = summary["contrast_system"]
    lines = [
        "# CLAMTTS case-contrast analysis",
        "",
        f"Records: {summary['records']}",
        f"Reference: `{ref}`",
        f"Contrast: `{contrast}`",
        "",
        "## Groups",
        "",
        "| Group | Records | Labels | Duration mean | RMS mean | Silence mean |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name, item in summary["groups"].items():
        audio = summary["audio_contrasts"].get(name, {})
        lines.append(
            f"| {name} | {item['records']} | `{json.dumps(item['labels'], sort_keys=True)}` | "
            f"{_fmt(item.get('duration_mean_sec'))} | {_fmt(audio.get('rms_mean'))} | "
            f"{_fmt(audio.get('silence_fraction_mean'))} |"
        )

    lines.extend(["", "## Score contrast", ""])
    for group, systems in summary["score_contrasts"].items():
        lines.extend(
            [
                f"### {group}",
                "",
                "| System | Count | Mean spoof score | Min | Max |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for system, item in systems.items():
            lines.append(
                f"| {system} | {item['count']} | {_fmt(item.get('mean'))} | "
                f"{_fmt(item.get('min'))} | {_fmt(item.get('max'))} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Feature centroid contrast",
            "",
            "Primary comparison is `contrast_fixes_reference` vs `reference_fixes_contrast`.",
            "",
            "| Feature | A count | B count | Centroid cosine | Centroid L2 | Mean abs standardized diff |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary["feature_contrasts"].items():
        primary = item.get("contrast_fixes_reference__vs__reference_fixes_contrast", {})
        lines.append(
            f"| {name} | {primary.get('left_count', 0)} | {primary.get('right_count', 0)} | "
            f"{_fmt(primary.get('centroid_cosine'))} | {_fmt(primary.get('centroid_l2'))} | "
            f"{_fmt(primary.get('mean_abs_standardized_diff'))} |"
        )

    if summary.get("rvq_entropy"):
        lines.extend(["", "## RVQ entropy", ""])
        for name, groups in summary["rvq_entropy"].items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Group | Quantizers | Mean entropy |")
            lines.append("|---|---:|---:|")
            for group, item in groups.items():
                lines.append(
                    f"| {group} | {item['quantizers']} | {_fmt(item.get('mean_entropy'))} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- This analysis can suggest mechanisms; it cannot prove causality.",
            "- The reverse group is small, so avoid overfitting explanations to individual files.",
            "- Any proposed mechanism should predict behavior on another source or transform before scaling.",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["caveats"])
    lines.append("")
    return "\n".join(lines)


def _validate_settings(settings: CaseContrastSettings) -> None:
    if not settings.cases_path.is_file():
        raise FileNotFoundError(settings.cases_path)
    if not settings.protocol.is_file():
        raise FileNotFoundError(settings.protocol)
    if not settings.feature_sources:
        raise ValueError("at least one feature source is required")
    names = [source.name for source in settings.feature_sources]
    if len(names) != len(set(names)):
        raise ValueError("feature source names must be unique")
    for source in settings.feature_sources:
        if not (source.path / "manifest.json").is_file():
            raise FileNotFoundError(source.path / "manifest.json")
        if not (source.path / "records.jsonl").is_file():
            raise FileNotFoundError(source.path / "records.jsonl")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def _load_protocol(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        rows[str(row["utterance_id"])] = row
    return rows


def _enrich_case(case: dict[str, Any], protocol: dict[str, dict[str, Any]]) -> dict[str, Any]:
    utterance_id = str(case["utterance_id"])
    merged = dict(case)
    protocol_row = protocol.get(utterance_id, {})
    merged["audio_path"] = protocol_row.get("audio_path")
    merged["audio_stats"] = _audio_stats(protocol_row.get("audio_path"))
    return merged


def _contrast_group(case: dict[str, Any], *, reference: str, contrast: str) -> str:
    ref_correct = bool(case["systems"][reference]["correct"])
    contrast_correct = bool(case["systems"][contrast]["correct"])
    if contrast_correct and not ref_correct:
        return "contrast_fixes_reference"
    if ref_correct and not contrast_correct:
        return "reference_fixes_contrast"
    if ref_correct and contrast_correct:
        return "both_correct"
    return "both_wrong"


def _group_summaries(
    cases: list[dict[str, Any]], *, positive_label: str
) -> dict[str, dict[str, Any]]:
    del positive_label
    output: dict[str, dict[str, Any]] = {}
    for group, items in _group_cases(cases).items():
        durations = [
            float(item["audio"].get("duration_sec"))
            for item in items
            if item.get("audio") and item["audio"].get("duration_sec") is not None
        ]
        output[group] = {
            "records": len(items),
            "labels": dict(sorted(Counter(str(item.get("target")) for item in items).items())),
            "duration_buckets": dict(
                sorted(Counter(str(item.get("duration_bucket")) for item in items).items())
            ),
            "duration_mean_sec": mean(durations) if durations else None,
        }
    return output


def _score_contrasts(
    cases: list[dict[str, Any]], *, positive_label: str
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    del positive_label
    output: dict[str, dict[str, dict[str, float | int | None]]] = {}
    systems = list(cases[0]["systems"])
    for group, items in _group_cases(cases).items():
        output[group] = {}
        for system in systems:
            scores = [
                item["systems"][system].get("positive_probability")
                for item in items
                if item["systems"][system].get("positive_probability") is not None
            ]
            values = [float(score) for score in scores]
            output[group][system] = _numeric_summary(values)
    return output


def _audio_contrasts(cases: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for group, items in _group_cases(cases).items():
        output[group] = {
            "records": len(items),
            "rms_mean": _mean_present(item["audio_stats"].get("rms") for item in items),
            "peak_abs_mean": _mean_present(item["audio_stats"].get("peak_abs") for item in items),
            "silence_fraction_mean": _mean_present(
                item["audio_stats"].get("silence_fraction") for item in items
            ),
        }
    return output


def _feature_vectors(feature_dir: Path, utterance_ids: list[str]) -> dict[str, np.ndarray]:
    manifest = json.loads((feature_dir / "manifest.json").read_text())
    records = {str(row["utterance_id"]): row for row in _load_jsonl(feature_dir / "records.jsonl")}
    vectors: dict[str, np.ndarray] = {}
    for utterance_id in utterance_ids:
        record = records.get(utterance_id)
        if record is None:
            raise KeyError(f"{feature_dir}: missing feature record {utterance_id}")
        vectors[utterance_id] = pool_feature_record(record, manifest, "auto").vector
    return vectors


def _feature_contrast(
    cases: list[dict[str, Any]], vectors: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    groups = _group_cases(cases)
    pairs = [
        ("contrast_fixes_reference", "reference_fixes_contrast"),
        ("contrast_fixes_reference", "both_correct"),
        ("contrast_fixes_reference", "both_wrong"),
    ]
    output: dict[str, dict[str, object]] = {}
    all_matrix = np.vstack([vectors[str(case["utterance_id"])] for case in cases])
    std = all_matrix.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    for left, right in pairs:
        if not groups.get(left) or not groups.get(right):
            continue
        left_matrix = np.vstack([vectors[str(case["utterance_id"])] for case in groups[left]])
        right_matrix = np.vstack([vectors[str(case["utterance_id"])] for case in groups[right]])
        left_centroid = left_matrix.mean(axis=0)
        right_centroid = right_matrix.mean(axis=0)
        diff = left_centroid - right_centroid
        standardized = np.abs(diff / std)
        top_indices = np.argsort(standardized)[-10:][::-1].tolist()
        output[f"{left}__vs__{right}"] = {
            "left_count": int(left_matrix.shape[0]),
            "right_count": int(right_matrix.shape[0]),
            "feature_dim": int(left_matrix.shape[1]),
            "centroid_l2": float(np.linalg.norm(diff)),
            "centroid_cosine": _cosine(left_centroid, right_centroid),
            "mean_abs_standardized_diff": float(np.mean(standardized)),
            "top_standardized_diff_indices": [int(index) for index in top_indices],
            "top_standardized_diff_values": [float(standardized[index]) for index in top_indices],
        }
    return output


def _rvq_entropy_summary(
    feature_dir: Path, cases: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    records = {str(row["utterance_id"]): row for row in _load_jsonl(feature_dir / "records.jsonl")}
    output: dict[str, dict[str, Any]] = {}
    for group, items in _group_cases(cases).items():
        entropies: list[float] = []
        quantizers = 0
        for case in items:
            record = records[str(case["utterance_id"])]
            values = np.load(record["array_path"])["values"]
            quantizers = int(values.shape[1])
            entropies.extend(_entropy(values[:, index]) for index in range(values.shape[1]))
        output[group] = {
            "records": len(items),
            "quantizers": quantizers,
            "mean_entropy": mean(entropies) if entropies else None,
        }
    return output


def _is_rvq_feature_dir(feature_dir: Path) -> bool:
    first = next(iter(_load_jsonl(feature_dir / "records.jsonl")), None)
    return bool(first and first.get("value_kind") == "rvq_codes")


def _audio_stats(path_value: object) -> dict[str, float | int | None]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.is_file() or path.suffix.lower() != ".wav":
        return {}
    try:
        with wave.open(str(path), "rb") as f:
            frames = f.readframes(f.getnframes())
            sample_width = f.getsampwidth()
            channels = f.getnchannels()
            sample_rate = f.getframerate()
    except (OSError, EOFError, wave.Error):
        return {}
    if sample_width != 2 or not frames:
        return {"sample_rate": sample_rate, "channels": channels}
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    abs_audio = np.abs(audio)
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else None,
        "peak_abs": float(abs_audio.max()) if audio.size else None,
        "silence_fraction": float((abs_audio < 1e-3).mean()) if audio.size else None,
    }


def _group_cases(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["contrast_group"])].append(case)
    order = [
        "contrast_fixes_reference",
        "reference_fixes_contrast",
        "both_correct",
        "both_wrong",
    ]
    return {key: grouped[key] for key in order if key in grouped}


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(values), "mean": mean(values), "min": min(values), "max": max(values)}


def _mean_present(values: object) -> float | None:
    items = [float(value) for value in values if value is not None]
    return mean(items) if items else None


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _entropy(values: np.ndarray) -> float:
    counts = np.bincount(values.astype(np.int64))
    probs = counts[counts > 0].astype(np.float64) / max(1, values.size)
    return float(-(probs * np.log2(probs)).sum())


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"
