"""Case-level mechanism analysis over existing feature-probe predictions."""

from __future__ import annotations

import json
import time
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from mimodf.features.common import command_argv, git_revision
from mimodf.features.predictions import PredictionSource

MECHANISM_REPORT_SCHEMA = "mimodf-feature-mechanism-analysis/v1"


@dataclass(frozen=True)
class MechanismAnalysisSettings:
    predictions: tuple[PredictionSource, ...]
    protocol: Path
    out_dir: Path
    reference: str
    positive_label: str = "spoof"
    source_model: str | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class MechanismAnalysisResult:
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


def run_mechanism_analysis(settings: MechanismAnalysisSettings) -> MechanismAnalysisResult:
    _validate_settings(settings)
    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    protocol = _load_protocol(settings.protocol)
    loaded = {source.name: _load_predictions(source.path) for source in settings.predictions}
    names = [source.name for source in settings.predictions]
    shared_ids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    if settings.source_model is not None:
        shared_ids = [
            utterance_id
            for utterance_id in shared_ids
            if str(protocol.get(utterance_id, {}).get("source_model")) == settings.source_model
        ]
    if not shared_ids:
        raise ValueError("prediction sources have no shared utterance_id values")

    cases = [
        _case_record(
            utterance_id,
            loaded=loaded,
            protocol=protocol,
            systems=names,
            positive_label=settings.positive_label,
        )
        for utterance_id in shared_ids
    ]
    summary = {
        "schema": MECHANISM_REPORT_SCHEMA,
        "predictions": {source.name: str(source.path) for source in settings.predictions},
        "protocol": str(settings.protocol),
        "source_model": settings.source_model,
        "reference": settings.reference,
        "positive_label": settings.positive_label,
        "records": len(cases),
        "support": _support(cases),
        "systems": _system_summary(cases, names, positive_label=settings.positive_label),
        "pairs": _pair_summary(cases, names),
        "reference_contrasts": _reference_contrasts(cases, names, settings.reference),
        "groups": {
            "label": _group_summary(cases, names, "label"),
            "decoder_type": _group_summary(cases, names, "decoder_type"),
            "quantizer_type": _group_summary(cases, names, "quantizer_type"),
            "duration_bucket": _group_summary(cases, names, "duration_bucket"),
        },
        "score_distributions": _score_distributions(
            cases, names, positive_label=settings.positive_label
        ),
        "duration": _duration_summary(cases),
        "started_unix": started,
        "finished_unix": time.time(),
        "git_revision": git_revision(),
        "command_argv": command_argv(),
        "caveats": [
            "case-level diagnostic over existing feature-probe predictions",
            "no feature extraction, model training, HPO, or external evaluation is run",
            "linear-probe predictions are diagnostic and not deployable detector outputs",
        ],
    }

    cases_path = settings.out_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, sort_keys=True) + "\n")
    summary_path = settings.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_path = settings.out_dir / "report.md"
    report_path.write_text(render_mechanism_report(summary), encoding="utf-8")
    return MechanismAnalysisResult(report_path, summary_path, cases_path, len(cases))


def render_mechanism_report(summary: dict[str, Any]) -> str:
    lines = [
        "# CLAMTTS mechanism analysis",
        "",
        f"Records: {summary['records']}",
        f"Reference: `{summary['reference']}`",
        f"Support: `{json.dumps(summary['support'], sort_keys=True)}`",
        "",
        "## System outcomes",
        "",
        "| System | Accuracy | Wrong | False positive | False negative | Mean spoof score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["systems"].items():
        lines.append(
            f"| {name} | {item['accuracy']:.3f} | {item['wrong']} | "
            f"{item['false_positive']} | {item['false_negative']} | "
            f"{_fmt(item['positive_score_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Reference contrasts",
            "",
            "| System vs reference | Both correct | Both wrong | System fixes reference | Reference fixes system |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary["reference_contrasts"].items():
        if name == summary["reference"]:
            continue
        lines.append(
            f"| {name} vs {summary['reference']} | {item['both_correct']} | "
            f"{item['both_wrong']} | {item['system_correct_reference_wrong']} | "
            f"{item['reference_correct_system_wrong']} |"
        )

    lines.extend(
        [
            "",
            "## Pair overlap",
            "",
            "| Pair | Both wrong | Left only correct | Right only correct | Disagreements |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for pair, item in summary["pairs"].items():
        lines.append(
            f"| {pair} | {item['both_wrong']} | {item['left_only_correct']} | "
            f"{item['right_only_correct']} | {item['disagreements']} |"
        )

    lines.extend(["", "## Grouped wrong counts", ""])
    for field, groups in summary["groups"].items():
        lines.extend([f"### {field}", "", "| Group | Records | Wrong counts |", "|---|---:|---|"])
        for value, item in groups.items():
            lines.append(
                f"| {value} | {item['records']} | `{json.dumps(item['wrong'], sort_keys=True)}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- This is a mechanism diagnostic, not a training/evaluation result.",
            "- Counts are over CLAMTTS held-out probe predictions only.",
            "- A mechanism claim needs a named explanation from the observed case pattern.",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["caveats"])
    lines.append("")
    return "\n".join(lines)


def _validate_settings(settings: MechanismAnalysisSettings) -> None:
    if len(settings.predictions) < 2:
        raise ValueError("at least two prediction sources are required")
    names = [source.name for source in settings.predictions]
    if len(names) != len(set(names)):
        raise ValueError("prediction source names must be unique")
    if settings.reference not in names:
        raise ValueError("reference must match one prediction source name")
    if not settings.protocol.is_file():
        raise FileNotFoundError(settings.protocol)
    for source in settings.predictions:
        if not source.path.is_file():
            raise FileNotFoundError(source.path)


def _load_protocol(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = str(record.get("utterance_id"))
            if utterance_id in rows:
                raise ValueError(f"duplicate protocol utterance_id: {utterance_id}")
            rows[utterance_id] = record
    return rows


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = str(record.get("utterance_id"))
            if utterance_id in rows:
                raise ValueError(f"duplicate prediction utterance_id in {path}: {utterance_id}")
            rows[utterance_id] = record
    return rows


def _case_record(
    utterance_id: str,
    *,
    loaded: dict[str, dict[str, dict[str, Any]]],
    protocol: dict[str, dict[str, Any]],
    systems: list[str],
    positive_label: str,
) -> dict[str, Any]:
    first = loaded[systems[0]][utterance_id]
    target = str(first.get("target"))
    metadata = {**first, **protocol.get(utterance_id, {})}
    audio = _audio_metadata(metadata.get("audio_path"))
    system_records: dict[str, dict[str, object]] = {}
    for system in systems:
        record = loaded[system][utterance_id]
        if str(record.get("target")) != target:
            raise ValueError(f"target mismatch for {utterance_id}")
        prediction = str(record.get("prediction"))
        probabilities = record.get("probabilities") or {}
        system_records[system] = {
            "prediction": prediction,
            "correct": prediction == target,
            "positive_probability": _optional_float(probabilities.get(positive_label)),
        }
    return {
        "schema": "mimodf-feature-mechanism-case/v1",
        "utterance_id": utterance_id,
        "target": target,
        "label": metadata.get("label"),
        "source_model": metadata.get("source_model"),
        "decoder_type": metadata.get("decoder_type"),
        "quantizer_type": metadata.get("quantizer_type"),
        "auxiliary_objective": metadata.get("auxiliary_objective"),
        "audio": audio,
        "duration_bucket": _duration_bucket(audio.get("duration_sec")),
        "systems": system_records,
    }


def _audio_metadata(path_value: object) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.is_file() or path.suffix.lower() != ".wav":
        return {}
    try:
        with wave.open(str(path), "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return {
                "sample_rate": rate,
                "frames": frames,
                "duration_sec": frames / rate if rate else None,
            }
    except (OSError, EOFError, wave.Error):
        return {}


def _system_summary(
    cases: list[dict[str, Any]], names: list[str], *, positive_label: str
) -> dict[str, dict[str, float | int | None]]:
    del positive_label
    output: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        correct = [bool(case["systems"][name]["correct"]) for case in cases]
        false_positive = sum(
            case["target"] != "spoof" and case["systems"][name]["prediction"] == "spoof"
            for case in cases
        )
        false_negative = sum(
            case["target"] == "spoof" and case["systems"][name]["prediction"] != "spoof"
            for case in cases
        )
        scores = [
            case["systems"][name]["positive_probability"]
            for case in cases
            if case["systems"][name]["positive_probability"] is not None
        ]
        output[name] = {
            "accuracy": sum(correct) / len(correct),
            "correct": int(sum(correct)),
            "wrong": int(len(correct) - sum(correct)),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "positive_score_mean": mean(scores) if scores else None,
        }
    return output


def _pair_summary(cases: list[dict[str, Any]], names: list[str]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            both_wrong = left_only = right_only = disagreements = 0
            for case in cases:
                left_correct = bool(case["systems"][left]["correct"])
                right_correct = bool(case["systems"][right]["correct"])
                both_wrong += int(not left_correct and not right_correct)
                left_only += int(left_correct and not right_correct)
                right_only += int(not left_correct and right_correct)
                disagreements += int(
                    case["systems"][left]["prediction"] != case["systems"][right]["prediction"]
                )
            output[f"{left} vs {right}"] = {
                "both_wrong": both_wrong,
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "disagreements": disagreements,
            }
    return output


def _reference_contrasts(
    cases: list[dict[str, Any]], names: list[str], reference: str
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for name in names:
        both_correct = both_wrong = system_fix = reference_fix = 0
        for case in cases:
            system_correct = bool(case["systems"][name]["correct"])
            reference_correct = bool(case["systems"][reference]["correct"])
            both_correct += int(system_correct and reference_correct)
            both_wrong += int(not system_correct and not reference_correct)
            system_fix += int(system_correct and not reference_correct)
            reference_fix += int(not system_correct and reference_correct)
        output[name] = {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "system_correct_reference_wrong": system_fix,
            "reference_correct_system_wrong": reference_fix,
        }
    return output


def _group_summary(
    cases: list[dict[str, Any]], names: list[str], field: str
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case.get(field))].append(case)
    return {
        value: {
            "records": len(items),
            "wrong": {
                name: sum(not item["systems"][name]["correct"] for item in items) for name in names
            },
        }
        for value, items in sorted(grouped.items())
    }


def _score_distributions(
    cases: list[dict[str, Any]], names: list[str], *, positive_label: str
) -> dict[str, Any]:
    del positive_label
    output: dict[str, Any] = {}
    for name in names:
        by_label: dict[str, list[float]] = defaultdict(list)
        by_correctness: dict[str, list[float]] = defaultdict(list)
        for case in cases:
            value = case["systems"][name]["positive_probability"]
            if value is None:
                continue
            score = float(value)
            by_label[str(case["target"])].append(score)
            key = "correct" if case["systems"][name]["correct"] else "wrong"
            by_correctness[key].append(score)
        output[name] = {
            "by_label": _mean_groups(by_label),
            "by_correctness": _mean_groups(by_correctness),
        }
    return output


def _duration_summary(cases: list[dict[str, Any]]) -> dict[str, object]:
    durations = [
        case["audio"].get("duration_sec")
        for case in cases
        if case.get("audio") and case["audio"].get("duration_sec") is not None
    ]
    if not durations:
        return {"readable": 0, "records": len(cases)}
    values = [float(item) for item in durations]
    return {
        "readable": len(values),
        "records": len(cases),
        "mean_duration_sec": mean(values),
        "min_duration_sec": min(values),
        "max_duration_sec": max(values),
    }


def _support(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        key = str(case["target"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _duration_bucket(value: object) -> str:
    if value is None:
        return "unknown"
    seconds = float(value)
    if seconds < 3.0:
        return "short_lt3s"
    if seconds < 5.0:
        return "mid_3_5s"
    return "long_ge5s"


def _mean_groups(groups: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {"count": len(values), "mean": mean(values), "min": min(values), "max": max(values)}
        for key, values in sorted(groups.items())
        if values
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"
