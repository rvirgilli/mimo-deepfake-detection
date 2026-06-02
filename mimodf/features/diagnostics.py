"""Diagnostics over already-generated feature-probe predictions."""

from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimodf.features.common import command_argv, git_revision

PREDICTION_DIAGNOSTIC_SCHEMA = "mimodf-feature-prediction-diagnostic/v1"


@dataclass(frozen=True)
class PredictionDiagnosticSettings:
    predictions_root: Path
    protocol: Path
    out_dir: Path
    sources: tuple[str, ...]
    systems: tuple[str, ...]
    positive_label: str = "spoof"
    overwrite: bool = False
    include_audio_metadata: bool = True


@dataclass(frozen=True)
class PredictionDiagnosticResult:
    report_path: Path
    summary_path: Path
    cases_path: Path
    records: int
    sources: int

    def to_dict(self) -> dict[str, object]:
        return {
            "report": str(self.report_path),
            "summary": str(self.summary_path),
            "cases": str(self.cases_path),
            "records": self.records,
            "sources": self.sources,
        }


def run_prediction_diagnostics(
    settings: PredictionDiagnosticSettings,
) -> PredictionDiagnosticResult:
    _validate_settings(settings)
    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    protocol = _load_protocol(settings.protocol)
    cases: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    for source in settings.sources:
        loaded = {
            system: _load_predictions(_prediction_path(settings.predictions_root, source, system))
            for system in settings.systems
        }
        shared_ids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
        if not shared_ids:
            raise ValueError(f"no shared utterance_id values for source {source}")
        source_cases = [
            _case_record(
                source=source,
                utterance_id=utterance_id,
                loaded=loaded,
                protocol=protocol,
                systems=list(settings.systems),
                positive_label=settings.positive_label,
                include_audio_metadata=settings.include_audio_metadata,
            )
            for utterance_id in shared_ids
        ]
        cases.extend(source_cases)
        source_summaries[source] = _source_summary(
            source_cases,
            systems=list(settings.systems),
            metrics_root=settings.predictions_root / source,
        )

    summary = {
        "schema": PREDICTION_DIAGNOSTIC_SCHEMA,
        "predictions_root": str(settings.predictions_root),
        "protocol": str(settings.protocol),
        "sources": list(settings.sources),
        "systems": list(settings.systems),
        "positive_label": settings.positive_label,
        "records": len(cases),
        "source_summary": source_summaries,
        "contrast_summary": _contrast_summary(source_summaries, systems=list(settings.systems)),
        "started_unix": started,
        "finished_unix": time.time(),
        "git_revision": git_revision(),
        "command_argv": command_argv(),
        "caveats": [
            "diagnostic summary over already-generated feature-probe predictions",
            "uses hard probe predictions and source-level support; not full model training/evaluation",
            "audio metadata is opportunistic and only present when local WAV files are readable",
        ],
    }

    cases_path = settings.out_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, sort_keys=True) + "\n")
    summary_path = settings.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = settings.out_dir / "report.md"
    report_path.write_text(render_prediction_diagnostic_report(summary), encoding="utf-8")
    return PredictionDiagnosticResult(
        report_path=report_path,
        summary_path=summary_path,
        cases_path=cases_path,
        records=len(cases),
        sources=len(settings.sources),
    )


def render_prediction_diagnostic_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Feature prediction source diagnostics",
        "",
        f"Records: {summary['records']}",
        f"Sources: {', '.join(summary['sources'])}",
        f"Systems: {', '.join(summary['systems'])}",
        "",
        "## Source summary",
        "",
    ]
    for source, item in summary["source_summary"].items():
        lines.extend(
            [
                f"### {source}",
                "",
                f"Support: `{json.dumps(item['support'], sort_keys=True)}`",
                "",
                "| System | Accuracy | Wrong | False positive | False negative | EER | AUROC | Bal acc |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for system, system_summary in item["systems"].items():
            metrics = system_summary.get("metrics", {})
            lines.append(
                f"| {system} | {system_summary['accuracy']:.3f} | {system_summary['wrong']} | "
                f"{system_summary['false_positive']} | {system_summary['false_negative']} | "
                f"{_fmt_metric(metrics.get('eer'))} | {_fmt_metric(metrics.get('auroc'))} | "
                f"{_fmt_metric(metrics.get('balanced_accuracy'))} |"
            )
        lines.extend(
            [
                "",
                "Pair overlap:",
                "",
                "| Pair | Both wrong | Left only correct | Right only correct | Disagreements |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for pair, pair_summary in item["pairs"].items():
            lines.append(
                f"| {pair} | {pair_summary['both_wrong']} | {pair_summary['left_only_correct']} | "
                f"{pair_summary['right_only_correct']} | {pair_summary['disagreements']} |"
            )
        audio = item.get("audio", {})
        if audio:
            lines.extend(
                [
                    "",
                    "Audio metadata:",
                    "",
                    f"- readable WAVs: {audio['readable']}/{audio['records']}",
                    f"- mean duration sec: {_fmt_metric(audio.get('mean_duration_sec'))}",
                    f"- min/max duration sec: {_fmt_metric(audio.get('min_duration_sec'))} / {_fmt_metric(audio.get('max_duration_sec'))}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Contrast summary",
            "",
            "| System | Best source acc | Worst source acc | Wrong by source |",
            "|---|---:|---:|---|",
        ]
    )
    for system, item in summary["contrast_summary"]["systems"].items():
        lines.append(
            f"| {system} | {item['best_accuracy']:.3f} ({item['best_source']}) | "
            f"{item['worst_accuracy']:.3f} ({item['worst_source']}) | "
            f"`{json.dumps(item['wrong_by_source'], sort_keys=True)}` |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend([f"- {caveat}" for caveat in summary["caveats"]])
    lines.append("")
    return "\n".join(lines)


def _validate_settings(settings: PredictionDiagnosticSettings) -> None:
    if not settings.predictions_root.is_dir():
        raise FileNotFoundError(settings.predictions_root)
    if not settings.protocol.is_file():
        raise FileNotFoundError(settings.protocol)
    if not settings.sources:
        raise ValueError("at least one source is required")
    if len(set(settings.sources)) != len(settings.sources):
        raise ValueError("sources must not contain duplicates")
    if len(settings.systems) < 2:
        raise ValueError("at least two systems are required")
    if len(set(settings.systems)) != len(settings.systems):
        raise ValueError("systems must not contain duplicates")


def _load_protocol(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = str(record.get("utterance_id"))
            if not utterance_id or utterance_id == "None":
                raise ValueError(f"protocol row missing utterance_id in {path}")
            if utterance_id in rows:
                raise ValueError(f"duplicate protocol utterance_id: {utterance_id}")
            rows[utterance_id] = record
    return rows


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def _prediction_path(root: Path, source: str, system: str) -> Path:
    return root / source / system / "predictions.jsonl"


def _case_record(
    *,
    source: str,
    utterance_id: str,
    loaded: dict[str, dict[str, dict[str, Any]]],
    protocol: dict[str, dict[str, Any]],
    systems: list[str],
    positive_label: str,
    include_audio_metadata: bool,
) -> dict[str, Any]:
    first = loaded[systems[0]][utterance_id]
    target = str(first.get("target"))
    protocol_record = protocol.get(utterance_id, {})
    merged = {**first, **protocol_record}
    system_records = {}
    for system in systems:
        record = loaded[system][utterance_id]
        if str(record.get("target")) != target:
            raise ValueError(f"target mismatch for {utterance_id}")
        prediction = str(record.get("prediction"))
        probabilities = record.get("probabilities") or {}
        positive_probability = probabilities.get(positive_label)
        system_records[system] = {
            "prediction": prediction,
            "correct": prediction == target,
            "probabilities": probabilities,
            "positive_probability": positive_probability,
        }
    audio = _audio_metadata(protocol_record) if include_audio_metadata else {}
    return {
        "schema": "mimodf-feature-prediction-diagnostic-case/v1",
        "utterance_id": utterance_id,
        "source": source,
        "target": target,
        "label": merged.get("label"),
        "source_model": merged.get("source_model"),
        "decoder_type": merged.get("decoder_type"),
        "quantizer_type": merged.get("quantizer_type"),
        "auxiliary_objective": merged.get("auxiliary_objective"),
        "audio": audio,
        "systems": system_records,
    }


def _source_summary(
    cases: list[dict[str, Any]], *, systems: list[str], metrics_root: Path
) -> dict[str, Any]:
    return {
        "records": len(cases),
        "support": _support(cases),
        "systems": {
            system: {
                **_system_summary(cases, system),
                "metrics": _load_metrics(metrics_root / system / "metrics.json"),
            }
            for system in systems
        },
        "pairs": _pair_summary(cases, systems),
        "audio": _audio_summary(cases),
    }


def _support(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        target = str(case["target"])
        counts[target] = counts.get(target, 0) + 1
    return dict(sorted(counts.items()))


def _system_summary(cases: list[dict[str, Any]], system: str) -> dict[str, Any]:
    wrong = false_positive = false_negative = 0
    positive_scores: list[float] = []
    for case in cases:
        item = case["systems"][system]
        prediction = str(item["prediction"])
        target = str(case["target"])
        wrong += int(not item["correct"])
        false_positive += int(target == "bonafide" and prediction != target)
        false_negative += int(target != "bonafide" and prediction != target)
        score = item.get("positive_probability")
        if isinstance(score, int | float):
            positive_scores.append(float(score))
    return {
        "accuracy": (len(cases) - wrong) / len(cases),
        "correct": len(cases) - wrong,
        "wrong": wrong,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "mean_positive_probability": _mean(positive_scores),
    }


def _pair_summary(cases: list[dict[str, Any]], systems: list[str]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(systems):
        for right in systems[left_index + 1 :]:
            both_wrong = left_only = right_only = disagreements = 0
            for case in cases:
                left_item = case["systems"][left]
                right_item = case["systems"][right]
                left_correct = bool(left_item["correct"])
                right_correct = bool(right_item["correct"])
                both_wrong += int(not left_correct and not right_correct)
                left_only += int(left_correct and not right_correct)
                right_only += int(not left_correct and right_correct)
                disagreements += int(left_item["prediction"] != right_item["prediction"])
            output[f"{left} vs {right}"] = {
                "both_wrong": both_wrong,
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "disagreements": disagreements,
            }
    return output


def _contrast_summary(source_summaries: dict[str, Any], *, systems: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {"systems": {}}
    for system in systems:
        per_source = {
            source: item["systems"][system]["accuracy"] for source, item in source_summaries.items()
        }
        best_source = max(per_source, key=per_source.get)
        worst_source = min(per_source, key=per_source.get)
        output["systems"][system] = {
            "best_source": best_source,
            "best_accuracy": per_source[best_source],
            "worst_source": worst_source,
            "worst_accuracy": per_source[worst_source],
            "wrong_by_source": {
                source: item["systems"][system]["wrong"]
                for source, item in source_summaries.items()
            },
        }
    return output


def _load_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: float(data[key])
        for key in ("eer", "auroc", "balanced_accuracy", "accuracy", "macro_f1")
        if isinstance(data.get(key), int | float)
    }


def _audio_metadata(record: dict[str, Any]) -> dict[str, Any]:
    audio_path = record.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path:
        return {}
    path = Path(audio_path)
    if not path.is_file():
        return {"path": audio_path, "readable": False}
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
    except (OSError, wave.Error):
        return {"path": audio_path, "readable": False}
    duration = frames / sample_rate if sample_rate else None
    return {
        "path": audio_path,
        "readable": True,
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_sec": duration,
    }


def _audio_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [
        float(case["audio"]["duration_sec"])
        for case in cases
        if isinstance(case.get("audio"), dict)
        and isinstance(case["audio"].get("duration_sec"), int | float)
    ]
    readable = sum(
        bool(case.get("audio", {}).get("readable"))
        for case in cases
        if isinstance(case.get("audio"), dict)
    )
    if not durations:
        return {"records": len(cases), "readable": readable}
    return {
        "records": len(cases),
        "readable": readable,
        "mean_duration_sec": _mean(durations),
        "min_duration_sec": min(durations),
        "max_duration_sec": max(durations),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt_metric(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "n/a"
