"""Prediction comparison reports for Wave 1 feature probes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimodf.features.common import command_argv, git_revision

PREDICTION_COMPARISON_SCHEMA = "mimodf-feature-prediction-comparison/v1"


@dataclass(frozen=True)
class PredictionSource:
    name: str
    path: Path


@dataclass(frozen=True)
class PredictionComparisonSettings:
    sources: tuple[PredictionSource, ...]
    out_dir: Path
    overwrite: bool = False


@dataclass(frozen=True)
class PredictionComparisonResult:
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


def parse_prediction_source(value: str) -> PredictionSource:
    if "=" not in value:
        raise ValueError("prediction source must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("prediction source must be NAME=PATH")
    return PredictionSource(name=name, path=Path(path))


def compare_predictions(settings: PredictionComparisonSettings) -> PredictionComparisonResult:
    if len(settings.sources) < 2:
        raise ValueError("at least two prediction sources are required")
    if settings.out_dir.exists() and any(settings.out_dir.iterdir()) and not settings.overwrite:
        raise FileExistsError(settings.out_dir)
    settings.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    loaded = {source.name: _load_by_utterance(source.path) for source in settings.sources}
    shared_ids = sorted(set.intersection(*(set(records) for records in loaded.values())))
    if not shared_ids:
        raise ValueError("prediction sources have no shared utterance_id values")

    cases = [_case_record(utterance_id, loaded) for utterance_id in shared_ids]
    summary = {
        "schema": PREDICTION_COMPARISON_SCHEMA,
        "sources": {source.name: str(source.path) for source in settings.sources},
        "records": len(cases),
        "system_summary": _system_summary(cases, [source.name for source in settings.sources]),
        "pair_summary": _pair_summary(cases, [source.name for source in settings.sources]),
        "group_summary": _group_summary(cases, [source.name for source in settings.sources]),
        "started_unix": started,
        "finished_unix": time.time(),
        "git_revision": git_revision(),
        "command_argv": command_argv(),
        "caveats": [
            "diagnostic error-case comparison over already-generated probe predictions",
            "source rows must come from the same held-out/test set for direct comparison",
        ],
    }

    cases_path = settings.out_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, sort_keys=True) + "\n")
    summary_path = settings.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = settings.out_dir / "report.md"
    report_path.write_text(render_prediction_comparison(summary), encoding="utf-8")
    return PredictionComparisonResult(report_path, summary_path, cases_path, len(cases))


def render_prediction_comparison(summary: dict[str, Any]) -> str:
    lines = [
        "# Prediction comparison report",
        "",
        f"Records: {summary['records']}",
        "",
        "## Systems",
        "",
        "| System | Accuracy | Wrong | Unique correct | Unique wrong |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary["system_summary"].items():
        lines.append(
            f"| {name} | {item['accuracy']:.4f} | {item['wrong']} | "
            f"{item['unique_correct']} | {item['unique_wrong']} |"
        )
    lines.extend(
        [
            "",
            "## Pair disagreement",
            "",
            "| Pair | Disagreements | Both wrong | Left only correct | Right only correct |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for pair, item in summary["pair_summary"].items():
        lines.append(
            f"| {pair} | {item['disagreements']} | {item['both_wrong']} | "
            f"{item['left_only_correct']} | {item['right_only_correct']} |"
        )
    lines.extend(["", "## Groups", ""])
    for field, groups in summary["group_summary"].items():
        lines.extend(
            [f"### {field}", "", "| Value | Records | Systems wrong counts |", "|---|---:|---|"]
        )
        for value, item in groups.items():
            lines.append(
                f"| {value} | {item['records']} | `{json.dumps(item['wrong'], sort_keys=True)}` |"
            )
        lines.append("")
    return "\n".join(lines)


def _load_by_utterance(path: Path) -> dict[str, dict[str, Any]]:
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
                raise ValueError(f"duplicate utterance_id in {path}: {utterance_id}")
            rows[utterance_id] = record
    return rows


def _case_record(utterance_id: str, loaded: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    first = next(iter(loaded.values()))[utterance_id]
    target = first.get("target")
    systems: dict[str, dict[str, object]] = {}
    for name, rows in loaded.items():
        record = rows[utterance_id]
        if record.get("target") != target:
            raise ValueError(f"target mismatch for {utterance_id}")
        prediction = record.get("prediction", record.get("fusion_prediction"))
        systems[name] = {
            "prediction": prediction,
            "correct": prediction == target,
            "probabilities": record.get("probabilities", record.get("fusion_probabilities")),
        }
    return {
        "schema": "mimodf-feature-prediction-comparison-case/v1",
        "utterance_id": utterance_id,
        "target": target,
        "label": first.get("label"),
        "source_model": first.get("source_model"),
        "quantizer_type": first.get("quantizer_type"),
        "auxiliary_objective": first.get("auxiliary_objective"),
        "decoder_type": first.get("decoder_type"),
        "systems": systems,
    }


def _system_summary(
    cases: list[dict[str, Any]], names: list[str]
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for name in names:
        correct = [bool(case["systems"][name]["correct"]) for case in cases]
        unique_correct = sum(
            current
            and all(not case["systems"][other]["correct"] for other in names if other != name)
            for current, case in zip(correct, cases, strict=True)
        )
        unique_wrong = sum(
            (not current)
            and all(case["systems"][other]["correct"] for other in names if other != name)
            for current, case in zip(correct, cases, strict=True)
        )
        output[name] = {
            "accuracy": sum(correct) / len(correct),
            "correct": int(sum(correct)),
            "wrong": int(len(correct) - sum(correct)),
            "unique_correct": int(unique_correct),
            "unique_wrong": int(unique_wrong),
        }
    return output


def _pair_summary(cases: list[dict[str, Any]], names: list[str]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            disagreements = both_wrong = left_only = right_only = 0
            for case in cases:
                left_pred = case["systems"][left]["prediction"]
                right_pred = case["systems"][right]["prediction"]
                left_correct = bool(case["systems"][left]["correct"])
                right_correct = bool(case["systems"][right]["correct"])
                disagreements += int(left_pred != right_pred)
                both_wrong += int(not left_correct and not right_correct)
                left_only += int(left_correct and not right_correct)
                right_only += int(not left_correct and right_correct)
            output[f"{left} vs {right}"] = {
                "disagreements": disagreements,
                "both_wrong": both_wrong,
                "left_only_correct": left_only,
                "right_only_correct": right_only,
            }
    return output


def _group_summary(
    cases: list[dict[str, Any]], names: list[str]
) -> dict[str, dict[str, dict[str, object]]]:
    fields = ["label", "source_model", "quantizer_type", "decoder_type"]
    output: dict[str, dict[str, dict[str, object]]] = {}
    for field in fields:
        groups: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            value = str(case.get(field))
            groups.setdefault(value, []).append(case)
        output[field] = {}
        for value, items in sorted(groups.items()):
            output[field][value] = {
                "records": len(items),
                "wrong": {
                    name: sum(not item["systems"][name]["correct"] for item in items)
                    for name in names
                },
            }
    return output
