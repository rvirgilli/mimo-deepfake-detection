"""Compare ASVspoof score files for reproduction audits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mimodf.scoring.write_scores import read_score_file


@dataclass(frozen=True)
class ScoreComparison:
    candidate: str
    reference: str
    candidate_count: int
    reference_count: int
    common_count: int
    candidate_only_count: int
    reference_only_count: int
    max_abs_diff: float | None
    mean_abs_diff: float | None
    tolerance: float | None
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_score_files(
    candidate: str | Path,
    reference: str | Path,
    *,
    tolerance: float | None = None,
) -> ScoreComparison:
    candidate_scores = {record.utterance_id: record.score for record in read_score_file(candidate)}
    reference_scores = {record.utterance_id: record.score for record in read_score_file(reference)}
    common = sorted(candidate_scores.keys() & reference_scores.keys())
    diffs = [abs(candidate_scores[utt] - reference_scores[utt]) for utt in common]
    max_abs_diff = max(diffs) if diffs else None
    mean_abs_diff = sum(diffs) / len(diffs) if diffs else None
    passed = bool(common)
    if tolerance is not None:
        passed = passed and max_abs_diff is not None and max_abs_diff <= tolerance
    return ScoreComparison(
        candidate=str(candidate),
        reference=str(reference),
        candidate_count=len(candidate_scores),
        reference_count=len(reference_scores),
        common_count=len(common),
        candidate_only_count=len(candidate_scores.keys() - reference_scores.keys()),
        reference_only_count=len(reference_scores.keys() - candidate_scores.keys()),
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        tolerance=tolerance,
        passed=passed,
    )


def render_score_comparison_json(comparison: ScoreComparison) -> str:
    return json.dumps(comparison.to_dict(), indent=2) + "\n"


def render_score_comparison_markdown(comparison: ScoreComparison) -> str:
    lines = [
        "# Score-file comparison",
        "",
        f"Status: **{'pass' if comparison.passed else 'fail'}**",
        "",
        f"Candidate: `{comparison.candidate}`",
        f"Reference: `{comparison.reference}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| candidate_count | {comparison.candidate_count} |",
        f"| reference_count | {comparison.reference_count} |",
        f"| common_count | {comparison.common_count} |",
        f"| candidate_only_count | {comparison.candidate_only_count} |",
        f"| reference_only_count | {comparison.reference_only_count} |",
        f"| max_abs_diff | {_format_optional_float(comparison.max_abs_diff)} |",
        f"| mean_abs_diff | {_format_optional_float(comparison.mean_abs_diff)} |",
    ]
    if comparison.tolerance is not None:
        lines.append(f"| tolerance | {comparison.tolerance:.10g} |")
    return "\n".join(lines) + "\n"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.10g}"
