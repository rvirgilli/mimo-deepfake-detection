"""Official ASVspoof scoring wrappers.

The project produced many result files with project-local ``min t-DCF`` values
that are on the wrong scale for the paper table. This module only accepts the
official LA evaluator format (``min_tDCF: ...``) for table tDCF values.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


class OfficialScoringError(ValueError):
    """Raised when official scoring output is missing or malformed."""


@dataclass(frozen=True)
class OfficialLAScore:
    score_file: str
    eval_root: str
    phase: str
    scorer_path: str
    eer_percent: float
    min_tdcf: float
    command: tuple[str, ...]
    score_sha256: str
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectResultFile:
    path: str
    eer_percent: float | None
    project_min_tdcf: float | None


def run_official_la_scorer(
    score_file: str | Path,
    eval_root: str | Path,
    *,
    scorer_path: str | Path = "SSL_Anti-spoofing/evaluate_2021_LA.py",
    phase: str = "eval",
    python: str = "python",
    cwd: str | Path | None = None,
) -> OfficialLAScore:
    """Run the official ASVspoof2021 LA evaluator and parse its output."""

    score_path = Path(score_file)
    scorer = Path(scorer_path)
    command = (python, str(scorer), str(score_path), str(eval_root), phase)
    result = subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise OfficialScoringError(
            f"official scorer failed with exit {result.returncode}: {result.stderr.strip()}"
        )

    parsed = parse_official_la_output(result.stdout)
    return OfficialLAScore(
        score_file=str(score_path),
        eval_root=str(eval_root),
        phase=phase,
        scorer_path=str(scorer),
        eer_percent=parsed.eer_percent,
        min_tdcf=parsed.min_tdcf,
        command=command,
        score_sha256=_sha256(score_path),
        stdout=result.stdout,
        stderr=result.stderr,
    )


def parse_official_la_output(text: str) -> OfficialLAScore:
    """Parse official LA evaluator stdout.

    Official output uses ``min_tDCF`` with an underscore. Project-local result
    files use ``min t-DCF`` and are intentionally rejected here.
    """

    min_tdcf = _search_float(r"^\s*min_tDCF:\s*([0-9]*\.?[0-9]+)\s*$", text)
    eer = _search_float(r"^\s*eer:\s*([0-9]*\.?[0-9]+)\s*$", text)
    if min_tdcf is None or eer is None:
        raise OfficialScoringError(
            "official LA output must contain 'min_tDCF:' and 'eer:' lines; "
            "project 'min t-DCF:' result files are not accepted"
        )
    return OfficialLAScore(
        score_file="",
        eval_root="",
        phase="",
        scorer_path="",
        eer_percent=eer,
        min_tdcf=min_tdcf,
        command=(),
        score_sha256="",
        stdout=text,
    )


def parse_project_result_file(path: str | Path) -> ProjectResultFile:
    """Parse a project result file for diagnostics only.

    The returned ``project_min_tdcf`` must not be used as official table tDCF.
    It is kept only so tests/audits can detect wrong-scale values explicitly.
    """

    text = Path(path).read_text()
    return ProjectResultFile(
        path=str(path),
        eer_percent=_search_float(r"^\s*EER:\s*([0-9]*\.?[0-9]+)%?\s*$", text),
        project_min_tdcf=_search_float(r"^\s*min t-DCF:\s*([0-9]*\.?[0-9]+)\s*$", text),
    )


def _search_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        return None
    return float(match.group(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
