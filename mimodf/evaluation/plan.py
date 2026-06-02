"""Dry-run evaluation planning.

This module validates the file/protocol contract for a future evaluation run
without importing Torch, loading a model, touching GPUs, or writing score files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from mimodf.config import load_experiment_config

EvalTrack = Literal["LA", "DF"]


@dataclass(frozen=True)
class EvalPathCheck:
    name: str
    path: str
    required: bool
    exists: bool
    kind: str
    status: str


@dataclass(frozen=True)
class EvaluationPlan:
    config_path: str
    model: str
    strategy: str
    seed: int
    track: EvalTrack
    checkpoint: str
    eval_root: str
    score_out: str
    scorer: str | None
    phase: str
    protocol_eval_set: str
    scorer_name: str | None
    path_checks: tuple[EvalPathCheck, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return all(check.status == "ok" for check in self.path_checks)


def build_evaluation_plan(
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    eval_root: str | Path,
    score_out: str | Path,
    track: EvalTrack,
    scorer: str | Path | None = None,
    phase: str = "eval",
) -> EvaluationPlan:
    """Validate an evaluation launch plan without running inference."""

    if track not in {"LA", "DF"}:
        raise ValueError("track must be 'LA' or 'DF'")
    if not phase:
        raise ValueError("phase must be non-empty")

    cfg = load_experiment_config(config_path)
    scorer_path = Path(scorer) if scorer is not None else None
    checks = [
        _path_check("config", Path(config_path), required=True, kind="file"),
        _path_check("checkpoint", Path(checkpoint), required=True, kind="file"),
        _path_check("eval_root", Path(eval_root), required=True, kind="dir"),
        _score_out_check(Path(score_out)),
    ]
    if track == "LA":
        scorer_path = scorer_path or Path("SSL_Anti-spoofing/evaluate_2021_LA.py")
        checks.append(_path_check("scorer", scorer_path, required=True, kind="file"))
        scorer_name = cfg.scorer.la
    else:
        scorer_name = cfg.scorer.df
        if scorer_path is not None:
            checks.append(_path_check("scorer", scorer_path, required=True, kind="file"))

    return EvaluationPlan(
        config_path=str(config_path),
        model=cfg.model,
        strategy=cfg.strategy,
        seed=cfg.seed,
        track=track,
        checkpoint=str(checkpoint),
        eval_root=str(eval_root),
        score_out=str(score_out),
        scorer=None if scorer_path is None else str(scorer_path),
        phase=phase,
        protocol_eval_set=cfg.protocol.eval_set,
        scorer_name=scorer_name,
        path_checks=tuple(checks),
    )


def render_evaluation_plan_json(plan: EvaluationPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2) + "\n"


def render_evaluation_plan_markdown(plan: EvaluationPlan) -> str:
    lines = [
        "# Evaluation dry-run plan",
        "",
        f"Status: **{'pass' if plan.passed else 'fail'}**",
        "",
        f"Model: `{plan.model}`",
        f"Strategy: `{plan.strategy}`",
        f"Seed: `{plan.seed}`",
        f"Track: `{plan.track}`",
        f"Protocol eval set: `{plan.protocol_eval_set}`",
        f"Configured scorer: `{plan.scorer_name or 'none'}`",
        "",
        "| Check | Required | Kind | Status | Path |",
        "|---|---:|---|---|---|",
    ]
    for check in plan.path_checks:
        lines.append(
            f"| {check.name} | {'yes' if check.required else 'no'} | {check.kind} | "
            f"{check.status} | `{check.path}` |"
        )
    return "\n".join(lines) + "\n"


def _path_check(name: str, path: Path, *, required: bool, kind: str) -> EvalPathCheck:
    if kind == "file":
        exists = path.is_file()
    elif kind == "dir":
        exists = path.is_dir()
    else:
        raise ValueError(f"unsupported path check kind: {kind}")
    return EvalPathCheck(
        name=name,
        path=str(path),
        required=required,
        exists=exists,
        kind=kind,
        status="ok" if exists or not required else "missing",
    )


def _score_out_check(path: Path) -> EvalPathCheck:
    if path.exists():
        status = "exists"
        ok = False
    else:
        blocking_parent = _first_blocking_parent(path.parent)
        if blocking_parent is not None:
            status = f"parent_not_dir:{blocking_parent}"
            ok = False
        else:
            status = "ok"
            ok = True
    return EvalPathCheck(
        name="score_out",
        path=str(path),
        required=True,
        exists=ok,
        kind="new_file",
        status=status,
    )


def _first_blocking_parent(parent: Path) -> Path | None:
    current = parent
    missing: list[Path] = []
    while not current.exists():
        missing.append(current)
        if current == current.parent:
            return current
        current = current.parent
    if not current.is_dir():
        return current
    return None
