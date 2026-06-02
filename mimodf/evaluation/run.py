"""Controlled evaluation execution.

Generic runner code stays framework-agnostic. Real Torch/legacy construction lives
in `legacy_components.py` so default tests can exercise execution semantics with
fakes and without model imports.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mimodf.config import load_experiment_config
from mimodf.evaluation.plan import EvaluationPlan
from mimodf.experiments.execution import (
    PreparedExperimentRun,
    complete_experiment_run,
    fail_experiment_run,
)
from mimodf.scoring.evaluate import EvaluationBatch, PredictBatch, write_scores_from_batches
from mimodf.scoring.official import run_official_la_scorer
from mimodf.training.manifest import TrainingManifest


@dataclass(frozen=True)
class EvaluationComponents:
    batches: Iterable[EvaluationBatch[Any]]
    predict_batch: PredictBatch[Any]


@dataclass(frozen=True)
class EvaluationRunSettings:
    overwrite: bool = False
    score_official: bool = False
    official_result_out: str | None = None
    manifest_out: str | None = None
    python: str = sys.executable
    experiment_run: PreparedExperimentRun | None = None


@dataclass(frozen=True)
class EvaluationRunResult:
    score_file: str
    manifest_file: str | None
    official_result_file: str | None
    official_result: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_evaluation(
    plan: EvaluationPlan,
    components: EvaluationComponents,
    settings: EvaluationRunSettings,
    *,
    command: list[str] | None = None,
) -> EvaluationRunResult:
    """Run evaluation from pre-built components and write auditable outputs."""

    if not plan.passed:
        raise ValueError("evaluation plan has failing path checks")
    score_path = Path(plan.score_out)
    if score_path.exists() and not settings.overwrite:
        raise FileExistsError(score_path)

    manifest = None
    manifest_path = Path(settings.manifest_out) if settings.manifest_out else None
    if manifest_path is not None:
        manifest = TrainingManifest.start(
            load_experiment_config(plan.config_path),
            command=command,
            working_dir=Path.cwd(),
        )

    try:
        write_scores_from_batches(components.batches, components.predict_batch, score_path)
        official_result = None
        official_result_path = None
        if settings.score_official:
            if plan.track != "LA":
                raise ValueError("official scoring is currently supported for LA only")
            if plan.scorer is None:
                raise ValueError("official scoring requires a scorer path")
            official_result = run_official_la_scorer(
                score_path,
                plan.eval_root,
                scorer_path=plan.scorer,
                phase=plan.phase,
                python=settings.python,
            )
            if settings.official_result_out is not None:
                official_result_path = Path(settings.official_result_out)
                official_result_path.parent.mkdir(parents=True, exist_ok=True)
                official_result_path.write_text(
                    json.dumps(official_result.to_dict(), indent=2) + "\n"
                )
        result = EvaluationRunResult(
            score_file=str(score_path),
            manifest_file=None if manifest_path is None else str(manifest_path),
            official_result_file=None
            if official_result_path is None
            else str(official_result_path),
            official_result=None if official_result is None else official_result.to_dict(),
        )
        artifacts = {"score_file": str(score_path)}
        if official_result_path is not None:
            artifacts["official_result"] = str(official_result_path)
        metrics = {
            "track": plan.track,
            "score_official": settings.score_official,
        }
        if manifest is not None and manifest_path is not None:
            manifest.complete(
                metrics=metrics,
                artifacts=artifacts,
            )
            manifest.save(manifest_path)
        if settings.experiment_run is not None:
            complete_experiment_run(
                settings.experiment_run,
                metrics=metrics,
                artifacts=artifacts,
            )
        return result
    except BaseException as exc:
        if manifest is not None and manifest_path is not None:
            manifest.fail(exc, metrics={"track": plan.track})
            manifest.save(manifest_path)
        if settings.experiment_run is not None:
            fail_experiment_run(settings.experiment_run, exc, metrics={"track": plan.track})
        raise
