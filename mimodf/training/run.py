"""Command-facing training orchestration.

This module wires the tested seams together without hiding protocol choices.  It
is intentionally thin: planning is cheap and dependency-light; actually running
training imports Torch/legacy code only after the plan is explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mimodf.config import ExperimentConfig, load_experiment_config
from mimodf.data.asvspoof import (
    ASVspoofDataSettings,
    ValidationProtocol,
    build_asvspoof_loaders,
    build_path_plan,
)
from mimodf.experiments.execution import (
    PreparedExperimentRun,
    complete_experiment_run,
    fail_experiment_run,
)
from mimodf.training.components import (
    TrainingComponents,
    build_optimizer,
    split_frontend_backend_params,
    train_with_components,
)
from mimodf.training.legacy_model import (
    LegacyFrontendSettings,
    LegacyModelSettings,
    build_legacy_frontend,
    build_legacy_model,
)
from mimodf.training.loop import TrainingRunResult, TrainLoopSettings


@dataclass(frozen=True)
class LegacyASVspoofTrainingPlan:
    """Fully explicit inputs for one legacy ASVspoof training run."""

    config: ExperimentConfig
    data: ASVspoofDataSettings
    frontend: LegacyFrontendSettings
    model: LegacyModelSettings
    loop: TrainLoopSettings
    frontend_prefix: str = "frontend"

    def to_dict(self) -> dict[str, Any]:
        plan = build_path_plan(self.data)
        return {
            "config": self.config.to_dict(),
            "data": _stringify_paths(asdict(self.data)),
            "frontend": asdict(self.frontend),
            "model": asdict(self.model),
            "loop": asdict(self.loop),
            "frontend_prefix": self.frontend_prefix,
            "required_paths": _stringify_paths(asdict(plan)),
        }


def build_legacy_asvspoof_plan(
    *,
    config_path: str | Path,
    database_path: str | Path,
    protocols_path: str | Path,
    frontend_name: str,
    validation_protocol: str,
    track: str = "LA",
    output_sample_rate: int = 16_000,
    batch_size: int = 14,
    eval_batch_size: int = 14,
    num_workers: int = 4,
    cut: int | None = 64_600,
    rawboost_algo: int = 6,
    rawboost_args: dict[str, Any] | None = None,
    frontend_checkpoint: str | None = None,
    frontend_model_path: str | None = None,
    frontend_model_name: str | None = None,
    freeze_frontend: bool | None = None,
    feature_type: str = "continuous",
    epochs: int = 1,
    device: str = "cpu",
    top_k_checkpoints: int = 1,
    max_grad_norm: float = 0.0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    frontend_prefix: str = "frontend",
) -> LegacyASVspoofTrainingPlan:
    """Create a run plan without touching datasets, Torch, or model weights."""

    if validation_protocol not in {"asvspoof2021_fast", "asvspoof2019_dev"}:
        raise ValueError("validation_protocol must be 'asvspoof2021_fast' or 'asvspoof2019_dev'")

    config = load_experiment_config(config_path)
    configured_validation_protocol = _validation_protocol_from_config(config)
    if configured_validation_protocol != validation_protocol:
        raise ValueError(
            "validation_protocol does not match config protocol: "
            f"config={configured_validation_protocol}, cli={validation_protocol}"
        )

    loop = TrainLoopSettings(
        epochs=epochs,
        device=device,
        top_k_checkpoints=top_k_checkpoints,
        max_grad_norm=max_grad_norm,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    loop.validate()

    return LegacyASVspoofTrainingPlan(
        config=config,
        data=ASVspoofDataSettings(
            database_path=Path(database_path),
            protocols_path=Path(protocols_path),
            track=track,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            sample_rate=output_sample_rate,
            cut=cut,
            rawboost_algo=rawboost_algo,
            rawboost_args=dict(rawboost_args or {}),
            validation_protocol=cast(ValidationProtocol, validation_protocol),
        ),
        frontend=LegacyFrontendSettings(
            name=frontend_name,
            checkpoint=frontend_checkpoint,
            model_path=frontend_model_path,
            model_name=frontend_model_name,
            freeze=_freeze_frontend(config, freeze_frontend),
            feature_type=feature_type,
            finetune_config=_finetune_config(config, frontend_name),
        ),
        model=LegacyModelSettings(),
        loop=loop,
        frontend_prefix=frontend_prefix,
    )


def run_legacy_asvspoof_training(
    plan: LegacyASVspoofTrainingPlan,
    *,
    output_dir: str | Path,
    experiment_run: PreparedExperimentRun | None = None,
) -> TrainingRunResult:
    """Run one controlled legacy ASVspoof training job from an explicit plan."""

    try:
        loaders = build_asvspoof_loaders(plan.data)
        frontend = build_legacy_frontend(plan.frontend)
        model = build_legacy_model(frontend, plan.model)

        if plan.config.optimizer.name.lower() == "adamw":
            encoder, backend = split_frontend_backend_params(
                model, frontend_prefix=plan.frontend_prefix
            )
            optimizer = build_optimizer(
                config=plan.config, model=model, encoder_params=encoder, backend_params=backend
            )
        else:
            optimizer = build_optimizer(plan.config, model)

        result = train_with_components(
            config=plan.config,
            components=TrainingComponents(
                model=model,
                train_loader=loaders.train_loader,
                val_loader=loaders.val_loader,
                optimizer=optimizer,
            ),
            output_dir=output_dir,
            settings=plan.loop,
        )
        if experiment_run is not None:
            complete_experiment_run(
                experiment_run,
                metrics={
                    "best_val_loss": result.best_val_loss,
                    "final_train_loss": result.final_train_loss,
                    "epochs_completed": result.epochs_completed,
                    "checkpoint_metric": plan.loop.checkpoint_metric,
                },
                artifacts={
                    "best_checkpoint": str(result.best_checkpoint),
                    "training_manifest": str(result.manifest_path),
                },
            )
        return result
    except BaseException as exc:
        if experiment_run is not None:
            fail_experiment_run(
                experiment_run,
                exc,
                metrics={"checkpoint_metric": plan.loop.checkpoint_metric},
            )
        raise


def _freeze_frontend(config: ExperimentConfig, requested: bool | None) -> bool | None:
    if requested is not None:
        return requested
    if config.strategy == "frozen":
        return True
    if config.strategy in {"adapter", "full", "full_ft", "full_external"}:
        return False
    return None


def _finetune_config(config: ExperimentConfig, frontend_name: str) -> dict[str, Any] | None:
    if config.strategy == "adapter" and frontend_name == "wav2vec2":
        return {
            "strategy": "adapter",
            "adapter": {"dim": 64, "dropout": 0.1, "layers": "last_n", "n_layers": 8},
        }
    return None


def _validation_protocol_from_config(config: ExperimentConfig) -> str:
    protocol_text = " ".join(
        [config.protocol.validation_set, config.protocol.checkpoint_selection_set]
    ).lower()
    if "asvspoof2021" in protocol_text and "fast" in protocol_text:
        return "asvspoof2021_fast"
    if "asvspoof2019" in protocol_text and "dev" in protocol_text:
        return "asvspoof2019_dev"
    raise ValueError(
        "config protocol is not launchable by legacy ASVspoof CLI: "
        f"validation_set={config.protocol.validation_set!r}, "
        f"checkpoint_selection_set={config.protocol.checkpoint_selection_set!r}"
    )


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value
