"""Integration seam between real model/data factories and the training loop."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimodf.config import ExperimentConfig
from mimodf.training.loop import TrainingRunResult, TrainLoopSettings, train_one_run


@dataclass(frozen=True)
class TrainingComponents:
    """Concrete pieces needed by `train_one_run`.

    Real ASVspoof/frontends code should build this object. The generic training
    loop should not know how MiMo, wav2vec2, or datasets are constructed.
    """

    model: Any
    train_loader: Any
    val_loader: Any
    optimizer: Any


def train_with_components(
    *,
    config: ExperimentConfig,
    components: TrainingComponents,
    output_dir: str | Path,
    settings: TrainLoopSettings,
) -> TrainingRunResult:
    return train_one_run(
        config=config,
        model=components.model,
        train_loader=components.train_loader,
        val_loader=components.val_loader,
        optimizer=components.optimizer,
        output_dir=output_dir,
        settings=settings,
    )


def build_optimizer(
    config: ExperimentConfig,
    model: Any,
    *,
    encoder_params: Iterable[Any] | None = None,
    backend_params: Iterable[Any] | None = None,
) -> Any:
    """Build optimizer from typed config.

    Mirrors the audited legacy rule:
    - `encoder_lr is None` -> Adam over all trainable params;
    - `encoder_lr is not None` -> AdamW with explicit encoder/backend groups.
    """

    torch = _import_torch()
    opt = config.optimizer
    name = opt.name.lower()

    if name == "adam":
        if opt.encoder_lr is not None:
            raise ValueError("Adam config must not set encoder_lr")
        return torch.optim.Adam(
            _non_empty_params(trainable_parameters(model), "model"),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )

    if name == "adamw":
        encoder = _non_empty_params(encoder_params, "encoder_params")
        backend = _non_empty_params(backend_params, "backend_params")
        return torch.optim.AdamW(
            [
                {"params": encoder, "lr": opt.encoder_lr, "name": "encoder"},
                {"params": backend, "lr": opt.lr, "name": "backend"},
            ],
            weight_decay=opt.weight_decay,
        )

    raise ValueError(f"unsupported optimizer: {opt.name}")


def trainable_parameters(model: Any) -> list[Any]:
    return [param for param in model.parameters() if getattr(param, "requires_grad", False)]


def split_frontend_backend_params(
    model: Any,
    *,
    frontend_prefix: str = "frontend",
) -> tuple[list[Any], list[Any]]:
    """Split trainable named parameters into frontend/backend groups.

    This is only a seam for legacy model structure. Real model factories may
    pass explicit groups instead.
    """

    prefix = f"{frontend_prefix}."
    encoder: list[Any] = []
    backend: list[Any] = []
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        if name.startswith(prefix):
            encoder.append(param)
        else:
            backend.append(param)
    return encoder, backend


def _non_empty_params(params: Iterable[Any] | None, label: str) -> list[Any]:
    materialized = list(params or [])
    if not materialized:
        raise ValueError(f"{label} must contain at least one trainable parameter")
    return materialized


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Torch is required for training components") from exc
    return torch
