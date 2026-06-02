"""Factories for legacy frontend/model construction behind the new seam."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LegacyFrontendSettings:
    name: str
    checkpoint: str | None = None
    model_path: str | None = None
    model_name: str | None = None
    freeze: bool | None = None
    use_bfloat16: bool = True
    upsample_to_50hz: bool = False
    upsample_mode: str = "linear"
    native_50hz: bool = False
    feature_type: str = "continuous"
    feature_config: dict[str, Any] | None = None
    finetune_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class LegacyModelSettings:
    filts_0: int = 128
    encoder_scale: float = 1.0
    gat_dims: list[int] = field(default_factory=lambda: [64, 32])
    pool_ratios: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    temperatures: list[float] = field(default_factory=lambda: [2.0, 2.0, 100.0, 100.0])
    dropout: float = 0.5
    dropout_way: float = 0.2
    projection_type: str = "linear"
    projection_hidden_dims: list[int] | None = field(default_factory=lambda: [512, 256])
    projection_activation: str = "gelu"
    projection_dropout: float = 0.1
    projection_use_batchnorm: bool = True


def build_legacy_frontend(settings: LegacyFrontendSettings) -> Any:
    """Build a legacy frontend without importing legacy code at module import time."""

    from src.frontends import get_frontend

    return get_frontend(
        settings.name,
        checkpoint=settings.checkpoint,
        model_path=settings.model_path,
        model_name=settings.model_name,
        freeze=settings.freeze,
        use_bfloat16=settings.use_bfloat16,
        upsample_to_50hz=settings.upsample_to_50hz,
        upsample_mode=settings.upsample_mode,
        native_50hz=settings.native_50hz,
        feature_type=settings.feature_type,
        feature_config=settings.feature_config,
        finetune_config=settings.finetune_config,
    )


def build_legacy_model(frontend: Any, settings: LegacyModelSettings) -> Any:
    """Build the legacy AASIST-style model with explicit settings."""

    from src.model import Model

    return Model(
        frontend=frontend,
        filts_0=settings.filts_0,
        encoder_scale=settings.encoder_scale,
        gat_dims=settings.gat_dims,
        pool_ratios=settings.pool_ratios,
        temperatures=settings.temperatures,
        dropout=settings.dropout,
        dropout_way=settings.dropout_way,
        projection_type=settings.projection_type,
        projection_hidden_dims=settings.projection_hidden_dims,
        projection_activation=settings.projection_activation,
        projection_dropout=settings.projection_dropout,
        projection_use_batchnorm=settings.projection_use_batchnorm,
    )
