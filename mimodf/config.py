"""Typed config schema for publishable experiments.

This is intentionally narrower than the legacy Hydra configs. It captures the
protocol facts that affected the paper assessment, especially validation set,
optimizer, and scorer identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a publish config is missing required protocol facts."""


@dataclass(frozen=True)
class ProtocolConfig:
    train_set: str
    validation_set: str
    checkpoint_selection_set: str
    eval_set: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProtocolConfig:
        required = ("train_set", "validation_set", "checkpoint_selection_set", "eval_set")
        _require(data, required, "protocol")
        return cls(
            train_set=str(data["train_set"]),
            validation_set=str(data["validation_set"]),
            checkpoint_selection_set=str(data["checkpoint_selection_set"]),
            eval_set=str(data["eval_set"]),
        )

    def validate(self) -> None:
        for field, value in vars(self).items():
            if not value or value == "implicit":
                raise ConfigError(f"protocol.{field} must be explicit")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    encoder_lr: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizerConfig:
        _require(data, ("name", "lr", "weight_decay"), "optimizer")
        return cls(
            name=str(data["name"]),
            lr=float(data["lr"]),
            weight_decay=float(data["weight_decay"]),
            encoder_lr=_optional_float(data.get("encoder_lr")),
        )

    def validate(self) -> None:
        name = self.name.lower()
        if name not in {"adam", "adamw"}:
            raise ConfigError(f"unsupported optimizer: {self.name}")
        if self.lr <= 0:
            raise ConfigError("optimizer.lr must be positive")
        if self.weight_decay < 0:
            raise ConfigError("optimizer.weight_decay must be non-negative")
        if name == "adamw" and self.encoder_lr is None:
            raise ConfigError(
                "repo-native AdamW requires explicit optimizer.encoder_lr; "
                "encoder_lr: null follows the legacy Adam path"
            )


@dataclass(frozen=True)
class ScorerConfig:
    la: str
    df: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScorerConfig:
        _require(data, ("la",), "scorer")
        return cls(la=str(data["la"]), df=None if data.get("df") is None else str(data["df"]))

    def validate(self) -> None:
        if self.la != "official_asvspoof2021_la":
            raise ConfigError("scorer.la must be official_asvspoof2021_la")


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    strategy: str
    seed: int
    protocol: ProtocolConfig
    optimizer: OptimizerConfig
    scorer: ScorerConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        required = ("model", "strategy", "seed", "protocol", "optimizer", "scorer")
        _require(data, required, "config")
        cfg = cls(
            model=str(data["model"]),
            strategy=str(data["strategy"]),
            seed=int(data["seed"]),
            protocol=ProtocolConfig.from_dict(_mapping(data["protocol"], "protocol")),
            optimizer=OptimizerConfig.from_dict(_mapping(data["optimizer"], "optimizer")),
            scorer=ScorerConfig.from_dict(_mapping(data["scorer"], "scorer")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.model:
            raise ConfigError("model must be non-empty")
        if not self.strategy:
            raise ConfigError("strategy must be non-empty")
        self.protocol.validate()
        self.optimizer.validate()
        self.scorer.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig.from_dict(_mapping(data, "config"))


def _require(data: dict[str, Any], keys: tuple[str, ...], prefix: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ConfigError(f"{prefix} missing required keys: {', '.join(missing)}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
