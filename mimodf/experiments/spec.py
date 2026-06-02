"""Versioned experiment specification parsing and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mimodf.components.registry import get_component

EXPERIMENT_SPEC_SCHEMA = "experiment-spec/v1"
INTENTS = {"exploratory", "confirmatory", "reproduction", "diagnostic"}
CHECKPOINT_SELECTIONS = {"dev_loss", "dev_eer", "fixed_epoch", "historical"}
LEAKAGE_POLICIES = {"no_eval_selection", "exploratory_eval_selection"}


class SpecValidationError(ValueError):
    """Raised when an experiment spec violates the public contract."""


@dataclass(frozen=True)
class ExperimentSpec:
    data: dict[str, Any]
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        spec_path = Path(path)
        data = yaml.safe_load(spec_path.read_text())
        if not isinstance(data, dict):
            raise SpecValidationError("experiment spec must be a YAML mapping")
        return cls(data=data, path=spec_path)

    def validate(self) -> None:
        validate_experiment_spec(self.data)

    def spec_hash(self) -> str:
        self.validate()
        payload = json.dumps(self.data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolved(self, *, resolved_at: str | None = None) -> dict[str, Any]:
        self.validate()
        spec_hash = self.spec_hash()
        frontend_id = _require_str(self.data, "model.frontend")
        backend_id = _require_str(self.data, "model.backend")
        adaptation_id = _require_str(self.data, "model.adaptation")
        optimizer_id = _require_str(self.data, "training.optimizer")
        frontend = get_component(frontend_id)
        return {
            **self.data,
            "resolved_at": resolved_at or datetime.now(UTC).isoformat(),
            "spec_hash": spec_hash,
            "component_versions": {
                "frontend": frontend_id,
                "backend": backend_id,
                "adaptation": adaptation_id,
                "optimizer": optimizer_id,
            },
            "frontend_facts": {
                **frontend.metadata,
                "known_caveats": list(frontend.caveats),
            },
        }

    def write_resolved(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.resolved(), sort_keys=False))
        return out


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    spec = ExperimentSpec.load(path)
    spec.validate()
    return spec


def validate_experiment_spec(data: dict[str, Any]) -> None:
    if data.get("schema_version") != EXPERIMENT_SPEC_SCHEMA:
        raise SpecValidationError("schema_version must be experiment-spec/v1")

    _require_str(data, "experiment_id")
    intent = _require_str(data, "intent")
    if intent not in INTENTS:
        raise SpecValidationError(f"intent must be one of {sorted(INTENTS)}")
    _require_str(data, "hypothesis")
    _require_str(data, "owner")
    _require_str(data, "created")

    protocol = _require_mapping(data, "protocol")
    _validate_registered_id(protocol, "train_dataset", expected_kind="dataset")
    _validate_registered_id(protocol, "validation_dataset", expected_kind="dataset")
    checkpoint_selection = _require_str(
        protocol, "checkpoint_selection", display="protocol.checkpoint_selection"
    )
    if checkpoint_selection not in CHECKPOINT_SELECTIONS:
        raise SpecValidationError(
            f"protocol.checkpoint_selection must be one of {sorted(CHECKPOINT_SELECTIONS)}"
        )
    eval_datasets = _require_list(protocol, "eval_datasets", display="protocol.eval_datasets")
    for index, dataset_id in enumerate(eval_datasets):
        _validate_component_value(dataset_id, f"protocol.eval_datasets[{index}]", "dataset")
    scorers = _require_mapping(protocol, "scorers", display="protocol.scorers")
    if not scorers:
        raise SpecValidationError("protocol.scorers must not be empty")
    for name, scorer_id in scorers.items():
        _validate_component_value(scorer_id, f"protocol.scorers.{name}", "scorer")
    leakage_policy = _require_str(protocol, "leakage_policy", display="protocol.leakage_policy")
    if leakage_policy not in LEAKAGE_POLICIES:
        raise SpecValidationError(
            f"protocol.leakage_policy must be one of {sorted(LEAKAGE_POLICIES)}"
        )

    seeds = _require_list(data, "seeds")
    if not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise SpecValidationError("seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise SpecValidationError("seeds must not contain duplicates")

    model = _require_mapping(data, "model")
    _validate_registered_id(model, "frontend", expected_kind="frontend")
    _validate_registered_id(model, "backend", expected_kind="backend")
    projection = _require_mapping(model, "projection", display="model.projection")
    if _require_str(projection, "type", display="model.projection.type") != "linear":
        raise SpecValidationError("model.projection.type currently must be linear")
    _require_positive_int(projection, "output_dim", display="model.projection.output_dim")
    _validate_registered_id(model, "adaptation", expected_kind="adaptation")

    training = _require_mapping(data, "training")
    _require_positive_int(training, "max_epochs", display="training.max_epochs")
    _require_positive_int(training, "batch_size", display="training.batch_size")
    _validate_registered_id(training, "optimizer", expected_kind="optimizer")
    _require_number(training, "learning_rate", display="training.learning_rate", positive=True)

    evaluation = _require_mapping(data, "evaluation")
    _require_positive_int(evaluation, "batch_size", display="evaluation.batch_size")
    for flag in ("write_scores", "official_scoring"):
        if not isinstance(evaluation.get(flag), bool):
            raise SpecValidationError(f"evaluation.{flag} must be boolean")

    _require_mapping(data, "comparability")
    _require_mapping(data, "artifacts")


def _require_mapping(
    data: dict[str, Any], key: str, *, display: str | None = None
) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise SpecValidationError(f"{display or key} must be a mapping")
    return value


def _require_list(data: dict[str, Any], key: str, *, display: str | None = None) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise SpecValidationError(f"{display or key} must be a list")
    return value


def _require_str(data: dict[str, Any], dotted_key: str, *, display: str | None = None) -> str:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise SpecValidationError(f"{display or dotted_key} is required")
        current = current[part]
    if not isinstance(current, str) or not current.strip():
        raise SpecValidationError(f"{display or dotted_key} must be a non-empty string")
    return current


def _require_number(
    data: dict[str, Any],
    key: str,
    *,
    display: str,
    positive: bool = False,
) -> float:
    value = data.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise SpecValidationError(f"{display} must be numeric")
    if positive and value <= 0:
        raise SpecValidationError(f"{display} must be positive")
    return float(value)


def _require_positive_int(data: dict[str, Any], key: str, *, display: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SpecValidationError(f"{display} must be a positive integer")
    return value


def _validate_registered_id(data: dict[str, Any], key: str, *, expected_kind: str) -> None:
    value = _require_str(data, key)
    _validate_component_value(value, key, expected_kind)


def _validate_component_value(value: Any, display: str, expected_kind: str) -> None:
    if not isinstance(value, str):
        raise SpecValidationError(f"{display} must be a component id string")
    try:
        component = get_component(value)
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc
    if component.kind != expected_kind:
        raise SpecValidationError(
            f"{display} must reference a {expected_kind} component, got {component.kind}: {value}"
        )
