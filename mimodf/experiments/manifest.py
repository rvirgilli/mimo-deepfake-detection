"""Versioned run manifest for experiment-system runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA = "run-manifest/v1"
RUN_STATUSES = {"planned", "running", "completed", "failed", "interrupted", "superseded", "retired"}
RUN_INTENTS = {"exploratory", "confirmatory", "reproduction", "diagnostic", "historical"}


@dataclass
class RunManifest:
    run_id: str
    experiment_id: str
    spec_hash: str
    seed: int
    intent: str
    status: str
    started_at: str | None = None
    ended_at: str | None = None
    git: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    schema_version: str = RUN_MANIFEST_SCHEMA

    def validate(self) -> None:
        if self.schema_version != RUN_MANIFEST_SCHEMA:
            raise ValueError("schema_version must be run-manifest/v1")
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.spec_hash.startswith("sha256:"):
            raise ValueError("spec_hash must start with sha256:")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.intent not in RUN_INTENTS:
            raise ValueError(f"intent must be one of {sorted(RUN_INTENTS)}")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"status must be one of {sorted(RUN_STATUSES)}")
        if "checkpoint_selection" not in self.protocol:
            raise ValueError("protocol.checkpoint_selection is required")
        if "leakage_policy" not in self.protocol:
            raise ValueError("protocol.leakage_policy is required")
        if "frontend" not in self.model or "backend" not in self.model:
            raise ValueError("model.frontend and model.backend are required")
        if "batch_size" not in self.evaluation:
            raise ValueError("evaluation.batch_size is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return out

    @classmethod
    def load(cls, path: str | Path) -> RunManifest:
        data = json.loads(Path(path).read_text())
        manifest = cls(**data)
        manifest.validate()
        return manifest

    @classmethod
    def from_resolved_spec(
        cls,
        resolved_spec: dict[str, Any],
        *,
        seed: int,
        status: str = "planned",
        run_id: str | None = None,
    ) -> RunManifest:
        experiment_id = str(resolved_spec["experiment_id"])
        spec_hash = str(resolved_spec["spec_hash"])
        return cls(
            run_id=run_id or f"{experiment_id}/{_hash_dir(spec_hash)}/seed_{seed}",
            experiment_id=experiment_id,
            spec_hash=spec_hash,
            seed=seed,
            intent=str(resolved_spec["intent"]),
            status=status,
            protocol={
                "checkpoint_selection": resolved_spec["protocol"]["checkpoint_selection"],
                "leakage_policy": resolved_spec["protocol"]["leakage_policy"],
            },
            model={
                "frontend": resolved_spec["model"]["frontend"],
                "backend": resolved_spec["model"]["backend"],
                "adaptation": resolved_spec["model"]["adaptation"],
            },
            evaluation={"batch_size": resolved_spec["evaluation"]["batch_size"]},
            warnings=list(resolved_spec.get("frontend_facts", {}).get("known_caveats", [])),
        )


def _hash_dir(spec_hash: str) -> str:
    return spec_hash.removeprefix("sha256:")
