"""Helpers for creating and updating versioned experiment run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mimodf.experiments.layout import RunLayout, build_run_layout
from mimodf.experiments.manifest import RunManifest
from mimodf.experiments.spec import load_experiment_spec


@dataclass(frozen=True)
class PreparedExperimentRun:
    layout: RunLayout
    resolved_spec: dict[str, Any]
    manifest: RunManifest


def prepare_experiment_run(
    *,
    spec_path: str | Path,
    seed: int,
    root: str | Path | None = None,
    overwrite: bool = False,
    status: str = "planned",
) -> PreparedExperimentRun:
    """Create a run-layout directory from a versioned spec and seed.

    Historical artifacts are never touched. This helper writes only the new
    run-layout files under the selected output root.
    """

    spec = load_experiment_spec(spec_path)
    if seed not in spec.data["seeds"]:
        raise ValueError(f"seed {seed} is not declared by experiment spec")
    resolved = spec.resolved()
    output_root = Path(root or resolved["artifacts"]["output_root"])
    layout = build_run_layout(
        root=output_root,
        experiment_id=resolved["experiment_id"],
        spec_hash=resolved["spec_hash"],
        seed=seed,
    ).create(overwrite=overwrite)
    layout.resolved_spec_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    manifest = RunManifest.from_resolved_spec(resolved, seed=seed, status=status)
    manifest.save(layout.manifest_path)
    return PreparedExperimentRun(layout=layout, resolved_spec=resolved, manifest=manifest)


def complete_experiment_run(
    prepared: PreparedExperimentRun,
    *,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
) -> None:
    manifest = prepared.manifest
    manifest.status = "completed"
    manifest.metrics = dict(metrics or {})
    manifest.artifacts = [
        {"name": name, "path": path} for name, path in sorted(dict(artifacts or {}).items())
    ]
    manifest.save(prepared.layout.manifest_path)


def fail_experiment_run(
    prepared: PreparedExperimentRun,
    error: BaseException | str,
    *,
    metrics: dict[str, Any] | None = None,
) -> None:
    manifest = prepared.manifest
    manifest.status = "failed"
    manifest.metrics = dict(metrics or {})
    manifest.failures.append(str(error))
    manifest.save(prepared.layout.manifest_path)
