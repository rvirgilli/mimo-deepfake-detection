"""Run-index generation for versioned and historical experiment evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mimodf.experiments.manifest import RunManifest
from mimodf.provenance import MainTableProvenance, load_main_table_provenance

RUN_INDEX_RECORD_SCHEMA = "run-index-record/v1"


@dataclass(frozen=True)
class RunIndexRecord:
    record_schema: str
    source_type: str
    experiment_id: str
    run_id: str
    seed: int | str
    status: str
    intent: str
    reproducibility_tier: int
    component_ids: dict[str, str] = field(default_factory=dict)
    protocol_ids: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_index(
    roots: Iterable[str | Path] = (),
    *,
    provenance_path: str | Path | None = None,
) -> list[RunIndexRecord]:
    records: list[RunIndexRecord] = []
    for root in roots:
        records.extend(index_run_layout_root(root))
    if provenance_path is not None:
        records.extend(index_main_table_provenance(provenance_path))
    return sorted(records, key=lambda item: (item.experiment_id, str(item.seed), item.run_id))


def index_run_layout_root(root: str | Path) -> list[RunIndexRecord]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    records: list[RunIndexRecord] = []
    for manifest_path in sorted(root_path.rglob("manifest.json")):
        try:
            manifest = RunManifest.load(manifest_path)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        records.append(_record_from_run_manifest(manifest, manifest_path))
    return records


def index_main_table_provenance(path: str | Path) -> list[RunIndexRecord]:
    return records_from_main_table_provenance(load_main_table_provenance(path))


def records_from_main_table_provenance(
    provenance: MainTableProvenance,
) -> list[RunIndexRecord]:
    records: list[RunIndexRecord] = []
    for row in provenance.rows:
        for seed in row.seeds:
            artifact_paths = [
                value for value in seed.artifacts.values() if value and value != "missing"
            ]
            metrics = {
                key: value
                for key, value in {
                    "la_eer": seed.metrics.la_eer,
                    "df_eer": seed.metrics.df_eer,
                    "la_tdcf": seed.metrics.la_tdcf,
                }.items()
                if value is not None
            }
            warnings = [row.notes]
            if row.tdcf_note:
                warnings.append(row.tdcf_note)
            records.append(
                RunIndexRecord(
                    record_schema=RUN_INDEX_RECORD_SCHEMA,
                    source_type="historical_provenance",
                    experiment_id=row.id,
                    run_id=f"{row.id}/seed_{seed.id}",
                    seed=_seed_value(seed.id),
                    status=seed.status,
                    intent="exploratory" if row.exploratory else "historical",
                    reproducibility_tier=_historical_tier(seed.artifacts),
                    metrics=metrics,
                    artifact_paths=artifact_paths,
                    warnings=warnings,
                )
            )
    return records


def load_run_index_jsonl(path: str | Path) -> list[RunIndexRecord]:
    records: list[RunIndexRecord] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        try:
            records.append(run_index_record_from_dict(data))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid run-index record at line {line_number}: {exc}") from exc
    return records


def run_index_record_from_dict(data: dict[str, Any]) -> RunIndexRecord:
    if data.get("record_schema") != RUN_INDEX_RECORD_SCHEMA:
        raise ValueError("record_schema must be run-index-record/v1")
    return RunIndexRecord(
        record_schema=str(data["record_schema"]),
        source_type=str(data["source_type"]),
        experiment_id=str(data["experiment_id"]),
        run_id=str(data["run_id"]),
        seed=data["seed"],
        status=str(data["status"]),
        intent=str(data["intent"]),
        reproducibility_tier=int(data["reproducibility_tier"]),
        component_ids={
            str(key): str(value) for key, value in dict(data.get("component_ids") or {}).items()
        },
        protocol_ids={
            str(key): str(value) for key, value in dict(data.get("protocol_ids") or {}).items()
        },
        metrics=dict(data.get("metrics") or {}),
        artifact_paths=[str(path) for path in list(data.get("artifact_paths") or [])],
        warnings=[str(warning) for warning in list(data.get("warnings") or [])],
    )


def render_run_index_jsonl(records: Iterable[RunIndexRecord]) -> str:
    return "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records)


def render_run_index_markdown(records: Iterable[RunIndexRecord]) -> str:
    lines = [
        "| Source | Experiment | Seed | Status | Intent | Tier | Metrics | Warnings |",
        "|---|---|---:|---|---|---:|---|---|",
    ]
    for record in records:
        metric_text = ", ".join(f"{key}={value}" for key, value in sorted(record.metrics.items()))
        warning_text = "; ".join(record.warnings)
        lines.append(
            f"| {record.source_type} | {record.experiment_id} | {record.seed} | "
            f"{record.status} | {record.intent} | {record.reproducibility_tier} | "
            f"{metric_text or '---'} | {warning_text or '---'} |"
        )
    return "\n".join(lines) + "\n"


def _record_from_run_manifest(manifest: RunManifest, manifest_path: Path) -> RunIndexRecord:
    run_dir = manifest_path.parent
    artifact_paths = [_artifact_path(artifact) for artifact in manifest.artifacts]
    artifact_paths = [path for path in artifact_paths if path]
    return RunIndexRecord(
        record_schema=RUN_INDEX_RECORD_SCHEMA,
        source_type="new_run",
        experiment_id=manifest.experiment_id,
        run_id=manifest.run_id,
        seed=manifest.seed,
        status=manifest.status,
        intent=manifest.intent,
        reproducibility_tier=_new_run_tier(manifest, run_dir),
        component_ids={
            key: value for key, value in manifest.model.items() if isinstance(value, str)
        },
        protocol_ids={
            **{key: value for key, value in manifest.protocol.items() if isinstance(value, str)},
            "evaluation_batch_size": str(manifest.evaluation["batch_size"]),
        },
        metrics=dict(manifest.metrics),
        artifact_paths=artifact_paths,
        warnings=list(manifest.warnings + manifest.failures),
    )


def _artifact_path(artifact: Any) -> str:
    if isinstance(artifact, dict):
        value = artifact.get("path")
        return "" if value is None else str(value)
    return ""


def _new_run_tier(manifest: RunManifest, run_dir: Path) -> int:
    has_resolved_spec = (run_dir / "resolved_spec.yaml").is_file()
    has_artifacts = bool(manifest.artifacts)
    if has_resolved_spec and has_artifacts and manifest.status == "completed":
        return 3
    if has_resolved_spec and manifest.status in {"planned", "running", "failed", "interrupted"}:
        return 1
    return 0


def _historical_tier(artifacts: dict[str, str]) -> int:
    values = {key: value for key, value in artifacts.items() if value and value != "missing"}
    has_score = any(key.endswith("score") for key in values)
    has_checkpoint = "checkpoint" in values
    has_config = "config" in values
    if has_score and has_checkpoint and has_config:
        return 2
    if has_score:
        return 1
    return 0


def _seed_value(seed: str) -> int | str:
    try:
        return int(seed)
    except ValueError:
        return seed
