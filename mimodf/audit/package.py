"""Package generated audit outputs for review/release."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from mimodf.audit.artifact_gaps import (
    DEFAULT_ARTIFACT_GAP_POLICY,
    load_artifact_gap_policy,
    render_artifact_gap_policy_json,
    render_artifact_gap_policy_markdown,
)
from mimodf.audit.artifacts import (
    ArtifactCheck,
    check_artifacts_from_file,
    render_artifact_checks_json,
    render_artifact_checks_markdown,
)
from mimodf.audit.dependencies import (
    DEFAULT_DEPENDENCY_SPEC,
    DEFAULT_LOCAL_DEPENDENCY_SPEC,
    audit_external_dependencies,
    render_dependency_checks_json,
    render_dependency_checks_markdown,
)
from mimodf.scoring.tdcf_summary import (
    load_tdcf_summary,
    render_tdcf_summary_json,
    render_tdcf_summary_markdown,
)
from mimodf.tables.main_table import render_main_table_from_file


@dataclass(frozen=True)
class AuditPackage:
    output_dir: str
    files: dict[str, str]
    artifact_status_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_audit_package(
    provenance_path: str | Path,
    output_dir: str | Path,
    *,
    root: str | Path = ".",
    include_provenance_copy: bool = True,
    tdcf_values_path: str | Path | None = "docs/current/official_tdcf_values.yaml",
    dependency_spec_path: str | Path | None = DEFAULT_DEPENDENCY_SPEC,
    dependency_local_spec_path: str | Path | None = DEFAULT_LOCAL_DEPENDENCY_SPEC,
    artifact_gap_policy_path: str | Path | None = DEFAULT_ARTIFACT_GAP_POLICY,
) -> AuditPackage:
    """Write generated audit outputs to a directory.

    Outputs are deterministic text/JSON artifacts derived from structured
    provenance. They are safe to regenerate and should not become a second source
    of truth.
    """

    provenance = Path(provenance_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checks = check_artifacts_from_file(provenance, root=root)
    files = {
        "main_table_markdown": "main_table.md",
        "artifact_checks_markdown": "artifact_checks.md",
        "artifact_checks_json": "artifact_checks.json",
        "artifact_gaps_markdown": "artifact_gaps.md",
        "artifact_gaps_json": "artifact_gaps.json",
        "dependency_checks_markdown": "dependency_checks.md",
        "dependency_checks_json": "dependency_checks.json",
        "summary_json": "summary.json",
    }
    tdcf_path = Path(tdcf_values_path) if tdcf_values_path is not None else None
    dependency_spec = Path(dependency_spec_path) if dependency_spec_path is not None else None
    dependency_local_spec = (
        Path(dependency_local_spec_path) if dependency_local_spec_path is not None else None
    )
    artifact_gap_policy = (
        Path(artifact_gap_policy_path) if artifact_gap_policy_path is not None else None
    )
    if tdcf_path is not None and tdcf_path.is_file():
        files["tdcf_summary_markdown"] = "tdcf_summary.md"
        files["tdcf_summary_json"] = "tdcf_summary.json"
        files["tdcf_values_yaml"] = "official_tdcf_values.yaml"
    if artifact_gap_policy is not None and artifact_gap_policy.is_file():
        files["artifact_gap_policy_yaml"] = "artifact_gap_decisions.yaml"
    if dependency_spec is not None and dependency_spec.is_file():
        files["dependency_spec_yaml"] = "external_dependencies.yaml"
    if dependency_local_spec is not None and dependency_local_spec.is_file():
        files["dependency_local_spec_yaml"] = "external_dependencies.local.yaml"
    if include_provenance_copy:
        files["provenance_yaml"] = "main_table_provenance.yaml"

    dependency_checks = audit_external_dependencies(
        root=root,
        spec_path=dependency_spec or DEFAULT_DEPENDENCY_SPEC,
        local_spec_path=dependency_local_spec,
    )

    (out / files["main_table_markdown"]).write_text(render_main_table_from_file(provenance))
    (out / files["artifact_checks_markdown"]).write_text(render_artifact_checks_markdown(checks))
    (out / files["artifact_checks_json"]).write_text(render_artifact_checks_json(checks))
    if artifact_gap_policy is not None and artifact_gap_policy.is_file():
        gap_policy = load_artifact_gap_policy(artifact_gap_policy)
        (out / files["artifact_gaps_markdown"]).write_text(
            render_artifact_gap_policy_markdown(gap_policy)
        )
        (out / files["artifact_gaps_json"]).write_text(render_artifact_gap_policy_json(gap_policy))
    (out / files["dependency_checks_markdown"]).write_text(
        render_dependency_checks_markdown(dependency_checks)
    )
    (out / files["dependency_checks_json"]).write_text(
        render_dependency_checks_json(dependency_checks)
    )

    if tdcf_path is not None and tdcf_path.is_file():
        tdcf_summary = load_tdcf_summary(tdcf_path)
        (out / files["tdcf_summary_markdown"]).write_text(
            render_tdcf_summary_markdown(tdcf_summary)
        )
        (out / files["tdcf_summary_json"]).write_text(render_tdcf_summary_json(tdcf_summary))
        shutil.copyfile(tdcf_path, out / files["tdcf_values_yaml"])
    if artifact_gap_policy is not None and artifact_gap_policy.is_file():
        shutil.copyfile(artifact_gap_policy, out / files["artifact_gap_policy_yaml"])
    if dependency_spec is not None and dependency_spec.is_file():
        shutil.copyfile(dependency_spec, out / files["dependency_spec_yaml"])
    if dependency_local_spec is not None and dependency_local_spec.is_file():
        shutil.copyfile(dependency_local_spec, out / files["dependency_local_spec_yaml"])

    package = AuditPackage(
        output_dir=str(out),
        files={key: str(out / filename) for key, filename in files.items()},
        artifact_status_counts=_status_counts(checks),
    )
    (out / files["summary_json"]).write_text(json.dumps(package.to_dict(), indent=2) + "\n")

    if include_provenance_copy:
        shutil.copyfile(provenance, out / files["provenance_yaml"])

    return package


def _status_counts(checks: list[ArtifactCheck]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return dict(sorted(counts.items()))
