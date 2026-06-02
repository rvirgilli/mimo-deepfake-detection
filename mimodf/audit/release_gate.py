"""One-command release readiness summary.

The gate is deliberately conservative. It does not make the repo release-ready;
it tells the truth about known blockers in a compact, machine-readable shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mimodf.audit.artifact_gaps import (
    DEFAULT_ARTIFACT_GAP_POLICY,
    ArtifactGapPolicy,
    load_artifact_gap_policy,
)
from mimodf.audit.artifacts import ArtifactCheck, check_artifacts_from_file
from mimodf.audit.dependencies import (
    DEFAULT_DEPENDENCY_SPEC,
    DEFAULT_LOCAL_DEPENDENCY_SPEC,
    ExternalDependencyCheck,
    audit_external_dependencies,
)


@dataclass(frozen=True)
class ReleaseGateIssue:
    code: str
    severity: str
    subject: str
    detail: str


@dataclass(frozen=True)
class ReleaseGateReport:
    verdict: str
    provenance_path: str
    dependency_spec_path: str
    dependency_local_spec_path: str | None
    dependency_local_spec_present: bool
    artifact_gap_policy_path: str | None
    artifact_gap_policy_present: bool
    allow_known_artifact_gaps: bool
    hash_files: bool
    artifact_status_counts: dict[str, int]
    dependency_counts: dict[str, int]
    blockers: tuple[ReleaseGateIssue, ...]
    warnings: tuple[ReleaseGateIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return not self.blockers


def build_release_gate_report(
    provenance_path: str | Path,
    *,
    root: str | Path = ".",
    dependency_spec_path: str | Path = DEFAULT_DEPENDENCY_SPEC,
    dependency_local_spec_path: str | Path | None = DEFAULT_LOCAL_DEPENDENCY_SPEC,
    artifact_gap_policy_path: str | Path | None = DEFAULT_ARTIFACT_GAP_POLICY,
    allow_known_artifact_gaps: bool = False,
    hash_files: bool = False,
) -> ReleaseGateReport:
    artifact_checks = check_artifacts_from_file(provenance_path, root=root)
    dependency_checks = audit_external_dependencies(
        root=root,
        spec_path=dependency_spec_path,
        local_spec_path=dependency_local_spec_path,
        hash_files=hash_files,
    )
    artifact_gap_policy = _load_optional_artifact_gap_policy(artifact_gap_policy_path)

    blockers: list[ReleaseGateIssue] = []
    warnings: list[ReleaseGateIssue] = []
    blockers.extend(
        _artifact_blockers(
            artifact_checks,
            artifact_gap_policy=artifact_gap_policy,
            allow_known_artifact_gaps=allow_known_artifact_gaps,
        )
    )
    warnings.extend(
        _artifact_warnings(
            artifact_checks,
            artifact_gap_policy=artifact_gap_policy,
            allow_known_artifact_gaps=allow_known_artifact_gaps,
        )
    )
    blockers.extend(_dependency_blockers(dependency_checks))
    warnings.extend(_dependency_warnings(dependency_checks, hash_files=hash_files))

    return ReleaseGateReport(
        verdict="pass" if not blockers else "fail",
        provenance_path=str(provenance_path),
        dependency_spec_path=str(dependency_spec_path),
        dependency_local_spec_path=None
        if dependency_local_spec_path is None
        else str(dependency_local_spec_path),
        dependency_local_spec_present=False
        if dependency_local_spec_path is None
        else Path(dependency_local_spec_path).is_file(),
        artifact_gap_policy_path=None
        if artifact_gap_policy_path is None
        else str(artifact_gap_policy_path),
        artifact_gap_policy_present=False
        if artifact_gap_policy_path is None
        else Path(artifact_gap_policy_path).is_file(),
        allow_known_artifact_gaps=allow_known_artifact_gaps,
        hash_files=hash_files,
        artifact_status_counts=_artifact_status_counts(artifact_checks),
        dependency_counts=_dependency_counts(dependency_checks),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def render_release_gate_json(report: ReleaseGateReport) -> str:
    return json.dumps(report.to_dict(), indent=2) + "\n"


def render_release_gate_markdown(report: ReleaseGateReport) -> str:
    lines = ["# Release gate", "", f"Verdict: **{report.verdict}**", ""]
    lines.extend(
        [
            f"Provenance: `{report.provenance_path}`",
            f"Dependency spec: `{report.dependency_spec_path}`",
            "Dependency local override: "
            + (
                "none"
                if report.dependency_local_spec_path is None
                else f"`{report.dependency_local_spec_path}` "
                f"({'present' if report.dependency_local_spec_present else 'absent'})"
            ),
            "Artifact gap policy: "
            + (
                "none"
                if report.artifact_gap_policy_path is None
                else f"`{report.artifact_gap_policy_path}` "
                f"({'present' if report.artifact_gap_policy_present else 'absent'})"
            ),
            f"Allow known artifact gaps: {'yes' if report.allow_known_artifact_gaps else 'no'}",
            f"Hash files: {'yes' if report.hash_files else 'no'}",
            "",
            "## Counts",
            "",
            "### Artifact statuses",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )
    for status, count in report.artifact_status_counts.items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "### Dependency statuses", "", "| Status | Count |", "|---|---:|"])
    for status, count in report.dependency_counts.items():
        lines.append(f"| {status} | {count} |")

    _append_issues(lines, "Blockers", report.blockers)
    _append_issues(lines, "Warnings", report.warnings)
    return "\n".join(lines).rstrip() + "\n"


def _artifact_blockers(
    checks: list[ArtifactCheck],
    *,
    artifact_gap_policy: ArtifactGapPolicy | None,
    allow_known_artifact_gaps: bool,
) -> list[ReleaseGateIssue]:
    issues: list[ReleaseGateIssue] = []
    missing = [
        check
        for check in checks
        if check.status == "missing"
        and not _is_allowed_known_gap(
            check,
            artifact_gap_policy=artifact_gap_policy,
            allow_known_artifact_gaps=allow_known_artifact_gaps,
        )
    ]
    if missing:
        issues.append(
            ReleaseGateIssue(
                code="artifact_missing",
                severity="blocker",
                subject="artifact provenance",
                detail=f"{len(missing)} declared artifact paths are missing without an allowed gap decision",
            )
        )
    not_files = [check for check in checks if check.status == "not_file"]
    if not_files:
        issues.append(
            ReleaseGateIssue(
                code="artifact_not_file",
                severity="blocker",
                subject="artifact provenance",
                detail=f"{len(not_files)} declared artifact paths are not files",
            )
        )
    return issues


def _artifact_warnings(
    checks: list[ArtifactCheck],
    *,
    artifact_gap_policy: ArtifactGapPolicy | None,
    allow_known_artifact_gaps: bool,
) -> list[ReleaseGateIssue]:
    issues: list[ReleaseGateIssue] = []
    declared_absent = [check for check in checks if check.status == "declared_absent"]
    if declared_absent:
        issues.append(
            ReleaseGateIssue(
                code="artifact_declared_absent",
                severity="warning",
                subject="artifact provenance",
                detail=f"{len(declared_absent)} artifacts are explicitly declared absent",
            )
        )
    not_declared = [check for check in checks if check.status == "not_declared"]
    if not_declared:
        issues.append(
            ReleaseGateIssue(
                code="artifact_not_declared",
                severity="warning",
                subject="artifact provenance",
                detail=f"{len(not_declared)} seeds have no artifact declarations",
            )
        )
    known_gaps = [
        check
        for check in checks
        if check.status == "missing"
        and _is_allowed_known_gap(
            check,
            artifact_gap_policy=artifact_gap_policy,
            allow_known_artifact_gaps=allow_known_artifact_gaps,
        )
    ]
    if known_gaps:
        issues.append(
            ReleaseGateIssue(
                code="artifact_known_gap_allowed",
                severity="warning",
                subject="artifact provenance",
                detail=f"{len(known_gaps)} missing artifacts are allowed by explicit gap decisions; this is not full reproducibility",
            )
        )
    return issues


def _load_optional_artifact_gap_policy(path: str | Path | None) -> ArtifactGapPolicy | None:
    if path is None:
        return None
    policy_path = Path(path)
    if not policy_path.is_file():
        return None
    return load_artifact_gap_policy(policy_path)


def _is_allowed_known_gap(
    check: ArtifactCheck,
    *,
    artifact_gap_policy: ArtifactGapPolicy | None,
    allow_known_artifact_gaps: bool,
) -> bool:
    if not allow_known_artifact_gaps or artifact_gap_policy is None:
        return False
    return artifact_gap_policy.decision_for(check) is not None


def _dependency_blockers(checks: list[ExternalDependencyCheck]) -> list[ReleaseGateIssue]:
    issues: list[ReleaseGateIssue] = []
    missing = [check.name for check in checks if not check.present]
    if missing:
        issues.append(
            ReleaseGateIssue(
                code="dependency_missing",
                severity="blocker",
                subject="external dependencies",
                detail="missing dependencies: " + ", ".join(missing),
            )
        )

    required_missing = [
        f"{check.name}:{item.path}"
        for check in checks
        for item in check.required_paths
        if not item.present
    ]
    if required_missing:
        issues.append(
            ReleaseGateIssue(
                code="dependency_required_path_missing",
                severity="blocker",
                subject="external dependencies",
                detail="missing required paths: " + ", ".join(required_missing),
            )
        )

    dirty = [check.name for check in checks if check.git_dirty]
    if dirty:
        issues.append(
            ReleaseGateIssue(
                code="dependency_dirty",
                severity="blocker",
                subject="external dependencies",
                detail="dirty dependency clones: " + ", ".join(dirty),
            )
        )

    revision_mismatch = [check.name for check in checks if check.git_head_matches_expected is False]
    if revision_mismatch:
        issues.append(
            ReleaseGateIssue(
                code="dependency_revision_mismatch",
                severity="blocker",
                subject="external dependencies",
                detail="revision mismatches: " + ", ".join(revision_mismatch),
            )
        )

    remote_mismatch = [check.name for check in checks if check.git_remote_matches_expected is False]
    if remote_mismatch:
        issues.append(
            ReleaseGateIssue(
                code="dependency_remote_mismatch",
                severity="blocker",
                subject="external dependencies",
                detail="remote mismatches: " + ", ".join(remote_mismatch),
            )
        )

    hash_mismatch = [
        f"{check.name}:{item.path}"
        for check in checks
        for item in check.required_paths
        if item.sha256_matches_expected is False
    ]
    if hash_mismatch:
        issues.append(
            ReleaseGateIssue(
                code="dependency_hash_mismatch",
                severity="blocker",
                subject="external dependencies",
                detail="hash mismatches: " + ", ".join(hash_mismatch),
            )
        )
    return issues


def _dependency_warnings(
    checks: list[ExternalDependencyCheck],
    *,
    hash_files: bool,
) -> list[ReleaseGateIssue]:
    issues: list[ReleaseGateIssue] = []
    expected_hashes = [
        f"{check.name}:{item.path}"
        for check in checks
        for item in check.required_paths
        if item.expected_sha256 is not None
    ]
    if expected_hashes and not hash_files:
        issues.append(
            ReleaseGateIssue(
                code="dependency_hashes_not_checked",
                severity="warning",
                subject="external dependencies",
                detail=f"{len(expected_hashes)} expected hashes were not checked; rerun with --hash-files",
            )
        )
    return issues


def _artifact_status_counts(checks: list[ArtifactCheck]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return dict(sorted(counts.items()))


def _dependency_counts(checks: list[ExternalDependencyCheck]) -> dict[str, int]:
    counts = {
        "present": sum(1 for check in checks if check.present),
        "missing": sum(1 for check in checks if not check.present),
        "dirty": sum(1 for check in checks if check.git_dirty),
        "revision_mismatch": sum(1 for check in checks if check.git_head_matches_expected is False),
        "remote_mismatch": sum(1 for check in checks if check.git_remote_matches_expected is False),
        "hash_mismatch": sum(
            1
            for check in checks
            for item in check.required_paths
            if item.sha256_matches_expected is False
        ),
        "required_path_missing": sum(
            1 for check in checks for item in check.required_paths if not item.present
        ),
    }
    return dict(sorted(counts.items()))


def _append_issues(lines: list[str], title: str, issues: tuple[ReleaseGateIssue, ...]) -> None:
    lines.extend(["", f"## {title}", ""])
    if not issues:
        lines.append("None.")
        return
    lines.extend(["| Code | Subject | Detail |", "|---|---|---|"])
    for issue in issues:
        lines.append(f"| {issue.code} | {issue.subject} | {issue.detail} |")
