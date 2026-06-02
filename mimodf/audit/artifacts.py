"""Artifact checks for structured provenance files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mimodf.provenance import MainTableProvenance, load_main_table_provenance

_DECLARED_ABSENT = {"", "missing", "external", "none local", "not local"}


@dataclass(frozen=True)
class ArtifactCheck:
    row_id: str
    seed_id: str
    kind: str
    value: str
    status: str
    sha256: str | None = None


def check_artifacts(
    provenance: MainTableProvenance,
    *,
    root: str | Path = ".",
) -> list[ArtifactCheck]:
    """Check declared artifact paths without treating expected gaps as fatal."""

    root_path = Path(root)
    checks: list[ArtifactCheck] = []

    for row in provenance.rows:
        for seed in row.seeds:
            if not seed.artifacts:
                checks.append(
                    ArtifactCheck(
                        row_id=row.id,
                        seed_id=seed.id,
                        kind="artifacts",
                        value="",
                        status="not_declared",
                    )
                )
                continue

            for kind, value in sorted(seed.artifacts.items()):
                checks.append(_check_one(root_path, row.id, seed.id, kind, value))

    return checks


def check_artifacts_from_file(path: str | Path, *, root: str | Path = ".") -> list[ArtifactCheck]:
    return check_artifacts(load_main_table_provenance(path), root=root)


def render_artifact_checks_markdown(checks: list[ArtifactCheck]) -> str:
    lines = [
        "| Row | Seed | Artifact | Status | Path/value | SHA-256 |",
        "|---|---|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.row_id} | {check.seed_id} | {check.kind} | {check.status} | "
            f"{check.value or '---'} | {check.sha256 or '---'} |"
        )
    return "\n".join(lines) + "\n"


def render_artifact_checks_json(checks: list[ArtifactCheck]) -> str:
    return json.dumps([asdict(check) for check in checks], indent=2) + "\n"


def _check_one(root: Path, row_id: str, seed_id: str, kind: str, value: str) -> ArtifactCheck:
    normalized = value.strip().lower()
    if normalized in _DECLARED_ABSENT:
        return ArtifactCheck(row_id, seed_id, kind, value, "declared_absent")

    path = Path(value)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        return ArtifactCheck(row_id, seed_id, kind, value, "missing")
    if not path.is_file():
        return ArtifactCheck(row_id, seed_id, kind, value, "not_file")

    return ArtifactCheck(row_id, seed_id, kind, value, "present", _sha256(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
