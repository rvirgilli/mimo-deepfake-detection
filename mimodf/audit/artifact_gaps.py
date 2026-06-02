"""Explicit decisions for known historical artifact gaps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from mimodf.audit.artifacts import ArtifactCheck

DEFAULT_ARTIFACT_GAP_POLICY = Path("docs/current/artifact_gap_decisions.yaml")


@dataclass(frozen=True)
class ArtifactGapDecision:
    row_id: str
    seed_id: str
    kind: str
    value: str
    decision: str
    reason: str
    action: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.row_id, self.seed_id, self.kind, self.value)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactGapPolicy:
    version: int
    policy: str
    gaps: tuple[ArtifactGapDecision, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def decision_for(self, check: ArtifactCheck) -> ArtifactGapDecision | None:
        key = (check.row_id, check.seed_id, check.kind, check.value)
        for gap in self.gaps:
            if gap.key == key:
                return gap
        return None


def load_artifact_gap_policy(path: str | Path = DEFAULT_ARTIFACT_GAP_POLICY) -> ArtifactGapPolicy:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("artifact gap policy root must be a mapping")
    gaps = data.get("gaps")
    if not isinstance(gaps, list):
        raise ValueError("artifact gap policy must contain a gaps list")
    policy = ArtifactGapPolicy(
        version=int(data.get("version", 0)),
        policy=str(data.get("policy") or ""),
        gaps=tuple(_gap_from_dict(item) for item in gaps),
    )
    if policy.version != 1:
        raise ValueError(f"unsupported artifact gap policy version: {policy.version}")
    keys = [gap.key for gap in policy.gaps]
    if len(keys) != len(set(keys)):
        raise ValueError("artifact gap policy contains duplicate gap keys")
    return policy


def render_artifact_gap_policy_json(policy: ArtifactGapPolicy) -> str:
    return json.dumps(policy.to_dict(), indent=2) + "\n"


def render_artifact_gap_policy_markdown(policy: ArtifactGapPolicy) -> str:
    lines = ["# Artifact gap decisions", "", policy.policy, ""]
    lines.extend(
        [
            "| Row | Seed | Artifact | Decision | Path/value | Action |",
            "|---|---|---|---|---|---|",
        ]
    )
    for gap in policy.gaps:
        lines.append(
            f"| {gap.row_id} | {gap.seed_id} | {gap.kind} | {gap.decision} | "
            f"`{gap.value}` | {gap.action} |"
        )
    return "\n".join(lines) + "\n"


def _gap_from_dict(data: Any) -> ArtifactGapDecision:
    if not isinstance(data, dict):
        raise ValueError("artifact gap entries must be mappings")
    return ArtifactGapDecision(
        row_id=_required_string(data, "row_id"),
        seed_id=_required_string(data, "seed_id"),
        kind=_required_string(data, "kind"),
        value=_required_string(data, "value"),
        decision=_required_string(data, "decision"),
        reason=_required_string(data, "reason"),
        action=_required_string(data, "action"),
    )


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact gap {key} must be a non-empty string")
    return value
