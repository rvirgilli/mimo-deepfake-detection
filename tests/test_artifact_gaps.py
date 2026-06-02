import json

import pytest

from mimodf.audit.artifact_gaps import (
    load_artifact_gap_policy,
    render_artifact_gap_policy_json,
    render_artifact_gap_policy_markdown,
)
from mimodf.cli import main


def test_current_artifact_gap_policy_covers_known_gaps():
    policy = load_artifact_gap_policy()

    assert policy.version == 1
    assert len(policy.gaps) == 11
    assert (
        policy.decision_for(
            type(
                "Check",
                (),
                {
                    "row_id": "mimo_adapter",
                    "seed_id": "2024",
                    "kind": "checkpoint",
                    "value": "outputs/2026-01-27/18-01-37/.../epoch_4_eer_9.87.pth",
                },
            )()
        )
        is not None
    )


def test_artifact_gap_policy_rejects_duplicate_keys(tmp_path):
    policy = tmp_path / "gaps.yaml"
    policy.write_text(
        """
version: 1
gaps:
  - row_id: row
    seed_id: seed
    kind: checkpoint
    value: missing.pt
    decision: known_missing
    reason: fixture
    action: fixture
  - row_id: row
    seed_id: seed
    kind: checkpoint
    value: missing.pt
    decision: known_missing
    reason: fixture
    action: fixture
""".strip()
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_artifact_gap_policy(policy)


def test_artifact_gap_renderers_and_cli(capsys):
    policy = load_artifact_gap_policy()

    rendered_json = json.loads(render_artifact_gap_policy_json(policy))
    rendered_markdown = render_artifact_gap_policy_markdown(policy)

    assert rendered_json["version"] == 1
    assert "# Artifact gap decisions" in rendered_markdown
    assert "mimo_adapter" in rendered_markdown

    rc = main(["audit", "artifact-gaps", "--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["version"] == 1
