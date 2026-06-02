import json
from pathlib import Path

import pytest
import yaml

from mimodf.components.registry import get_component
from mimodf.experiments.execution import (
    complete_experiment_run,
    fail_experiment_run,
    prepare_experiment_run,
)
from mimodf.experiments.layout import build_run_layout
from mimodf.experiments.manifest import RunManifest
from mimodf.experiments.spec import SpecValidationError, load_experiment_spec

EXAMPLE_SPEC = Path("docs/current/examples/experiment_spec_v1_minimal.yaml")


def _write_variant(tmp_path, mutator):
    data = yaml.safe_load(EXAMPLE_SPEC.read_text())
    mutator(data)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_experiment_spec_example_validates_and_hashes_deterministically():
    spec = load_experiment_spec(EXAMPLE_SPEC)

    first = spec.spec_hash()
    second = spec.spec_hash()

    assert first == second
    assert first.startswith("sha256:")
    resolved = spec.resolved(resolved_at="2026-05-26T00:00:00+00:00")
    assert resolved["spec_hash"] == first
    assert resolved["component_versions"]["frontend"] == "frontend:mimo-continuous-native50/v1"
    assert "bf16_flashattention_batch_size_sensitive" in resolved["frontend_facts"]["known_caveats"]


def test_experiment_spec_rejects_missing_checkpoint_selection(tmp_path):
    path = _write_variant(tmp_path, lambda data: data["protocol"].pop("checkpoint_selection"))

    with pytest.raises(SpecValidationError, match="checkpoint_selection"):
        load_experiment_spec(path)


def test_experiment_spec_rejects_missing_eval_batch_size(tmp_path):
    path = _write_variant(tmp_path, lambda data: data["evaluation"].pop("batch_size"))

    with pytest.raises(SpecValidationError, match="evaluation.batch_size"):
        load_experiment_spec(path)


def test_experiment_spec_rejects_unknown_component(tmp_path):
    path = _write_variant(
        tmp_path,
        lambda data: data["model"].__setitem__("frontend", "frontend:unknown/v1"),
    )

    with pytest.raises(SpecValidationError, match="unknown component id"):
        load_experiment_spec(path)


def test_component_registry_exposes_mimo_batch_sensitivity_caveat():
    component = get_component("frontend:mimo-continuous-native50/v1")

    assert component.kind == "frontend"
    assert component.metadata["sample_rate"] == 24000
    assert "bf16_flashattention_batch_size_sensitive" in component.caveats


def test_component_registry_exposes_wavlm_candidate():
    component = get_component("frontend:wavlm-base-plus/hf-b211941/v1")

    assert component.kind == "frontend"
    assert component.metadata["sample_rate"] == 16000
    assert component.metadata["feature_dim"] == 768
    assert component.metadata["revision"] == "b21194173c0af7e94822c1776d162e2659fd4761"
    assert "wave2_full_source_holdout_completed" in component.caveats


def test_run_manifest_roundtrip_from_resolved_spec(tmp_path):
    spec = load_experiment_spec(EXAMPLE_SPEC)
    resolved = spec.resolved(resolved_at="2026-05-26T00:00:00+00:00")
    manifest = RunManifest.from_resolved_spec(resolved, seed=42, status="planned")

    path = manifest.save(tmp_path / "manifest.json")
    loaded = RunManifest.load(path)

    assert loaded.run_id.endswith("/seed_42")
    assert loaded.status == "planned"
    assert loaded.protocol == {
        "checkpoint_selection": "dev_eer",
        "leakage_policy": "no_eval_selection",
    }
    assert loaded.evaluation == {"batch_size": 64}
    assert "bf16_flashattention_batch_size_sensitive" in loaded.warnings
    assert json.loads(path.read_text())["schema_version"] == "run-manifest/v1"


def test_run_manifest_rejects_missing_protocol_fact():
    manifest = RunManifest(
        run_id="run",
        experiment_id="exp",
        spec_hash="sha256:" + "0" * 64,
        seed=42,
        intent="exploratory",
        status="planned",
        protocol={"checkpoint_selection": "dev_eer"},
        model={"frontend": "frontend:mimo-continuous-native50/v1", "backend": "backend:aasist/v1"},
        evaluation={"batch_size": 64},
    )

    with pytest.raises(ValueError, match="leakage_policy"):
        manifest.to_dict()


def test_prepare_experiment_run_writes_resolved_spec_and_manifest(tmp_path):
    prepared = prepare_experiment_run(
        spec_path=EXAMPLE_SPEC,
        seed=42,
        root=tmp_path,
        status="planned",
    )

    assert prepared.layout.resolved_spec_path.is_file()
    assert prepared.layout.manifest_path.is_file()
    assert prepared.manifest.status == "planned"
    assert prepared.manifest.seed == 42

    complete_experiment_run(
        prepared,
        metrics={"track": "LA"},
        artifacts={"score_file": "scores.txt"},
    )
    completed = RunManifest.load(prepared.layout.manifest_path)
    assert completed.status == "completed"
    assert completed.metrics == {"track": "LA"}
    assert completed.artifacts == [{"name": "score_file", "path": "scores.txt"}]


def test_prepare_experiment_run_rejects_undeclared_seed(tmp_path):
    with pytest.raises(ValueError, match="not declared"):
        prepare_experiment_run(spec_path=EXAMPLE_SPEC, seed=999, root=tmp_path)


def test_fail_experiment_run_records_failure(tmp_path):
    prepared = prepare_experiment_run(spec_path=EXAMPLE_SPEC, seed=42, root=tmp_path)

    fail_experiment_run(prepared, RuntimeError("boom"), metrics={"track": "LA"})

    failed = RunManifest.load(prepared.layout.manifest_path)
    assert failed.status == "failed"
    assert failed.failures == ["boom"]
    assert failed.metrics == {"track": "LA"}


def test_run_layout_creates_expected_dirs_and_guards_overwrite(tmp_path):
    layout = build_run_layout(
        root=tmp_path,
        experiment_id="exp",
        spec_hash="sha256:" + "a" * 64,
        seed=123,
    ).create()

    assert (
        layout.resolved_spec_path
        == tmp_path / "exp" / ("a" * 64) / "seed_123" / "resolved_spec.yaml"
    )
    assert layout.logs_dir.is_dir()
    assert layout.eval_dir.is_dir()

    with pytest.raises(FileExistsError):
        layout.create()
