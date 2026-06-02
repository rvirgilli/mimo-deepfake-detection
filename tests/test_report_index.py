import json

from mimodf.experiments.execution import complete_experiment_run, prepare_experiment_run
from mimodf.report.index import (
    build_run_index,
    index_main_table_provenance,
    index_run_layout_root,
    render_run_index_jsonl,
    render_run_index_markdown,
)

EXAMPLE_SPEC = "docs/current/examples/experiment_spec_v1_minimal.yaml"
PROVENANCE = "docs/current/main_table_provenance.yaml"


def test_index_run_layout_root_reads_versioned_manifest(tmp_path):
    prepared = prepare_experiment_run(
        spec_path=EXAMPLE_SPEC,
        seed=42,
        root=tmp_path,
        status="running",
    )
    complete_experiment_run(
        prepared,
        metrics={"track": "LA", "score_official": False},
        artifacts={"score_file": "scores.txt"},
    )

    records = index_run_layout_root(tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record.record_schema == "run-index-record/v1"
    assert record.source_type == "new_run"
    assert record.experiment_id == "controlled_mimo_wav2vec2_v1"
    assert record.seed == 42
    assert record.status == "completed"
    assert record.reproducibility_tier == 3
    assert record.component_ids["frontend"] == "frontend:mimo-continuous-native50/v1"
    assert record.protocol_ids["evaluation_batch_size"] == "64"
    assert record.metrics == {"track": "LA", "score_official": False}
    assert record.artifact_paths == ["scores.txt"]
    assert "bf16_flashattention_batch_size_sensitive" in record.warnings


def test_index_run_layout_root_skips_non_run_manifest(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "manifest.json").write_text('{"status": "completed"}\n')

    assert index_run_layout_root(tmp_path) == []


def test_index_main_table_provenance_creates_historical_seed_records():
    records = index_main_table_provenance(PROVENANCE)

    wav2vec2_adapter_123 = next(
        record for record in records if record.run_id == "wav2vec2_adapter/seed_123"
    )
    mimo_adapter_trial = next(
        record for record in records if record.run_id == "mimo_adapter/seed_trial39_42ish"
    )

    assert wav2vec2_adapter_123.source_type == "historical_provenance"
    assert wav2vec2_adapter_123.intent == "historical"
    assert wav2vec2_adapter_123.reproducibility_tier == 2
    assert wav2vec2_adapter_123.metrics["la_eer"] == 2.3284
    assert any(path.endswith("scores_LA_eval.txt") for path in wav2vec2_adapter_123.artifact_paths)
    assert mimo_adapter_trial.intent == "exploratory"


def test_build_run_index_combines_sources(tmp_path):
    prepared = prepare_experiment_run(spec_path=EXAMPLE_SPEC, seed=42, root=tmp_path)

    records = build_run_index([tmp_path], provenance_path=PROVENANCE)

    assert any(record.run_id == prepared.manifest.run_id for record in records)
    assert any(record.run_id == "mimo_frozen/seed_42" for record in records)


def test_render_run_index_jsonl_and_markdown(tmp_path):
    prepared = prepare_experiment_run(spec_path=EXAMPLE_SPEC, seed=42, root=tmp_path)
    records = index_run_layout_root(tmp_path)

    json_lines = render_run_index_jsonl(records).splitlines()
    markdown = render_run_index_markdown(records)

    assert json.loads(json_lines[0])["run_id"] == prepared.manifest.run_id
    assert "| Source | Experiment | Seed | Status |" in markdown
    assert "controlled_mimo_wav2vec2_v1" in markdown
