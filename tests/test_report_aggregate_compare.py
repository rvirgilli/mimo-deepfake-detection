import json

import pytest

from mimodf.experiments.execution import complete_experiment_run, prepare_experiment_run
from mimodf.report.aggregate import (
    aggregate_records,
    render_aggregates_json,
    render_aggregates_markdown,
)
from mimodf.report.compare import compare_experiments, render_comparison_markdown
from mimodf.report.index import (
    RunIndexRecord,
    index_main_table_provenance,
    index_run_layout_root,
    load_run_index_jsonl,
    render_run_index_jsonl,
)

EXAMPLE_SPEC = "docs/current/examples/experiment_spec_v1_minimal.yaml"
PROVENANCE = "docs/current/main_table_provenance.yaml"


def _new_record(tmp_path, *, seed=42, metric=1.0):
    prepared = prepare_experiment_run(
        spec_path=EXAMPLE_SPEC, seed=seed, root=tmp_path, status="running"
    )
    complete_experiment_run(
        prepared,
        metrics={"la_eer": metric, "df_eer": metric + 1},
        artifacts={"score_file": f"scores_{seed}.txt"},
    )
    return index_run_layout_root(tmp_path)[-1]


def test_load_run_index_jsonl_roundtrip(tmp_path):
    record = _new_record(tmp_path)
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(render_run_index_jsonl([record]))

    loaded = load_run_index_jsonl(index_path)

    assert loaded == [record]
    assert loaded[0].protocol_ids["evaluation_batch_size"] == "64"


def test_aggregate_records_computes_sample_std_from_historical_index():
    records = index_main_table_provenance(PROVENANCE)
    aggregate = next(
        item for item in aggregate_records(records) if item.experiment_id == "wav2vec2_frozen"
    )
    metrics = {metric.metric: metric for metric in aggregate.metrics}

    assert aggregate.min_reproducibility_tier == 1
    assert aggregate.statuses == {"partial": 5}
    assert aggregate.intents == {"historical": 5}
    assert metrics["la_eer"].n == 5
    assert metrics["la_eer"].mean == pytest.approx(8.046)
    assert metrics["la_eer"].sample_std == pytest.approx(0.731526, abs=1e-6)


def test_render_aggregates_outputs_json_and_markdown():
    records = [
        RunIndexRecord(
            record_schema="run-index-record/v1",
            source_type="new_run",
            experiment_id="exp",
            run_id="exp/seed_1",
            seed=1,
            status="completed",
            intent="confirmatory",
            reproducibility_tier=3,
            metrics={"la_eer": 1.0},
        )
    ]
    aggregates = aggregate_records(records)

    assert json.loads(render_aggregates_json(aggregates))[0]["experiment_id"] == "exp"
    assert "| Experiment | Seeds | Tier |" in render_aggregates_markdown(aggregates)


def test_compare_strict_passes_for_same_protocol_and_seed_sets(tmp_path):
    left = _new_record(tmp_path / "left", seed=42, metric=1.0)
    right = RunIndexRecord(
        **{
            **left.to_dict(),
            "experiment_id": "other_exp",
            "run_id": "other_exp/seed_42",
            "component_ids": {**left.component_ids, "frontend": "frontend:wav2vec2-xlsr-300m/v1"},
            "warnings": [],
            "intent": "confirmatory",
        }
    )
    left = RunIndexRecord(**{**left.to_dict(), "intent": "confirmatory", "warnings": []})

    report = compare_experiments(
        [left, right], [left.experiment_id, right.experiment_id], strict=True
    )

    assert report.passed is True
    assert all(check.passed for check in report.checks)


def test_compare_strict_fails_for_historical_missing_protocol_ids():
    records = index_main_table_provenance(PROVENANCE)

    report = compare_experiments(records, ["wav2vec2_frozen", "mimo_frozen"], strict=True)

    assert report.passed is False
    same_protocol = next(check for check in report.checks if check.name == "same_protocol")
    assert same_protocol.passed is False
    assert "missing protocol ids" in same_protocol.message


def test_compare_non_strict_warns_but_does_not_fail_for_seed_mismatch():
    records = index_main_table_provenance(PROVENANCE)

    report = compare_experiments(records, ["wav2vec2_full_local", "mimo_adapter"], strict=False)

    assert report.passed is True
    same_seed = next(check for check in report.checks if check.name == "same_seed_set")
    assert same_seed.passed is False
    assert same_seed.severity == "warning"
    assert "wav2vec2_full_local" in render_comparison_markdown(report)


def test_compare_strict_fails_when_batch_sensitive_record_lacks_batch_policy():
    sensitive = RunIndexRecord(
        record_schema="run-index-record/v1",
        source_type="new_run",
        experiment_id="mimo_like",
        run_id="mimo_like/seed_1",
        seed=1,
        status="completed",
        intent="confirmatory",
        reproducibility_tier=3,
        protocol_ids={"checkpoint_selection": "dev_eer"},
        warnings=["bf16_flashattention_batch_size_sensitive"],
    )
    other = RunIndexRecord(
        record_schema="run-index-record/v1",
        source_type="new_run",
        experiment_id="other",
        run_id="other/seed_1",
        seed=1,
        status="completed",
        intent="confirmatory",
        reproducibility_tier=3,
        protocol_ids={"checkpoint_selection": "dev_eer"},
    )

    report = compare_experiments([sensitive, other], ["mimo_like", "other"], strict=True)

    assert report.passed is False
    batch_check = next(check for check in report.checks if check.name == "batch_size_policy")
    assert batch_check.passed is False
