import json

import pytest

from mimodf.evaluation.legacy_components import (
    _checkpoint_state_dict,
    _legacy_feature_config,
    _legacy_feature_type,
    _legacy_model_settings,
)
from mimodf.evaluation.plan import build_evaluation_plan
from mimodf.evaluation.run import (
    EvaluationComponents,
    EvaluationRunSettings,
    run_evaluation,
)
from mimodf.experiments.execution import prepare_experiment_run
from mimodf.experiments.manifest import RunManifest
from mimodf.scoring.evaluate import EvaluationBatch, EvaluationItem


def write_config(path):
    path.write_text(
        """
model: wav2vec2
strategy: adapter
seed: 42
protocol:
  train_set: ASVspoof2019_LA_train
  validation_set: ASVspoof2021_LA_fast_eval_subset
  checkpoint_selection_set: ASVspoof2021_LA_fast_eval_subset
  eval_set: ASVspoof2021_LA_eval_and_DF_eval
optimizer:
  name: adam
  lr: 0.0001
  weight_decay: 0.0
  encoder_lr: null
scorer:
  la: official_asvspoof2021_la
  df: eer_only
""".strip()
    )


def passing_plan(tmp_path, *, score_name="scores.txt"):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    eval_root = tmp_path / "eval_root"
    scorer = tmp_path / "evaluate_2021_LA.py"
    write_config(config)
    checkpoint.write_text("checkpoint")
    eval_root.mkdir()
    scorer.write_text("print('score')\n")
    return build_evaluation_plan(
        config_path=config,
        checkpoint=checkpoint,
        eval_root=eval_root,
        score_out=tmp_path / score_name,
        track="LA",
        scorer=scorer,
    )


def fake_components():
    return EvaluationComponents(
        batches=[
            EvaluationBatch.from_items([EvaluationItem("utt_b", 2.0), EvaluationItem("utt_a", 1.0)])
        ],
        predict_batch=lambda inputs: [value * 10 for value in inputs],
    )


def test_run_evaluation_writes_scores_and_manifest(tmp_path):
    plan = passing_plan(tmp_path)
    manifest = tmp_path / "manifest.json"

    result = run_evaluation(
        plan,
        fake_components(),
        EvaluationRunSettings(manifest_out=str(manifest)),
        command=["mimodf", "eval", "run"],
    )

    assert result.score_file == str(tmp_path / "scores.txt")
    assert (tmp_path / "scores.txt").read_text().splitlines() == ["utt_a 10", "utt_b 20"]
    data = json.loads(manifest.read_text())
    assert data["status"] == "completed"
    assert data["artifacts"]["score_file"] == str(tmp_path / "scores.txt")


def test_run_evaluation_updates_versioned_experiment_manifest(tmp_path):
    plan = passing_plan(tmp_path)
    prepared = prepare_experiment_run(
        spec_path="docs/current/examples/experiment_spec_v1_minimal.yaml",
        seed=42,
        root=tmp_path / "runs",
        status="running",
    )

    run_evaluation(
        plan,
        fake_components(),
        EvaluationRunSettings(experiment_run=prepared),
    )

    manifest = RunManifest.load(prepared.layout.manifest_path)
    assert manifest.status == "completed"
    assert manifest.metrics == {"track": "LA", "score_official": False}
    assert manifest.artifacts == [{"name": "score_file", "path": str(tmp_path / "scores.txt")}]
    assert prepared.layout.resolved_spec_path.is_file()


def test_run_evaluation_records_versioned_experiment_failure(tmp_path):
    plan = passing_plan(tmp_path)
    prepared = prepare_experiment_run(
        spec_path="docs/current/examples/experiment_spec_v1_minimal.yaml",
        seed=42,
        root=tmp_path / "runs",
        status="running",
    )
    components = EvaluationComponents(
        batches=[EvaluationBatch.from_items([EvaluationItem("utt", 1.0)])],
        predict_batch=lambda inputs: [],
    )

    with pytest.raises(ValueError, match="returned 0 scores for 1 inputs"):
        run_evaluation(plan, components, EvaluationRunSettings(experiment_run=prepared))

    manifest = RunManifest.load(prepared.layout.manifest_path)
    assert manifest.status == "failed"
    assert manifest.metrics == {"track": "LA"}
    assert "returned 0 scores" in manifest.failures[0]


def test_run_evaluation_rejects_existing_score_without_overwrite(tmp_path):
    plan = passing_plan(tmp_path)
    (tmp_path / "scores.txt").write_text("old\n")

    with pytest.raises(FileExistsError):
        run_evaluation(plan, fake_components(), EvaluationRunSettings())


def test_run_evaluation_records_manifest_failure(tmp_path):
    plan = passing_plan(tmp_path)
    manifest = tmp_path / "manifest.json"
    components = EvaluationComponents(
        batches=[EvaluationBatch.from_items([EvaluationItem("utt", 1.0)])],
        predict_batch=lambda inputs: [],
    )

    with pytest.raises(ValueError, match="returned 0 scores for 1 inputs"):
        run_evaluation(plan, components, EvaluationRunSettings(manifest_out=str(manifest)))

    data = json.loads(manifest.read_text())
    assert data["status"] == "failed"
    assert "returned 0 scores" in data["error"]


def test_checkpoint_state_dict_accepts_raw_and_wrapped_checkpoints():
    raw = {"layer.weight": object()}
    wrapped = {"model_state_dict": raw, "epoch": 0}

    assert _checkpoint_state_dict(raw) is raw
    assert _checkpoint_state_dict(wrapped) is raw


def test_legacy_feature_settings_use_historical_frontend_config():
    legacy = {
        "frontend": {
            "feature": {
                "type": "continuous",
                "layer_select": {"num_layers": 12, "start_layer": 0},
            }
        }
    }

    assert (
        _legacy_feature_type(type("Settings", (), {"feature_type": "rvq_sum"})(), legacy)
        == "continuous"
    )
    assert _legacy_feature_config(legacy) == {"layer_select": {"num_layers": 12, "start_layer": 0}}


def test_legacy_model_settings_use_historical_architecture_config():
    settings = _legacy_model_settings(
        {
            "model": {
                "gat_dims": [128, 48],
                "pool_ratios": [0.77, 0.77, 0.77, 0.77],
                "temperatures": [2.0, 2.0, 100.0, 100.0],
                "dropout": 0.54,
                "dropout_way": 0.28,
                "filts_0": 192,
                "encoder_scale": 1.5,
                "projection": {
                    "type": "linear",
                    "hidden_dims": [512, 256],
                    "activation": "gelu",
                    "dropout": 0.1,
                    "use_batchnorm": True,
                },
            }
        }
    )

    assert settings.filts_0 == 192
    assert settings.encoder_scale == 1.5
    assert settings.gat_dims == [128, 48]
    assert settings.pool_ratios == [0.77, 0.77, 0.77, 0.77]
    assert settings.dropout == 0.54


def test_run_evaluation_fails_before_components_if_plan_failed(tmp_path):
    config = tmp_path / "config.yaml"
    write_config(config)
    plan = build_evaluation_plan(
        config_path=config,
        checkpoint=tmp_path / "missing.pt",
        eval_root=tmp_path,
        score_out=tmp_path / "scores.txt",
        track="DF",
    )

    with pytest.raises(ValueError, match="failing path checks"):
        run_evaluation(plan, fake_components(), EvaluationRunSettings())
