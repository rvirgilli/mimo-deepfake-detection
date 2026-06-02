import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimodf.cli import main
from mimodf.experiments.execution import prepare_experiment_run
from mimodf.experiments.manifest import RunManifest
from mimodf.training.loop import TrainingRunResult
from mimodf.training.run import build_legacy_asvspoof_plan, run_legacy_asvspoof_training


def test_legacy_asvspoof_plan_is_dependency_light_and_explicit(tmp_path):
    plan = build_legacy_asvspoof_plan(
        config_path="configs/publish/mimo_full.yaml",
        database_path=tmp_path / "db",
        protocols_path=tmp_path / "protocols",
        frontend_name="mimo",
        validation_protocol="asvspoof2021_fast",
        freeze_frontend=False,
        epochs=2,
    )

    rendered = plan.to_dict()

    assert rendered["config"]["model"] == "MiMo"
    assert rendered["frontend"]["name"] == "mimo"
    assert rendered["frontend"]["freeze"] is False
    assert rendered["loop"]["epochs"] == 2
    assert rendered["required_paths"]["validation_protocol"] == "asvspoof2021_fast"
    assert rendered["required_paths"]["validation_trial_file"].endswith(
        "ASVspoof2021.LA.cm.eval.fast.trl.txt"
    )
    assert rendered["required_paths"]["validation_key_file"].endswith(
        "ASVspoof2021.LA.cm.eval.fast.key.txt"
    )


def test_legacy_asvspoof_plan_rejects_implicit_validation_protocol(tmp_path):
    with pytest.raises(ValueError, match="validation_protocol"):
        build_legacy_asvspoof_plan(
            config_path="configs/publish/mimo_full.yaml",
            database_path=tmp_path / "db",
            protocols_path=tmp_path / "protocols",
            frontend_name="mimo",
            validation_protocol="implicit",
        )


def test_legacy_asvspoof_plan_rejects_config_cli_protocol_mismatch(tmp_path):
    with pytest.raises(ValueError, match="does not match config protocol"):
        build_legacy_asvspoof_plan(
            config_path="configs/publish/mimo_full.yaml",
            database_path=tmp_path / "db",
            protocols_path=tmp_path / "protocols",
            frontend_name="mimo",
            validation_protocol="asvspoof2019_dev",
        )


def test_train_legacy_asvspoof_dry_run_cli_prints_plan(tmp_path, capsys):
    rc = main(
        [
            "train",
            "legacy-asvspoof",
            "--config",
            "configs/publish/wav2vec2_adapter.yaml",
            "--out",
            str(tmp_path / "run"),
            "--database-path",
            str(tmp_path / "db"),
            "--protocols-path",
            str(tmp_path / "protocols"),
            "--validation-protocol",
            "asvspoof2021_fast",
            "--frontend",
            "wav2vec2",
            "--max-train-batches",
            "1",
            "--max-val-batches",
            "1",
            "--dry-run",
        ]
    )

    assert rc == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["config"]["model"] == "wav2vec2"
    assert rendered["frontend"]["freeze"] is False
    assert rendered["frontend"]["finetune_config"]["strategy"] == "adapter"
    assert rendered["loop"]["max_train_batches"] == 1
    assert rendered["loop"]["max_val_batches"] == 1
    assert rendered["required_paths"]["validation_protocol"] == "asvspoof2021_fast"
    assert rendered["required_paths"]["validation_key_file"].endswith(
        "ASVspoof2021.LA.cm.eval.fast.key.txt"
    )


def test_train_legacy_asvspoof_cli_rejects_non_object_rawboost_args(tmp_path):
    with pytest.raises(SystemExit, match="rawboost-args-json"):
        main(
            [
                "train",
                "legacy-asvspoof",
                "--config",
                "configs/publish/mimo_full.yaml",
                "--out",
                str(tmp_path / "run"),
                "--database-path",
                str(tmp_path / "db"),
                "--protocols-path",
                str(tmp_path / "protocols"),
                "--validation-protocol",
                "asvspoof2021_fast",
                "--frontend",
                "mimo",
                "--rawboost-args-json",
                "[]",
                "--dry-run",
            ]
        )


def test_train_legacy_asvspoof_cli_passes_versioned_experiment_run(monkeypatch, tmp_path):
    captured = {}

    def fake_run(plan, *, output_dir, experiment_run=None):
        captured["experiment_run"] = experiment_run
        captured["output_dir"] = output_dir
        return TrainingRunResult(
            best_checkpoint=Path(output_dir) / "checkpoints" / "epoch_0.pt",
            manifest_path=Path(output_dir) / "manifest.json",
            best_val_loss=0.1,
            final_train_loss=0.2,
            epochs_completed=1,
        )

    monkeypatch.setattr("mimodf.cli.run_legacy_asvspoof_training", fake_run)

    rc = main(
        [
            "train",
            "legacy-asvspoof",
            "--config",
            "configs/publish/wav2vec2_adapter.yaml",
            "--out",
            str(tmp_path / "run"),
            "--database-path",
            str(tmp_path / "db"),
            "--protocols-path",
            str(tmp_path / "protocols"),
            "--validation-protocol",
            "asvspoof2021_fast",
            "--frontend",
            "wav2vec2",
            "--experiment-spec",
            "docs/current/examples/experiment_spec_v1_minimal.yaml",
            "--run-seed",
            "42",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )

    assert rc == 0
    assert captured["output_dir"] == str(tmp_path / "run")
    assert captured["experiment_run"] is not None
    assert captured["experiment_run"].layout.manifest_path.is_file()


def test_run_legacy_asvspoof_training_wires_tested_seams(monkeypatch, tmp_path):
    plan = build_legacy_asvspoof_plan(
        config_path="configs/publish/mimo_full.yaml",
        database_path=tmp_path / "db",
        protocols_path=tmp_path / "protocols",
        frontend_name="mimo",
        validation_protocol="asvspoof2021_fast",
    )
    calls = {}

    monkeypatch.setattr(
        "mimodf.training.run.build_asvspoof_loaders",
        lambda data: SimpleNamespace(train_loader="train_loader", val_loader="val_loader"),
    )
    monkeypatch.setattr("mimodf.training.run.build_legacy_frontend", lambda frontend: "frontend")
    monkeypatch.setattr(
        "mimodf.training.run.build_legacy_model",
        lambda frontend, model_settings: "model",
    )

    def fake_build_optimizer(config, model, *, encoder_params=None, backend_params=None):
        calls["optimizer"] = (config, model, encoder_params, backend_params)
        return "optimizer"

    def fake_train_with_components(*, config, components, output_dir, settings):
        calls["train"] = (config, components, Path(output_dir), settings)
        return TrainingRunResult(
            best_checkpoint=Path(output_dir) / "checkpoints" / "epoch_0.pt",
            manifest_path=Path(output_dir) / "manifest.json",
            best_val_loss=0.1,
            final_train_loss=0.2,
            epochs_completed=1,
        )

    monkeypatch.setattr("mimodf.training.run.build_optimizer", fake_build_optimizer)
    monkeypatch.setattr("mimodf.training.run.train_with_components", fake_train_with_components)

    experiment_run = prepare_experiment_run(
        spec_path="docs/current/examples/experiment_spec_v1_minimal.yaml",
        seed=42,
        root=tmp_path / "runs",
        status="running",
    )

    result = run_legacy_asvspoof_training(
        plan,
        output_dir=tmp_path / "run",
        experiment_run=experiment_run,
    )

    assert result.best_val_loss == pytest.approx(0.1)
    assert calls["optimizer"][1] == "model"
    _, components, output_dir, _ = calls["train"]
    assert components.model == "model"
    assert components.train_loader == "train_loader"
    assert components.val_loader == "val_loader"
    assert components.optimizer == "optimizer"
    assert output_dir == tmp_path / "run"
    manifest = RunManifest.load(experiment_run.layout.manifest_path)
    assert manifest.status == "completed"
    assert manifest.metrics == {
        "best_val_loss": 0.1,
        "final_train_loss": 0.2,
        "epochs_completed": 1,
        "checkpoint_metric": "val_loss",
    }
    assert manifest.artifacts == [
        {"name": "best_checkpoint", "path": str(tmp_path / "run" / "checkpoints" / "epoch_0.pt")},
        {"name": "training_manifest", "path": str(tmp_path / "run" / "manifest.json")},
    ]


def test_run_legacy_asvspoof_training_records_versioned_failure(monkeypatch, tmp_path):
    plan = build_legacy_asvspoof_plan(
        config_path="configs/publish/mimo_full.yaml",
        database_path=tmp_path / "db",
        protocols_path=tmp_path / "protocols",
        frontend_name="mimo",
        validation_protocol="asvspoof2021_fast",
    )
    experiment_run = prepare_experiment_run(
        spec_path="docs/current/examples/experiment_spec_v1_minimal.yaml",
        seed=42,
        root=tmp_path / "runs",
        status="running",
    )

    def fail_loaders(data):
        raise RuntimeError("loader boom")

    monkeypatch.setattr("mimodf.training.run.build_asvspoof_loaders", fail_loaders)

    with pytest.raises(RuntimeError, match="loader boom"):
        run_legacy_asvspoof_training(
            plan,
            output_dir=tmp_path / "run",
            experiment_run=experiment_run,
        )

    manifest = RunManifest.load(experiment_run.layout.manifest_path)
    assert manifest.status == "failed"
    assert manifest.metrics == {"checkpoint_metric": "val_loss"}
    assert manifest.failures == ["loader boom"]
