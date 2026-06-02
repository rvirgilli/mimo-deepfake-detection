import json

from mimodf.config import load_experiment_config
from mimodf.training.manifest import TrainingManifest, hash_existing_artifacts, sha256_file


def test_training_manifest_roundtrip_and_completion(tmp_path):
    cfg = load_experiment_config("configs/publish/mimo_full.yaml")
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_text("weights")

    manifest = TrainingManifest.start(
        cfg,
        command=["python", "-m", "mimodf", "train"],
        working_dir=".",
    )
    manifest.complete(
        metrics={"best_eer": 0.123, "best_epoch": 4},
        artifacts={"best_checkpoint": str(checkpoint)},
    )
    path = manifest.save(tmp_path / "manifest.json")
    loaded = TrainingManifest.load(path)

    assert loaded.run_id == manifest.run_id
    assert loaded.status == "completed"
    assert loaded.metrics == {"best_eer": 0.123, "best_epoch": 4}
    assert loaded.config["model"] == "MiMo"
    assert (
        loaded.config["protocol"]["checkpoint_selection_set"] == "ASVspoof2021_LA_fast_eval_subset"
    )
    assert loaded.git.commit
    assert loaded.artifact_hashes["best_checkpoint"] == sha256_file(checkpoint)
    assert loaded.duration_seconds is not None
    assert loaded.duration_seconds >= 0

    # Ensure JSON is plain data, not dataclass reprs.
    payload = json.loads(path.read_text())
    assert payload["git"]["branch"] == loaded.git.branch


def test_training_manifest_failure_status(tmp_path):
    cfg = load_experiment_config("configs/publish/wav2vec2_adapter.yaml")
    manifest = TrainingManifest.start(cfg, command=["train"], working_dir=".")

    manifest.fail(RuntimeError("boom"))

    assert manifest.status == "failed"
    assert manifest.error == "boom"
    assert manifest.ended_at is not None


def test_hash_existing_artifacts_skips_missing_files(tmp_path):
    present = tmp_path / "present.txt"
    present.write_text("abc")

    hashes = hash_existing_artifacts(
        {"present": "present.txt", "missing": "missing.txt"},
        root=tmp_path,
    )

    assert hashes == {"present": sha256_file(present)}
