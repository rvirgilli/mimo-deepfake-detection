import json

import pytest

torch = pytest.importorskip("torch")

from mimodf.config import load_experiment_config
from mimodf.training.loop import TrainLoopSettings, evaluate_loss, train_one_epoch, train_one_run


def make_loader():
    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    targets = torch.tensor([0, 1, 1, 0])
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)


def make_loader_with_ids():
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.tensor([float(index), 1.0]), torch.tensor(index % 2), f"utt_{index}"

    return torch.utils.data.DataLoader(Dataset(), batch_size=2, shuffle=False)


def test_train_one_run_writes_checkpoint_and_completed_manifest(tmp_path):
    cfg = load_experiment_config("configs/publish/wav2vec2_adapter.yaml")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = train_one_run(
        config=cfg,
        model=model,
        train_loader=make_loader(),
        val_loader=make_loader(),
        optimizer=optimizer,
        output_dir=tmp_path,
        settings=TrainLoopSettings(epochs=2, top_k_checkpoints=1),
    )

    assert result.best_checkpoint.is_file()
    assert result.manifest_path.is_file()
    assert result.epochs_completed == 2
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["metrics"]["epochs_completed"] == 2
    assert manifest["metrics"]["checkpoint_metric"] == "val_loss"
    assert manifest["artifacts"]["best_checkpoint"] == str(result.best_checkpoint)
    assert manifest["artifact_hashes"]["best_checkpoint"]
    assert manifest["config"]["protocol"]["eval_set"] == "ASVspoof2021_LA_eval_and_DF_eval"
    assert len(list((tmp_path / "checkpoints").glob("*.pth"))) == 1


def test_train_one_run_records_failed_manifest(tmp_path):
    cfg = load_experiment_config("configs/publish/wav2vec2_adapter.yaml")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="val_loader produced no items"):
        train_one_run(
            config=cfg,
            model=model,
            train_loader=make_loader(),
            val_loader=[],
            optimizer=optimizer,
            output_dir=tmp_path,
            settings=TrainLoopSettings(epochs=1),
        )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "val_loader produced no items" in manifest["error"]


def test_train_epoch_and_eval_accept_batch_limits_and_extra_ids():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_loss = train_one_epoch(
        model=model,
        loader=make_loader(),
        optimizer=optimizer,
        device="cpu",
        max_batches=1,
    )
    val_loss = evaluate_loss(
        model=model,
        loader=make_loader_with_ids(),
        device="cpu",
        max_batches=1,
    )

    assert train_loss > 0
    assert val_loss > 0


def test_train_loop_settings_validate_supported_contract():
    with pytest.raises(ValueError, match="epochs"):
        TrainLoopSettings(epochs=0).validate()
    with pytest.raises(ValueError, match="checkpoint_metric"):
        TrainLoopSettings(epochs=1, checkpoint_metric="eer").validate()
    with pytest.raises(ValueError, match="top_k"):
        TrainLoopSettings(epochs=1, top_k_checkpoints=0).validate()
    with pytest.raises(ValueError, match="max_train_batches"):
        TrainLoopSettings(epochs=1, max_train_batches=0).validate()
    with pytest.raises(ValueError, match="max_val_batches"):
        TrainLoopSettings(epochs=1, max_val_batches=0).validate()
