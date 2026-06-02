import pytest

torch = pytest.importorskip("torch")

from mimodf.config import ExperimentConfig, load_experiment_config
from mimodf.training.components import (
    TrainingComponents,
    build_optimizer,
    split_frontend_backend_params,
    train_with_components,
)
from mimodf.training.loop import TrainLoopSettings


class FrontendBackendModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = torch.nn.Linear(2, 2)
        self.backend = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.backend(self.frontend(x))


def make_loader():
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets), batch_size=2
    )


def adamw_config():
    data = load_experiment_config("configs/publish/mimo_full.yaml").to_dict()
    data["optimizer"] = {
        "name": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "encoder_lr": 1e-6,
    }
    return ExperimentConfig.from_dict(data)


def test_build_optimizer_uses_adam_for_null_encoder_lr():
    cfg = load_experiment_config("configs/publish/mimo_full.yaml")
    model = FrontendBackendModel()

    optimizer = build_optimizer(cfg, model)

    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(cfg.optimizer.lr)


def test_split_frontend_backend_params_uses_legacy_prefix():
    model = FrontendBackendModel()

    encoder, backend = split_frontend_backend_params(model)

    assert len(encoder) == 2
    assert len(backend) == 2
    assert sum(param.numel() for param in encoder) == 6
    assert sum(param.numel() for param in backend) == 6


def test_build_optimizer_uses_explicit_adamw_groups():
    cfg = adamw_config()
    model = FrontendBackendModel()
    encoder, backend = split_frontend_backend_params(model)

    optimizer = build_optimizer(cfg, model, encoder_params=encoder, backend_params=backend)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["name"] for group in optimizer.param_groups] == ["encoder", "backend"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-4)


def test_build_optimizer_requires_adamw_groups():
    cfg = adamw_config()
    model = FrontendBackendModel()

    with pytest.raises(ValueError, match="encoder_params"):
        build_optimizer(cfg, model)


def test_train_with_components_delegates_to_train_one_run(tmp_path):
    cfg = load_experiment_config("configs/publish/wav2vec2_adapter.yaml")
    model = FrontendBackendModel()
    components = TrainingComponents(
        model=model,
        train_loader=make_loader(),
        val_loader=make_loader(),
        optimizer=build_optimizer(cfg, model),
    )

    result = train_with_components(
        config=cfg,
        components=components,
        output_dir=tmp_path,
        settings=TrainLoopSettings(epochs=1),
    )

    assert result.best_checkpoint.is_file()
    assert result.manifest_path.is_file()
