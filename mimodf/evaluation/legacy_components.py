"""Legacy ASVspoof evaluation component construction.

Imports Torch and legacy modules only inside functions. This keeps CLI planning
and lightweight tests free of model dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mimodf.config import ExperimentConfig, load_experiment_config
from mimodf.evaluation.run import EvaluationComponents
from mimodf.scoring.evaluate import EvaluationBatch, EvaluationItem
from mimodf.scoring.torch_eval import TorchBatchPredictor
from mimodf.training.legacy_model import (
    LegacyFrontendSettings,
    LegacyModelSettings,
    build_legacy_frontend,
    build_legacy_model,
)


@dataclass(frozen=True)
class LegacyEvaluationSettings:
    config_path: str
    checkpoint: str
    eval_root: str
    protocols_path: str
    track: str
    frontend: str
    legacy_run_config: str | None = None
    frontend_checkpoint: str | None = None
    frontend_model_path: str | None = None
    frontend_model_name: str | None = None
    freeze_frontend: bool | None = None
    feature_type: str = "continuous"
    sample_rate: int = 16000
    cut: int | None = 64600
    batch_size: int = 14
    num_workers: int = 4
    device: str = "cpu"
    max_items: int | None = None


def build_legacy_evaluation_components(settings: LegacyEvaluationSettings) -> EvaluationComponents:
    if settings.track not in {"LA", "DF"}:
        raise ValueError("track must be 'LA' or 'DF'")
    if settings.max_items is not None and settings.max_items <= 0:
        raise ValueError("max_items must be positive")

    torch = _import_torch()
    data_utils = _import_legacy_data_utils()
    cfg = load_experiment_config(settings.config_path)

    trial_file = _eval_trial_file(Path(settings.protocols_path), settings.track)
    if not trial_file.is_file():
        raise FileNotFoundError(trial_file)
    utterance_ids = list(data_utils.genSpoof_list(str(trial_file), is_eval=True))
    if settings.max_items is not None:
        utterance_ids = utterance_ids[: settings.max_items]

    dataset = data_utils.Dataset_ASVspoof2021_eval(
        list_IDs=utterance_ids,
        base_dir=_dir_with_slash(Path(settings.eval_root)),
        sample_rate=settings.sample_rate,
        cut=settings.cut,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        shuffle=False,
        drop_last=False,
    )

    model = _load_legacy_model(settings, cfg, torch)
    predictor = TorchBatchPredictor(
        model=model,
        score_fn=lambda output: output[:, 1],
        device=settings.device,
    )
    return EvaluationComponents(
        batches=_evaluation_batches(loader),
        predict_batch=predictor,
    )


def _load_legacy_model(
    settings: LegacyEvaluationSettings,
    cfg: ExperimentConfig,
    torch: Any,
) -> Any:
    legacy_cfg = _load_legacy_run_config(settings.legacy_run_config)
    frontend_settings = LegacyFrontendSettings(
        name=settings.frontend,
        checkpoint=settings.frontend_checkpoint
        or _legacy_get(legacy_cfg, "frontend", "checkpoint"),
        model_path=settings.frontend_model_path
        or _legacy_get(legacy_cfg, "frontend", "model_path"),
        model_name=settings.frontend_model_name
        or _legacy_get(legacy_cfg, "frontend", "model_name"),
        freeze=_freeze_frontend(settings, cfg, legacy_cfg),
        use_bfloat16=bool(
            _legacy_get(legacy_cfg, "frontend", "use_bfloat16")
            if _legacy_get(legacy_cfg, "frontend", "use_bfloat16") is not None
            else True
        ),
        upsample_to_50hz=bool(_legacy_get(legacy_cfg, "frontend", "upsample_to_50hz") or False),
        upsample_mode=str(_legacy_get(legacy_cfg, "frontend", "upsample_mode") or "linear"),
        native_50hz=bool(_legacy_get(legacy_cfg, "frontend", "native_50hz") or False),
        feature_type=_legacy_feature_type(settings, legacy_cfg),
        feature_config=_legacy_feature_config(legacy_cfg),
        finetune_config=_finetune_config(settings, cfg, legacy_cfg),
    )
    frontend = build_legacy_frontend(frontend_settings)
    model = build_legacy_model(frontend, _legacy_model_settings(legacy_cfg))
    checkpoint = torch.load(settings.checkpoint, map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)
    state_dict = {
        key.replace("ssl_model.", "frontend."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    return model


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a state dict")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint model_state_dict must be a mapping")
        return state_dict
    return checkpoint


def _freeze_frontend(
    settings: LegacyEvaluationSettings,
    cfg: ExperimentConfig,
    legacy_cfg: dict[str, Any],
) -> bool | None:
    if settings.freeze_frontend is not None:
        return settings.freeze_frontend
    legacy_freeze = _legacy_get(legacy_cfg, "frontend", "freeze")
    if legacy_freeze is not None:
        return bool(legacy_freeze)
    if cfg.strategy == "frozen":
        return True
    if cfg.strategy in {"full", "full_ft", "full_external"}:
        return False
    if cfg.strategy == "adapter":
        return False
    return None


def _legacy_feature_type(settings: LegacyEvaluationSettings, legacy_cfg: dict[str, Any]) -> str:
    legacy_type = _legacy_get(legacy_cfg, "frontend", "feature", "type")
    return str(legacy_type or settings.feature_type)


def _legacy_feature_config(legacy_cfg: dict[str, Any]) -> dict[str, Any] | None:
    feature = _legacy_get(legacy_cfg, "frontend", "feature")
    if not isinstance(feature, dict):
        return None
    return {key: value for key, value in feature.items() if key != "type"}


def _finetune_config(
    settings: LegacyEvaluationSettings,
    cfg: ExperimentConfig,
    legacy_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    legacy_finetune = _legacy_get(legacy_cfg, "frontend", "finetune")
    if isinstance(legacy_finetune, dict):
        return legacy_finetune
    if cfg.strategy == "adapter" and settings.frontend == "wav2vec2":
        return {
            "strategy": "adapter",
            "adapter": {"dim": 64, "dropout": 0.1, "layers": "last_n", "n_layers": 8},
        }
    return None


def _legacy_model_settings(legacy_cfg: dict[str, Any]) -> LegacyModelSettings:
    model_cfg = legacy_cfg.get("model")
    if not isinstance(model_cfg, dict):
        return LegacyModelSettings()
    projection = (
        model_cfg.get("projection") if isinstance(model_cfg.get("projection"), dict) else {}
    )
    return LegacyModelSettings(
        filts_0=int(model_cfg.get("filts_0", 128)),
        encoder_scale=float(model_cfg.get("encoder_scale", 1.0)),
        gat_dims=list(model_cfg.get("gat_dims", [64, 32])),
        pool_ratios=list(model_cfg.get("pool_ratios", [0.5, 0.5, 0.5, 0.5])),
        temperatures=list(model_cfg.get("temperatures", [2.0, 2.0, 100.0, 100.0])),
        dropout=float(model_cfg.get("dropout", 0.5)),
        dropout_way=float(model_cfg.get("dropout_way", 0.2)),
        projection_type=str(projection.get("type", "linear")),
        projection_hidden_dims=projection.get("hidden_dims", [512, 256]),
        projection_activation=str(projection.get("activation", "gelu")),
        projection_dropout=float(projection.get("dropout", 0.1)),
        projection_use_batchnorm=bool(projection.get("use_batchnorm", True)),
    )


def _load_legacy_run_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("legacy run config must be a mapping")
    return data


def _legacy_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _evaluation_batches(loader: Any) -> Iterable[EvaluationBatch[Any]]:
    for waveforms, utterance_ids in loader:
        items = [
            EvaluationItem(str(utterance_id), waveform)
            for utterance_id, waveform in zip(utterance_ids, waveforms, strict=True)
        ]
        yield EvaluationBatch.from_items(items)


def _eval_trial_file(protocols_path: Path, track: str) -> Path:
    return (
        protocols_path / f"ASVspoof_{track}_cm_protocols" / f"ASVspoof2021.{track}.cm.eval.trl.txt"
    )


def _dir_with_slash(path: Path) -> str:
    return str(path) + "/"


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Torch is required for legacy evaluation") from exc
    return torch


def _import_legacy_data_utils() -> Any:
    from src import data_utils

    return data_utils
