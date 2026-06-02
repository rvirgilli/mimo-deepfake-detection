from pathlib import Path

import pytest

from mimodf.config import ConfigError, ExperimentConfig, load_experiment_config


def valid_config_dict():
    return {
        "model": "wav2vec2",
        "strategy": "adapter",
        "seed": 42,
        "protocol": {
            "train_set": "ASVspoof2019_LA_train",
            "validation_set": "ASVspoof2021_LA_fast_eval_subset",
            "checkpoint_selection_set": "ASVspoof2021_LA_fast_eval_subset",
            "eval_set": "ASVspoof2021_LA_eval_and_DF_eval",
        },
        "optimizer": {
            "name": "adam",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "encoder_lr": None,
        },
        "scorer": {"la": "official_asvspoof2021_la", "df": "eer_only"},
    }


def test_publish_config_loads():
    cfg = load_experiment_config(Path("configs/publish/wav2vec2_adapter.yaml"))

    assert cfg.model == "wav2vec2"
    assert cfg.strategy == "adapter"
    assert cfg.optimizer.name == "adam"
    assert cfg.protocol.checkpoint_selection_set == "ASVspoof2021_LA_fast_eval_subset"


def test_implicit_validation_set_fails():
    data = valid_config_dict()
    data["protocol"]["validation_set"] = "implicit"

    with pytest.raises(ConfigError, match="protocol.validation_set must be explicit"):
        ExperimentConfig.from_dict(data)


def test_missing_scorer_fails():
    data = valid_config_dict()
    del data["scorer"]

    with pytest.raises(ConfigError, match="scorer"):
        ExperimentConfig.from_dict(data)


def test_adamw_claim_with_null_encoder_lr_fails():
    data = valid_config_dict()
    data["optimizer"]["name"] = "adamw"
    data["optimizer"]["encoder_lr"] = None

    with pytest.raises(ConfigError, match="AdamW requires explicit"):
        ExperimentConfig.from_dict(data)


def test_adam_with_null_encoder_lr_is_valid_legacy_fact():
    cfg = ExperimentConfig.from_dict(valid_config_dict())

    assert cfg.optimizer.name == "adam"
    assert cfg.optimizer.encoder_lr is None


def test_publish_configs_are_complete_and_portable():
    paths = sorted(path.name for path in Path("configs/publish").glob("*.yaml"))
    assert paths == [
        "mimo_adapter_exploratory.yaml",
        "mimo_frozen.yaml",
        "mimo_full.yaml",
        "wav2vec2_adapter.yaml",
        "wav2vec2_frozen.yaml",
        "wav2vec2_full_external.yaml",
    ]

    for path in Path("configs/publish").glob("*.yaml"):
        text = path.read_text()
        assert "/home/ufg" not in text
        load_experiment_config(path)
