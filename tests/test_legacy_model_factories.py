import sys
import types

from mimodf.training.legacy_model import (
    LegacyFrontendSettings,
    LegacyModelSettings,
    build_legacy_frontend,
    build_legacy_model,
)


def test_build_legacy_frontend_passes_explicit_settings(monkeypatch):
    calls = []
    frontends = types.ModuleType("src.frontends")

    def get_frontend(name, **kwargs):
        calls.append((name, kwargs))
        return "frontend"

    frontends.get_frontend = get_frontend
    monkeypatch.setitem(sys.modules, "src.frontends", frontends)

    frontend = build_legacy_frontend(
        LegacyFrontendSettings(
            name="mimo",
            model_path="models/MiMo-Audio-Tokenizer",
            freeze=True,
            feature_type="continuous",
            finetune_config={"strategy": "adapter"},
        )
    )

    assert frontend == "frontend"
    assert calls == [
        (
            "mimo",
            {
                "checkpoint": None,
                "model_path": "models/MiMo-Audio-Tokenizer",
                "model_name": None,
                "freeze": True,
                "use_bfloat16": True,
                "upsample_to_50hz": False,
                "upsample_mode": "linear",
                "native_50hz": False,
                "feature_type": "continuous",
                "feature_config": None,
                "finetune_config": {"strategy": "adapter"},
            },
        )
    ]


def test_build_legacy_model_passes_explicit_settings(monkeypatch):
    calls = []
    model_module = types.ModuleType("src.model")

    class FakeModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    model_module.Model = FakeModel
    monkeypatch.setitem(sys.modules, "src.model", model_module)

    model = build_legacy_model(
        frontend="frontend",
        settings=LegacyModelSettings(dropout=0.25, projection_type="mlp"),
    )

    assert isinstance(model, FakeModel)
    assert calls[0]["frontend"] == "frontend"
    assert calls[0]["dropout"] == 0.25
    assert calls[0]["projection_type"] == "mlp"
    assert calls[0]["gat_dims"] == [64, 32]
