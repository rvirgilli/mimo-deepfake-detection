"""
Frontend feature extractors for audio deepfake detection.

Available frontends:
- Wav2Vec2Frontend: wav2vec 2.0 XLSR (requires fairseq, Python 3.10)
- MiMoFrontend: MiMo-Audio-Tokenizer (requires flash-attn, Python 3.12)
- EnCodecFrontend: Meta's EnCodec neural codec (reconstruction-based)
- HuBERTFrontend: HuBERT-Large (masked prediction pretraining)

Usage:
    from src.frontends import get_frontend

    frontend = get_frontend("mimo", model_path="./models/MiMo-Audio-Tokenizer")
    features = frontend.extract_feat(waveform)
"""

from .base import BaseFrontend

# Lazy imports to handle environment-specific dependencies
_FRONTEND_REGISTRY = {}


def register_frontend(name: str):
    """Decorator to register a frontend class"""
    def decorator(cls):
        _FRONTEND_REGISTRY[name] = cls
        return cls
    return decorator


def get_frontend(name: str, **kwargs) -> BaseFrontend:
    """
    Factory function to get a frontend by name.

    Args:
        name: Frontend name ("wav2vec2" or "mimo")
        **kwargs: Frontend-specific arguments (mapped to constructor args)
            - checkpoint -> checkpoint_path (wav2vec2)
            - model_path -> model_path (mimo)
            - freeze -> freeze (wav2vec2 only, mimo uses finetune_config)
            - use_bfloat16 -> use_bfloat16 (mimo)
            - upsample_to_50hz -> upsample_to_50hz (mimo)
            - upsample_mode -> upsample_mode (mimo: linear, nearest, learnable)
            - native_50hz -> native_50hz (mimo: extract at 50Hz, skip final downsample)
            - feature_type -> feature_type (mimo: continuous, rvq_sum, etc.)
            - feature_config -> feature_config (mimo: strategy-specific params)
            - finetune_config -> finetune_config (mimo, wav2vec2)

    Returns:
        Initialized frontend instance
    """
    # Filter out None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if name == "wav2vec2":
        from .wav2vec2 import Wav2Vec2Frontend
        # Map config keys to constructor args
        init_kwargs = {}
        if "checkpoint" in kwargs:
            init_kwargs["checkpoint_path"] = kwargs["checkpoint"]
        if "freeze" in kwargs:
            init_kwargs["freeze"] = kwargs["freeze"]
        if "finetune_config" in kwargs:
            init_kwargs["finetune_config"] = kwargs["finetune_config"]
        return Wav2Vec2Frontend(**init_kwargs)
    elif name == "encodec":
        from .encodec import EnCodecFrontend
        init_kwargs = {}
        if "model_name" in kwargs:
            init_kwargs["model_name"] = kwargs["model_name"]
        if "freeze" in kwargs:
            init_kwargs["freeze"] = kwargs["freeze"]
        return EnCodecFrontend(**init_kwargs)
    elif name == "mimo":
        from .mimo import MiMoFrontend
        # Map config keys to constructor args
        init_kwargs = {}
        if "model_path" in kwargs:
            init_kwargs["model_path"] = kwargs["model_path"]
        if "use_bfloat16" in kwargs:
            init_kwargs["use_bfloat16"] = kwargs["use_bfloat16"]
        if "upsample_to_50hz" in kwargs:
            init_kwargs["upsample_to_50hz"] = kwargs["upsample_to_50hz"]
        if "upsample_mode" in kwargs:
            init_kwargs["upsample_mode"] = kwargs["upsample_mode"]
        if "native_50hz" in kwargs:
            init_kwargs["native_50hz"] = kwargs["native_50hz"]
        if "feature_type" in kwargs:
            init_kwargs["feature_type"] = kwargs["feature_type"]
        if "feature_config" in kwargs:
            init_kwargs["feature_config"] = kwargs["feature_config"]
        if "finetune_config" in kwargs:
            init_kwargs["finetune_config"] = kwargs["finetune_config"]
        return MiMoFrontend(**init_kwargs)
    elif name == "hubert":
        from .hubert import HuBERTFrontend
        init_kwargs = {}
        if "model_name" in kwargs:
            init_kwargs["model_name"] = kwargs["model_name"]
        if "freeze" in kwargs:
            init_kwargs["freeze"] = kwargs["freeze"]
        if "finetune_config" in kwargs:
            init_kwargs["finetune_config"] = kwargs["finetune_config"]
        return HuBERTFrontend(**init_kwargs)
    else:
        raise ValueError(f"Unknown frontend: {name}. Available: wav2vec2, mimo, hubert")


__all__ = ["BaseFrontend", "get_frontend"]
