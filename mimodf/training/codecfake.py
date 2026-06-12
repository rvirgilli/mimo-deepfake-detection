"""Controlled planning utilities for CodecFake+ XLS-R runs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimodf.data.codecfake_splits import build_source_holdout_rows

MODEL_CONDITIONS: dict[str, dict[str, str]] = {
    "xlsr_frozen_backend": {
        "frontend": "frontend:wav2vec2-xlsr-300m/v1",
        "adaptation": "adaptation:frozen/v1",
        "backend": "backend:aasist/v1",
        "trainable_scope": "backend_only",
    },
    "xlsr_peft_adapter": {
        "frontend": "frontend:wav2vec2-xlsr-300m/v1",
        "adaptation": "adaptation:houlsby-adapter-last8/v1",
        "backend": "backend:aasist/v1",
        "trainable_scope": "last_8_layer_adapters_and_backend",
    },
    "xlsr_full_finetune": {
        "frontend": "frontend:wav2vec2-xlsr-300m/v1",
        "adaptation": "adaptation:full-finetune/v1",
        "backend": "backend:aasist/v1",
        "trainable_scope": "frontend_and_backend",
    },
    "wavlm_frozen_backend": {
        "frontend": "frontend:wavlm-base-plus/hf-b211941/v1",
        "adaptation": "adaptation:frozen/v1",
        "backend": "backend:aasist/v1",
        "trainable_scope": "backend_only",
    },
}
CLASSES = ["bonafide", "spoof"]


@dataclass(frozen=True)
class CodecfakeXlsrPlanSettings:
    split_plan: Path
    protocol: Path
    fold: str
    condition: str
    seed: int
    out_dir: Path
    require_audio: bool = True


@dataclass(frozen=True)
class CodecfakeXlsrModelSmokeSettings:
    plan: CodecfakeXlsrPlanSettings
    checkpoint_path: Path
    batch_size: int = 1
    eval_batch_size: int = 1
    cut: int | None = 64_600
    num_workers: int = 0
    device: str = "cpu"
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    deterministic: bool = False


@dataclass(frozen=True)
class CodecfakeXlsrTrainSettings:
    plan: CodecfakeXlsrPlanSettings
    checkpoint_path: Path
    epochs: int
    batch_size: int = 1
    eval_batch_size: int = 1
    cut: int | None = 64_600
    num_workers: int = 0
    device: str = "cpu"
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None
    save_checkpoints: bool = False
    checkpoint_metric: str = "val_loss"
    deterministic: bool = False


@dataclass(frozen=True)
class CodecfakeXlsrDryRunPlan:
    settings: CodecfakeXlsrPlanSettings
    split_plan_metadata: dict[str, Any]
    condition_metadata: dict[str, str]
    counts: dict[str, Any]
    output_paths: dict[str, str]
    implementation_status: str = "dry_run_only_no_training"

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_plan": str(self.settings.split_plan),
            "protocol": str(self.settings.protocol),
            "fold": self.settings.fold,
            "condition": self.settings.condition,
            "seed": self.settings.seed,
            "out_dir": str(self.settings.out_dir),
            "require_audio": self.settings.require_audio,
            "split_plan_metadata": self.split_plan_metadata,
            "condition_metadata": self.condition_metadata,
            "counts": self.counts,
            "output_paths": self.output_paths,
            "implementation_status": self.implementation_status,
            "caveats": [
                "dry-run planning only; does not import Torch, load XLS-R, train, score, or write checkpoints",
                "CoSG source-holdout is a custom diagnostic protocol, not the official CoRS train/validation/eval split",
            ],
        }


def build_codecfake_xlsr_dry_run_plan(
    settings: CodecfakeXlsrPlanSettings,
) -> CodecfakeXlsrDryRunPlan:
    _validate_plan_settings(settings)
    split_plan = json.loads(settings.split_plan.read_text())
    rows = _build_split_rows(settings, split_plan)
    counts = {
        "train": _partition_counts(rows.train_rows),
        "validation": _partition_counts(rows.validation_rows),
        "test": _partition_counts(rows.test_rows),
    }
    run_dir = settings.out_dir / settings.condition / f"seed_{settings.seed}" / settings.fold
    output_paths = {
        "run_dir": str(run_dir),
        "resolved_spec": str(run_dir / "resolved_spec.yaml"),
        "manifest": str(run_dir / "manifest.json"),
        "train_history": str(run_dir / "train_history.jsonl"),
        "best_checkpoint": str(run_dir / "checkpoints/best.pt"),
        "scores": str(run_dir / "scores.jsonl"),
        "metrics": str(run_dir / "metrics.json"),
        "report": str(run_dir / "report.md"),
    }
    return CodecfakeXlsrDryRunPlan(
        settings=settings,
        split_plan_metadata={
            "subset": split_plan.get("subset"),
            "validation_policy": split_plan.get("validation_policy"),
            "validation_fraction": split_plan.get("validation_fraction"),
            "eligible_sources": split_plan.get("eligible_sources"),
            "total_records": split_plan.get("total_records"),
        },
        condition_metadata=MODEL_CONDITIONS[settings.condition],
        counts=counts,
        output_paths=output_paths,
    )


def check_codecfake_xlsr_loaders(
    settings: CodecfakeXlsrPlanSettings,
    *,
    batch_size: int,
    eval_batch_size: int,
    cut: int | None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Build loaders and inspect one batch from each partition without training."""

    loaders = _build_loaders(settings, batch_size, eval_batch_size, cut, num_workers)
    return {
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "cut": cut,
        "num_workers": num_workers,
        "train": _batch_summary(next(iter(loaders.train_loader))),
        "validation": _batch_summary(next(iter(loaders.val_loader))),
        "test": _batch_summary(next(iter(loaders.test_loader))),
    }


def _build_loaders(
    settings: CodecfakeXlsrPlanSettings,
    batch_size: int,
    eval_batch_size: int,
    cut: int | None,
    num_workers: int,
    *,
    seed: int | None = None,
) -> Any:
    from mimodf.data.codecfake_loader import CodecfakeLoaderSettings, build_codecfake_loaders

    split_plan = json.loads(settings.split_plan.read_text())
    return build_codecfake_loaders(
        _build_split_rows(settings, split_plan),
        CodecfakeLoaderSettings(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            cut=cut,
            seed=seed,
        ),
    )


def _build_split_rows(settings: CodecfakeXlsrPlanSettings, split_plan: dict[str, Any]) -> Any:
    fold_names = {fold["heldout_source"] for fold in split_plan.get("folds", [])}
    if settings.fold not in fold_names:
        raise ValueError(f"fold {settings.fold!r} not found in split plan")
    return build_source_holdout_rows(
        protocol=settings.protocol,
        heldout_source=settings.fold,
        subset=split_plan.get("subset", "CoSG"),
        validation_policy=split_plan.get("validation_policy", "source"),
        validation_source_count=int(split_plan.get("validation_source_count", 1)),
        validation_fraction=float(split_plan.get("validation_fraction", 0.15)),
        seed=int(split_plan.get("seed", settings.seed)),
        require_audio=settings.require_audio,
    )


def _build_xlsr_frontend(condition: str, checkpoint_path: Path) -> Any:
    from src.frontends.wav2vec2 import Wav2Vec2Frontend

    if condition == "wavlm_frozen_backend":
        from src.frontends.wavlm import WavLMFrontend

        return WavLMFrontend(local_files_only=False)
    if condition == "xlsr_frozen_backend":
        return Wav2Vec2Frontend(
            checkpoint_path=str(checkpoint_path),
            freeze=True,
            finetune_config={"strategy": "frozen"},
        )
    if condition == "xlsr_peft_adapter":
        return Wav2Vec2Frontend(
            checkpoint_path=str(checkpoint_path),
            freeze=False,
            finetune_config={
                "strategy": "adapter",
                "adapter": {"dim": 64, "dropout": 0.1, "layers": "last_n", "n_layers": 8},
            },
        )
    if condition == "xlsr_full_finetune":
        frontend = Wav2Vec2Frontend(
            checkpoint_path=str(checkpoint_path),
            freeze=False,
            finetune_config={"strategy": "full"},
        )
        frontend.unfreeze()
        return frontend
    raise ValueError(f"unknown condition: {condition}")


def _batch_summary(batch: tuple[Any, Any, dict[str, Any]]) -> dict[str, Any]:
    inputs, targets, metadata = batch
    return {
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "target_values": sorted({int(value) for value in targets.tolist()}),
        "metadata_keys": sorted(metadata),
    }


def _partition_counts(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    labels = Counter(str(row["label"]) for row in rows)
    sources = Counter(str(row.get("source_model")) for row in rows)
    missing_audio = sum(
        1
        for row in rows
        if not isinstance(row.get("audio_path"), str) or not Path(str(row["audio_path"])).is_file()
    )
    return {
        "records": len(rows),
        "labels": dict(sorted(labels.items())),
        "sources": dict(sorted(sources.items())),
        "missing_audio": missing_audio,
    }


def _validate_plan_settings(settings: CodecfakeXlsrPlanSettings) -> None:
    if settings.condition not in MODEL_CONDITIONS:
        raise ValueError(f"unknown condition: {settings.condition}")
    if settings.seed < 0:
        raise ValueError("seed must be non-negative")
