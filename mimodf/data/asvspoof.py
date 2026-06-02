"""ASVspoof data-loader planning and construction.

The important design choice: validation protocol is explicit. We do not inspect
file existence and silently switch between ASVspoof2021-fast and ASVspoof2019-dev.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

ValidationProtocol = Literal["asvspoof2021_fast", "asvspoof2019_dev"]


@dataclass(frozen=True)
class ASVspoofDataSettings:
    database_path: Path
    protocols_path: Path
    track: str
    batch_size: int
    eval_batch_size: int
    num_workers: int
    sample_rate: int
    cut: int | None
    rawboost_algo: int
    rawboost_args: dict[str, Any]
    validation_protocol: ValidationProtocol
    auto_scale_rawboost: bool = True
    drop_last_train: bool = True


@dataclass(frozen=True)
class ASVspoofPathPlan:
    train_protocol: Path
    train_audio_dir: Path
    validation_protocol: ValidationProtocol
    validation_trial_file: Path
    validation_key_file: Path | None
    validation_audio_dir: Path


@dataclass(frozen=True)
class ASVspoofLoaders:
    train_loader: Any
    val_loader: Any
    validation_labels: dict[str, str] | None
    plan: ASVspoofPathPlan


def build_path_plan(settings: ASVspoofDataSettings) -> ASVspoofPathPlan:
    _validate_track(settings.track)
    prefix = f"ASVspoof_{settings.track}"
    prefix_2019 = f"ASVspoof2019.{settings.track}"
    protocols_dir = settings.protocols_path / f"{prefix}_cm_protocols"

    train_protocol = protocols_dir / f"{prefix_2019}.cm.train.trn.txt"
    train_audio_dir = settings.database_path / f"ASVspoof2019_{settings.track}_train"

    if settings.validation_protocol == "asvspoof2021_fast":
        return ASVspoofPathPlan(
            train_protocol=train_protocol,
            train_audio_dir=train_audio_dir,
            validation_protocol=settings.validation_protocol,
            validation_trial_file=protocols_dir
            / f"ASVspoof2021.{settings.track}.cm.eval.fast.trl.txt",
            validation_key_file=protocols_dir
            / f"ASVspoof2021.{settings.track}.cm.eval.fast.key.txt",
            validation_audio_dir=settings.database_path / f"ASVspoof2021_{settings.track}_eval",
        )

    if settings.validation_protocol == "asvspoof2019_dev":
        return ASVspoofPathPlan(
            train_protocol=train_protocol,
            train_audio_dir=train_audio_dir,
            validation_protocol=settings.validation_protocol,
            validation_trial_file=protocols_dir / f"{prefix_2019}.cm.dev.trl.txt",
            validation_key_file=None,
            validation_audio_dir=settings.database_path / f"ASVspoof2019_{settings.track}_dev",
        )

    raise ValueError(f"unsupported validation_protocol: {settings.validation_protocol}")


def build_asvspoof_loaders(settings: ASVspoofDataSettings) -> ASVspoofLoaders:
    """Build train/validation loaders using legacy dataset classes."""

    torch = _import_torch()
    data_utils = _import_legacy_data_utils()
    plan = build_path_plan(settings)
    _require_file(plan.train_protocol)
    _require_file(plan.validation_trial_file)
    if plan.validation_key_file is not None:
        _require_file(plan.validation_key_file)

    rawboost_args = SimpleNamespace(**settings.rawboost_args)
    train_labels, train_ids = data_utils.genSpoof_list(str(plan.train_protocol), is_train=True)
    train_set = data_utils.Dataset_ASVspoof2019_train(
        args=rawboost_args,
        list_IDs=train_ids,
        labels=train_labels,
        base_dir=_dir_with_slash(plan.train_audio_dir),
        algo=settings.rawboost_algo,
        sample_rate=settings.sample_rate,
        cut=settings.cut,
        auto_scale_rawboost=settings.auto_scale_rawboost,
    )

    validation_labels = None
    if settings.validation_protocol == "asvspoof2021_fast":
        val_ids = data_utils.genSpoof_list(str(plan.validation_trial_file), is_eval=True)
        validation_labels = load_fast_eval_labels(plan.validation_key_file)
        val_set = data_utils.Dataset_ASVspoof2021_fast_eval(
            list_IDs=val_ids,
            labels=validation_labels,
            base_dir=_dir_with_slash(plan.validation_audio_dir),
            sample_rate=settings.sample_rate,
            cut=settings.cut,
        )
    else:
        val_labels, val_ids = data_utils.genSpoof_list(str(plan.validation_trial_file))
        val_set = data_utils.Dataset_ASVspoof2019_train(
            args=rawboost_args,
            list_IDs=val_ids,
            labels=val_labels,
            base_dir=_dir_with_slash(plan.validation_audio_dir),
            algo=settings.rawboost_algo,
            sample_rate=settings.sample_rate,
            cut=settings.cut,
            auto_scale_rawboost=settings.auto_scale_rawboost,
        )

    return ASVspoofLoaders(
        train_loader=torch.utils.data.DataLoader(
            train_set,
            batch_size=settings.batch_size,
            num_workers=settings.num_workers,
            shuffle=True,
            drop_last=settings.drop_last_train,
        ),
        val_loader=torch.utils.data.DataLoader(
            val_set,
            batch_size=settings.eval_batch_size,
            num_workers=settings.num_workers,
            shuffle=False,
        ),
        validation_labels=validation_labels,
        plan=plan,
    )


def load_fast_eval_labels(key_path: str | Path | None) -> dict[str, str]:
    if key_path is None:
        raise ValueError("key_path is required for ASVspoof2021 fast validation")
    labels: dict[str, str] = {}
    with Path(key_path).open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels[parts[1]] = parts[4]
    return labels


def _validate_track(track: str) -> None:
    if track not in {"LA", "DF"}:
        raise ValueError("track must be 'LA' or 'DF'")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def _dir_with_slash(path: Path) -> str:
    return str(path) + "/"


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Torch is required to build ASVspoof loaders") from exc
    return torch


def _import_legacy_data_utils() -> Any:
    from src import data_utils

    return data_utils
