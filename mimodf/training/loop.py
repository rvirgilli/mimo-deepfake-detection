"""Minimal Torch training loop for future controlled runs.

This is deliberately small. It proves the infrastructure contract around
seeding, checkpointing, and manifests before we connect ASVspoof/MiMo specifics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimodf.config import ExperimentConfig
from mimodf.training.checkpoint import TopKCheckpointTracker
from mimodf.training.manifest import TrainingManifest
from mimodf.training.seeding import SeedSettings, seed_everything


@dataclass(frozen=True)
class TrainLoopSettings:
    epochs: int
    checkpoint_metric: str = "val_loss"
    top_k_checkpoints: int = 1
    device: str = "cpu"
    max_grad_norm: float = 0.0
    max_train_batches: int | None = None
    max_val_batches: int | None = None

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.checkpoint_metric != "val_loss":
            raise ValueError(
                "only checkpoint_metric='val_loss' is supported in the foundation loop"
            )
        if self.top_k_checkpoints < 1:
            raise ValueError("top_k_checkpoints must be >= 1")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be >= 1 when set")
        if self.max_val_batches is not None and self.max_val_batches < 1:
            raise ValueError("max_val_batches must be >= 1 when set")


@dataclass(frozen=True)
class TrainingRunResult:
    best_checkpoint: Path
    manifest_path: Path
    best_val_loss: float
    final_train_loss: float
    epochs_completed: int


def train_one_run(
    *,
    config: ExperimentConfig,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: Any,
    output_dir: str | Path,
    settings: TrainLoopSettings,
) -> TrainingRunResult:
    """Train a model with explicit loaders/optimizer and write manifest.

    The caller owns model/data construction. This function owns the generic
    lifecycle: seed, train/eval, top-k checkpointing, manifest completion.
    """

    torch = _import_torch()
    settings.validate()
    seed_everything(SeedSettings(seed=config.seed))

    output = Path(output_dir)
    checkpoints_dir = output / "checkpoints"
    manifest_path = output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)

    manifest = TrainingManifest.start(
        config,
        command=["train_one_run"],
        working_dir=Path.cwd(),
    )
    manifest.save(manifest_path)

    def save_state(state: dict[str, Any], path: Path) -> None:
        torch.save(state, path)

    tracker: TopKCheckpointTracker[dict[str, Any]] = TopKCheckpointTracker(
        checkpoints_dir,
        k=settings.top_k_checkpoints,
        save_fn=save_state,
    )

    try:
        model.to(settings.device)
        final_train_loss = float("nan")
        best_val_loss = float("inf")

        for epoch in range(settings.epochs):
            final_train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=settings.device,
                max_grad_norm=settings.max_grad_norm,
                max_batches=settings.max_train_batches,
            )
            val_loss = evaluate_loss(
                model=model,
                loader=val_loader,
                device=settings.device,
                max_batches=settings.max_val_batches,
            )
            best_val_loss = min(best_val_loss, val_loss)
            tracker.consider(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": final_train_loss,
                },
                epoch=epoch,
                metric=val_loss,
                val_loss=val_loss,
                metric_name="val_loss",
            )

        if tracker.best is None:
            raise RuntimeError("no checkpoint was saved")

        result = TrainingRunResult(
            best_checkpoint=tracker.best.path,
            manifest_path=manifest_path,
            best_val_loss=best_val_loss,
            final_train_loss=final_train_loss,
            epochs_completed=settings.epochs,
        )
        manifest.complete(
            metrics={
                "best_val_loss": best_val_loss,
                "final_train_loss": final_train_loss,
                "epochs_completed": settings.epochs,
                "best_epoch": tracker.best.epoch,
                "checkpoint_metric": settings.checkpoint_metric,
            },
            artifacts={"best_checkpoint": str(result.best_checkpoint)},
        )
        manifest.save(manifest_path)
        return result
    except BaseException as exc:
        manifest.fail(exc)
        manifest.save(manifest_path)
        raise


def train_one_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    device: str,
    max_grad_norm: float = 0.0,
    max_batches: int | None = None,
) -> float:
    torch = _import_torch()
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets = _inputs_targets(batch)
        inputs = inputs.to(device)
        targets = targets.to(device).long().view(-1)
        logits = model(inputs)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    if total_items == 0:
        raise ValueError("train_loader produced no items")
    return total_loss / total_items


def evaluate_loss(*, model: Any, loader: Any, device: str, max_batches: int | None = None) -> float:
    torch = _import_torch()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs, targets = _inputs_targets(batch)
            inputs = inputs.to(device)
            targets = targets.to(device).long().view(-1)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size

    if total_items == 0:
        raise ValueError("val_loader produced no items")
    return total_loss / total_items


def _inputs_targets(batch: Any) -> tuple[Any, Any]:
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("batch must contain at least inputs and targets")
    return batch[0], batch[1]


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Torch is required for training loop") from exc
    return torch
