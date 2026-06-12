"""Runtime training/scoring implementation for CodecFake+ XLS-R runs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import yaml

from mimodf.training.codecfake import (
    CLASSES,
    CodecfakeXlsrModelSmokeSettings,
    CodecfakeXlsrTrainSettings,
    _build_loaders,
    _build_xlsr_frontend,
    _validate_plan_settings,
    build_codecfake_xlsr_dry_run_plan,
)
from mimodf.training.codecfake_metrics import build_prediction_metrics
from mimodf.training.seeding import SeedSettings, seed_everything


def run_codecfake_xlsr_model_smoke(settings: CodecfakeXlsrModelSmokeSettings) -> dict[str, Any]:
    """Run one optimizer step and one validation forward pass.

    This is technical plumbing evidence only. It intentionally writes a small
    smoke JSON, not checkpoints or scores.
    """

    train_settings = CodecfakeXlsrTrainSettings(
        plan=settings.plan,
        checkpoint_path=settings.checkpoint_path,
        epochs=1,
        batch_size=settings.batch_size,
        eval_batch_size=settings.eval_batch_size,
        cut=settings.cut,
        num_workers=settings.num_workers,
        device=settings.device,
        lr=settings.lr,
        weight_decay=settings.weight_decay,
        deterministic=settings.deterministic,
        max_train_batches=1,
        max_val_batches=1,
        max_test_batches=0,
        save_checkpoints=False,
    )
    result = _run_training_core(train_settings, mode="model_smoke")
    smoke = {
        "status": "completed_model_smoke_only",
        "technical_scope": "one optimizer step and one validation forward pass; no checkpoint, scoring, or metrics claim",
        "plan": result["plan"],
        "checkpoint_path": result["checkpoint_path"],
        "device": result["device"],
        "batch_size": result["batch_size"],
        "eval_batch_size": result["eval_batch_size"],
        "cut": result["cut"],
        "num_workers": result["num_workers"],
        "lr": result["lr"],
        "weight_decay": result["weight_decay"],
        "trainable_parameters": result["trainable_parameters"],
        "total_parameters": result["total_parameters"],
        "train_batch": result["history"][0]["train_first_batch"],
        "validation_batch": result["history"][0]["validation_first_batch"],
        "elapsed_sec": result["elapsed_sec"],
    }
    output_path = Path(result["plan"]["output_paths"]["run_dir"]) / "model_smoke.json"
    output_path.write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n")
    smoke["output_path"] = str(output_path)
    return smoke


def run_codecfake_xlsr_training(settings: CodecfakeXlsrTrainSettings) -> dict[str, Any]:
    """Run a bounded audited training job and write manifest/history/scores."""

    return _run_training_core(settings, mode="train")


def _run_training_core(settings: CodecfakeXlsrTrainSettings, *, mode: str) -> dict[str, Any]:
    _validate_train_settings(settings)
    torch = _import_torch()
    seed_everything(
        SeedSettings(
            seed=settings.plan.seed,
            cudnn_deterministic=settings.deterministic,
            cudnn_benchmark=not settings.deterministic,
        )
    )
    started = time.time()

    plan = build_codecfake_xlsr_dry_run_plan(settings.plan).to_dict()
    run_dir = Path(plan["output_paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    if settings.save_checkpoints:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    _write_resolved_spec(run_dir / "resolved_spec.yaml", settings, mode, plan)
    manifest_path = run_dir / "manifest.json"
    manifest = _manifest(settings, mode, plan, status="running", started=started)
    _write_json(manifest_path, manifest)

    try:
        loaders = _build_loaders(
            settings.plan,
            settings.batch_size,
            settings.eval_batch_size,
            settings.cut,
            settings.num_workers,
            seed=settings.plan.seed if settings.deterministic else None,
        )
        from src.model import Model

        frontend = _build_xlsr_frontend(settings.plan.condition, settings.checkpoint_path)
        model = Model(frontend=frontend)
        if _is_frozen_frontend_condition(settings.plan.condition):
            model.frontend.freeze()
        model.to(settings.device)
        optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=settings.lr,
            weight_decay=settings.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()

        history_path = run_dir / "train_history.jsonl"
        best_metric_value: float | None = None
        best_val_loss = math.inf
        best_epoch: int | None = None
        best_checkpoint: str | None = None
        best_state: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        history_path.write_text("")
        for epoch in range(1, settings.epochs + 1):
            train_summary = _train_epoch(
                model,
                loaders.train_loader,
                optimizer,
                criterion,
                device=settings.device,
                max_batches=settings.max_train_batches,
                freeze_frontend=_is_frozen_frontend_condition(settings.plan.condition),
            )
            val_summary = _eval_loss(
                model,
                loaders.val_loader,
                criterion,
                device=settings.device,
                max_batches=settings.max_val_batches,
            )
            validation_metrics = build_prediction_metrics(val_summary["records"])
            selection_value = _selection_metric(validation_metrics, val_summary["loss"], settings)
            row = {
                "epoch": epoch,
                "train_loss": train_summary["loss"],
                "validation_loss": val_summary["loss"],
                "train_examples": train_summary["examples"],
                "validation_examples": val_summary["examples"],
                "train_batches": train_summary["batches"],
                "validation_batches": val_summary["batches"],
                "train_first_batch": train_summary["first_batch"],
                "validation_first_batch": val_summary["first_batch"],
                "validation_metrics": _history_metrics(validation_metrics),
                "checkpoint_metric": settings.checkpoint_metric,
                "checkpoint_metric_value": selection_value,
            }
            history.append(row)
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            if _is_better_metric(selection_value, best_metric_value, settings.checkpoint_metric):
                best_metric_value = selection_value
                best_val_loss = float(val_summary["loss"])
                best_epoch = epoch
                best_state = _checkpoint_model_state(model, settings.plan.condition)
                if settings.save_checkpoints:
                    best_checkpoint = str(checkpoints_dir / "best.pt")
                    torch.save(
                        _checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            settings=settings,
                            epoch=epoch,
                            validation_loss=best_val_loss,
                            validation_metrics=validation_metrics,
                            model_state_dict=best_state,
                        ),
                        best_checkpoint,
                    )

        if best_state is not None:
            _load_model_state(model, best_state)

        scores_path = run_dir / "scores.jsonl"
        metrics_path = run_dir / "metrics.json"
        report_path = run_dir / "report.md"
        prediction_records: list[dict[str, Any]] = []
        metrics: dict[str, Any] | None = None
        scores_sha256: str | None = None
        metrics_sha256: str | None = None
        if settings.max_test_batches != 0:
            prediction_records = _predict_records(
                model,
                loaders.test_loader,
                device=settings.device,
                max_batches=settings.max_test_batches,
            )
            _write_jsonl(scores_path, prediction_records)
            metrics = build_prediction_metrics(prediction_records)
            metrics.update(
                {
                    "status": "technical_smoke_metrics_only"
                    if settings.max_test_batches is not None
                    else "completed_test_metrics",
                    "test_batches_scored": _batches_scored(
                        settings.max_test_batches, prediction_records
                    ),
                    "caveats": _metric_caveats(settings),
                }
            )
            _write_json(metrics_path, metrics)
            report_path.write_text(_render_report(metrics, plan), encoding="utf-8")
            scores_sha256 = _sha256_file(scores_path)
            metrics_sha256 = _sha256_file(metrics_path)

        best_checkpoint_sha256 = _sha256_file(Path(best_checkpoint)) if best_checkpoint else None
        result = {
            "status": "completed_model_smoke_only" if mode == "model_smoke" else "completed",
            "mode": mode,
            "plan": plan,
            "checkpoint_path": _frontend_checkpoint_id(settings),
            "device": settings.device,
            "batch_size": settings.batch_size,
            "eval_batch_size": settings.eval_batch_size,
            "cut": settings.cut,
            "num_workers": settings.num_workers,
            "epochs": settings.epochs,
            "max_train_batches": settings.max_train_batches,
            "max_val_batches": settings.max_val_batches,
            "max_test_batches": settings.max_test_batches,
            "save_checkpoints": settings.save_checkpoints,
            "lr": settings.lr,
            "weight_decay": settings.weight_decay,
            "trainable_parameters": int(
                sum(param.numel() for param in model.parameters() if param.requires_grad)
            ),
            "total_parameters": int(sum(param.numel() for param in model.parameters())),
            "best_validation_loss": best_val_loss,
            "best_epoch": best_epoch,
            "checkpoint_metric": settings.checkpoint_metric,
            "best_checkpoint_metric_value": best_metric_value,
            "best_checkpoint": best_checkpoint,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "scores_sha256": scores_sha256,
            "metrics_sha256": metrics_sha256,
            "history": history,
            "metrics": metrics,
            "artifacts": {
                "resolved_spec": str(run_dir / "resolved_spec.yaml"),
                "manifest": str(manifest_path),
                "train_history": str(history_path),
                "scores": str(scores_path) if prediction_records else None,
                "metrics": str(metrics_path) if metrics is not None else None,
                "report": str(report_path) if metrics is not None else None,
                "best_checkpoint": best_checkpoint,
                "best_checkpoint_sha256": best_checkpoint_sha256,
                "scores_sha256": scores_sha256,
                "metrics_sha256": metrics_sha256,
            },
            "elapsed_sec": round(time.time() - started, 3),
        }
        manifest.update(
            {
                "status": "completed",
                "finished_unix": time.time(),
                "elapsed_sec": result["elapsed_sec"],
                "result_summary": {
                    "best_validation_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "checkpoint_metric": settings.checkpoint_metric,
                    "best_checkpoint_metric_value": best_metric_value,
                    "epochs_completed": settings.epochs,
                    "scored_records": len(prediction_records),
                    "best_checkpoint": best_checkpoint,
                    "best_checkpoint_sha256": best_checkpoint_sha256,
                    "scores_sha256": scores_sha256,
                    "metrics_sha256": metrics_sha256,
                },
                "artifacts": result["artifacts"],
            }
        )
        _write_json(manifest_path, manifest)
        return result
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_unix": time.time(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(manifest_path, manifest)
        raise


def _train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    *,
    device: str,
    max_batches: int | None,
    freeze_frontend: bool,
) -> dict[str, Any]:
    model.train()
    if freeze_frontend:
        model.frontend.eval()
    total_loss = 0.0
    total_examples = 0
    batches = 0
    first_batch: dict[str, Any] | None = None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets, _metadata = batch
        inputs = inputs.to(device)
        targets = targets.to(device).long().view(-1)
        logits = model(inputs)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = int(targets.shape[0])
        if first_batch is None:
            first_batch = _tensor_batch_summary(inputs, targets, logits, loss)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        batches += 1
    if total_examples == 0:
        raise ValueError("train_loader produced no items")
    return {
        "loss": total_loss / total_examples,
        "examples": total_examples,
        "batches": batches,
        "first_batch": first_batch,
    }


def _eval_loss(
    model: Any,
    loader: Any,
    criterion: Any,
    *,
    device: str,
    max_batches: int | None,
) -> dict[str, Any]:
    torch = _import_torch()
    model.eval()
    total_loss = 0.0
    total_examples = 0
    batches = 0
    first_batch: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs, targets, metadata = batch
            inputs = inputs.to(device)
            targets = targets.to(device).long().view(-1)
            logits = model(inputs)
            loss = criterion(logits, targets)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            predictions = probabilities.argmax(dim=1)
            _extend_prediction_records(
                records,
                seen,
                probabilities=probabilities,
                predictions=predictions,
                targets=targets.detach().cpu(),
                metadata=metadata,
            )
            batch_size = int(targets.shape[0])
            if first_batch is None:
                first_batch = _tensor_batch_summary(inputs, targets, logits, loss)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            batches += 1
    if total_examples == 0:
        raise ValueError("validation loader produced no items")
    return {
        "loss": total_loss / total_examples,
        "examples": total_examples,
        "batches": batches,
        "first_batch": first_batch,
        "records": records,
    }


def _predict_records(
    model: Any, loader: Any, *, device: str, max_batches: int | None
) -> list[dict[str, Any]]:
    torch = _import_torch()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs, targets, metadata = batch
            inputs = inputs.to(device)
            targets = targets.long().view(-1)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            predictions = probabilities.argmax(dim=1)
            _extend_prediction_records(
                records,
                seen,
                probabilities=probabilities,
                predictions=predictions,
                targets=targets,
                metadata=metadata,
            )
    if not records:
        raise ValueError("test loader produced no scored items")
    return records


def _write_resolved_spec(
    path: Path, settings: CodecfakeXlsrTrainSettings, mode: str, plan: dict[str, Any]
) -> None:
    payload = {
        "schema": "mimodf-codecfake-xlsr-resolved-spec/v1",
        "mode": mode,
        "plan": plan,
        "training": _train_settings_payload(settings),
        "label_convention": {label: index for index, label in enumerate(CLASSES)},
        "caveats": [
            "CoSG source-holdout is a custom diagnostic protocol, not an official CodecFake+ benchmark split",
            "bounded smoke metrics are plumbing evidence only unless a confirmatory run is explicitly predeclared",
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _manifest(
    settings: CodecfakeXlsrTrainSettings,
    mode: str,
    plan: dict[str, Any],
    *,
    status: str,
    started: float,
) -> dict[str, Any]:
    return {
        "schema": "mimodf-codecfake-xlsr-run-manifest/v1",
        "status": status,
        "mode": mode,
        "started_unix": started,
        "plan": plan,
        "training": _train_settings_payload(settings),
        "label_convention": {label: index for index, label in enumerate(CLASSES)},
    }


def _train_settings_payload(settings: CodecfakeXlsrTrainSettings) -> dict[str, Any]:
    return {
        "checkpoint_path": _frontend_checkpoint_id(settings),
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "eval_batch_size": settings.eval_batch_size,
        "cut": settings.cut,
        "num_workers": settings.num_workers,
        "device": settings.device,
        "lr": settings.lr,
        "weight_decay": settings.weight_decay,
        "max_train_batches": settings.max_train_batches,
        "max_val_batches": settings.max_val_batches,
        "max_test_batches": settings.max_test_batches,
        "save_checkpoints": settings.save_checkpoints,
        "checkpoint_metric": settings.checkpoint_metric,
        "deterministic": settings.deterministic,
        "cudnn_deterministic": settings.deterministic,
        "cudnn_benchmark": not settings.deterministic,
        "dataloader_seed": settings.plan.seed if settings.deterministic else None,
    }


def _render_report(metrics: dict[str, Any], plan: dict[str, Any]) -> str:
    lines = [
        "# CodecFake XLS-R run report",
        "",
        f"Fold: `{plan['fold']}`",
        f"Condition: `{plan['condition']}`",
        f"Records scored: `{metrics['records']}`",
        "",
        "## Metrics",
        "",
        f"- accuracy: {metrics['accuracy']:.4f}",
        f"- balanced_accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- macro_f1: {metrics['macro_f1']:.4f}",
        f"- auroc: {_format_metric(metrics.get('auroc'))}",
        f"- eer: {_format_metric(metrics.get('eer'))}",
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {item}" for item in metrics["caveats"])
    lines.append("")
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _metric_caveats(settings: CodecfakeXlsrTrainSettings) -> list[str]:
    caveats = [
        "CoSG source-holdout is a custom diagnostic protocol, not the official CodecFake+ split",
        "label convention is bonafide=0, spoof=1; score is P(spoof)",
    ]
    if settings.max_test_batches is not None:
        caveats.append("test scoring was batch-limited; metrics are technical smoke evidence only")
    return caveats


def _batches_scored(max_test_batches: int | None, records: list[dict[str, Any]]) -> int | str:
    if max_test_batches is None:
        return "all"
    return 0 if not records else max_test_batches


def _selection_metric(
    validation_metrics: dict[str, Any], validation_loss: float, settings: CodecfakeXlsrTrainSettings
) -> float:
    if settings.checkpoint_metric == "val_loss":
        return float(validation_loss)
    if settings.checkpoint_metric == "val_auroc":
        value = validation_metrics.get("auroc")
    elif settings.checkpoint_metric == "val_eer":
        value = validation_metrics.get("eer")
    else:
        raise ValueError(f"unsupported checkpoint_metric: {settings.checkpoint_metric}")
    if value is None:
        raise ValueError(f"{settings.checkpoint_metric} is undefined for validation support")
    return float(value)


def _is_better_metric(value: float, best: float | None, checkpoint_metric: str) -> bool:
    if best is None:
        return True
    if checkpoint_metric == "val_auroc":
        return value > best
    return value < best


def _history_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "records",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "eer",
        "binary_metric_status",
        "score_summary_by_label",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def _tensor_batch_summary(inputs: Any, targets: Any, logits: Any, loss: Any) -> dict[str, Any]:
    return {
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
    }


def _extend_prediction_records(
    records: list[dict[str, Any]],
    seen: set[str],
    *,
    probabilities: Any,
    predictions: Any,
    targets: Any,
    metadata: dict[str, Any],
) -> None:
    ids = _metadata_values(metadata, "utterance_id")
    sources = _metadata_values(metadata, "source_model")
    labels = _metadata_values(metadata, "label")
    audio_paths = _metadata_values(metadata, "audio_path")
    for index, utterance_id in enumerate(ids):
        if utterance_id in seen:
            raise ValueError(f"duplicate utterance_id in predictions: {utterance_id}")
        seen.add(utterance_id)
        records.append(
            {
                "schema": "mimodf-codecfake-xlsr-prediction/v1",
                "utterance_id": utterance_id,
                "target": CLASSES[int(targets[index])],
                "prediction": CLASSES[int(predictions[index])],
                "score": float(probabilities[index, CLASSES.index("spoof")]),
                "probabilities": {
                    label: float(probabilities[index, class_index])
                    for class_index, label in enumerate(CLASSES)
                },
                "label": labels[index],
                "source_model": sources[index],
                "audio_path": audio_paths[index],
                "label_convention": {
                    label: class_index for class_index, label in enumerate(CLASSES)
                },
            }
        )


def _metadata_values(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        return [str(item) for item in value.tolist()]
    return [str(value)]


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    settings: CodecfakeXlsrTrainSettings,
    epoch: int,
    validation_loss: float,
    validation_metrics: dict[str, Any],
    model_state_dict: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "mimodf-codecfake-xlsr-checkpoint/v2",
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "model_state_scope": _checkpoint_state_scope(settings.plan.condition),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_metrics": validation_metrics,
        "checkpoint_metric": settings.checkpoint_metric,
        "label_convention": {label: index for index, label in enumerate(CLASSES)},
        "condition": settings.plan.condition,
        "frontend_checkpoint": _frontend_checkpoint_id(settings),
        "frontend_checkpoint_sha256": _frontend_checkpoint_sha256(settings),
        "total_parameters": int(sum(param.numel() for param in model.parameters())),
        "trainable_parameters": int(
            sum(param.numel() for param in model.parameters() if param.requires_grad)
        ),
    }


def _checkpoint_model_state(model: Any, condition: str) -> dict[str, Any]:
    state = model.state_dict()
    if condition == "xlsr_full_finetune":
        selected = state
    elif condition == "xlsr_peft_adapter":
        selected = {
            key: value for key, value in state.items() if not key.startswith("frontend.model.")
        }
    else:
        selected = {key: value for key, value in state.items() if not key.startswith("frontend.")}
    return {key: value.detach().cpu().clone() for key, value in selected.items()}


def _checkpoint_state_scope(condition: str) -> str:
    if condition == "xlsr_full_finetune":
        return "full_model_state_dict"
    if condition == "xlsr_peft_adapter":
        return "backend_projection_and_adapter_state_dict"
    return "backend_projection_state_dict_excludes_frozen_frontend"


def _load_model_state(model: Any, state: dict[str, Any]) -> None:
    model.load_state_dict(state, strict=False)


def _validate_train_settings(settings: CodecfakeXlsrTrainSettings) -> None:
    _validate_plan_settings(settings.plan)
    if settings.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if settings.batch_size < 1 or settings.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if settings.lr <= 0:
        raise ValueError("lr must be positive")
    if settings.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if settings.checkpoint_metric not in {"val_loss", "val_auroc", "val_eer"}:
        raise ValueError("checkpoint_metric must be val_loss, val_auroc, or val_eer")
    for name, value in (
        ("max_train_batches", settings.max_train_batches),
        ("max_val_batches", settings.max_val_batches),
        ("max_test_batches", settings.max_test_batches),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative when set")
    if settings.max_train_batches == 0:
        raise ValueError("max_train_batches must be positive when set")
    if settings.max_val_batches == 0:
        raise ValueError("max_val_batches must be positive when set")
    if settings.plan.condition != "wavlm_frozen_backend" and not settings.checkpoint_path.is_file():
        raise FileNotFoundError(settings.checkpoint_path)


def _is_frozen_frontend_condition(condition: str) -> bool:
    return condition in {"xlsr_frozen_backend", "wavlm_frozen_backend"}


def _frontend_checkpoint_id(settings: CodecfakeXlsrTrainSettings) -> str:
    if settings.plan.condition == "wavlm_frozen_backend":
        return "hf:microsoft/wavlm-base-plus@b21194173c0af7e94822c1776d162e2659fd4761"
    return str(settings.checkpoint_path)


def _frontend_checkpoint_sha256(settings: CodecfakeXlsrTrainSettings) -> str:
    if settings.plan.condition == "wavlm_frozen_backend":
        return "not_applicable_huggingface_revision_b21194173c0af7e94822c1776d162e2659fd4761"
    return _sha256_file(settings.checkpoint_path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Torch is required for CodecFake XLS-R training") from exc
    return torch
