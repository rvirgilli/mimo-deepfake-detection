"""
Training script for audio deepfake detection with swappable frontends.

Usage:
    # Train with wav2vec2 (default)
    python train.py

    # Train with MiMo frontend
    python train.py frontend=mimo

    # Override dataset path
    python train.py dataset.database_path=/path/to/data

Based on SSL_Anti-spoofing by Hemlata Tak.
"""

import warnings
# Suppress librosa deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

import os
import sys
import logging
from datetime import datetime
from typing import Any

import hydra
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from src.frontends import get_frontend
from src.model import Model
from src.data_utils import (
    genSpoof_list,
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2021_eval,
    Dataset_ASVspoof2021_fast_eval,
)
from src.experiment import ExperimentManifest, TrainingResult
from src.results import ResultsDB

# Add SSL_Anti-spoofing to path for eval_metric_LA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SSL_Anti-spoofing"))
from eval_metric_LA import compute_eer


def set_random_seed(seed: int, cfg: DictConfig) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = cfg.training.cudnn_deterministic
        torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark


def evaluate_accuracy(
    dev_loader: DataLoader, model: nn.Module, device: str, desc: str = "Validating"
) -> float:
    """Evaluate model on validation set."""
    val_loss = 0.0
    num_total = 0.0
    model.eval()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    pbar = tqdm(dev_loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch_x, batch_y in pbar:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)

            batch_loss = criterion(batch_out, batch_y)
            val_loss += batch_loss.item() * batch_size
            pbar.set_postfix({"loss": f"{val_loss/num_total:.4f}"})

    val_loss /= num_total
    return val_loss


def load_fast_eval_labels(key_path: str) -> dict:
    """Load labels from fast eval key file."""
    labels = {}
    with open(key_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Format: - file_id - - label
            if len(parts) >= 5:
                file_id = parts[1]
                label = parts[4]  # 'bonafide' or 'spoof'
                labels[file_id] = label
    return labels


def evaluate_fast_eer(
    eval_loader: DataLoader,
    model: nn.Module,
    device: str,
    labels: dict,
    desc: str = "Fast Eval"
) -> tuple[float, float]:
    """
    Evaluate model on fast eval subset with EER computation.

    Returns:
        (val_loss, eer)
    """
    val_loss = 0.0
    num_total = 0.0
    model.eval()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Collect scores and corresponding labels
    all_scores = []
    all_labels = []

    pbar = tqdm(eval_loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch_x, batch_y, file_ids in pbar:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)

            batch_loss = criterion(batch_out, batch_y)
            val_loss += batch_loss.item() * batch_size

            # Score = logit for bonafide class (higher = more likely bonafide)
            scores = batch_out[:, 1].cpu().numpy()

            for score, file_id in zip(scores, file_ids):
                all_scores.append(score)
                all_labels.append(labels.get(file_id, 'spoof'))

            pbar.set_postfix({"loss": f"{val_loss/num_total:.4f}"})

    val_loss /= num_total

    # Compute EER
    bonafide_scores = np.array([s for s, l in zip(all_scores, all_labels) if l == 'bonafide'])
    spoof_scores = np.array([s for s, l in zip(all_scores, all_labels) if l == 'spoof'])

    if len(bonafide_scores) > 0 and len(spoof_scores) > 0:
        eer, _ = compute_eer(bonafide_scores, spoof_scores)
    else:
        eer = 0.5  # Default to 50% if missing data

    return val_loss, eer


def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_weights: list,
    epoch: int = 0,
    total_epochs: int = 100,
    max_grad_norm: float = 0.0,
) -> float:
    """Train for one epoch."""
    running_loss = 0.0
    num_total = 0.0
    model.train()

    weight = torch.FloatTensor(loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
    for batch_x, batch_y in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size

        optimizer.zero_grad()
        batch_loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        pbar.set_postfix({"loss": f"{running_loss/num_total:.4f}"})

    running_loss /= num_total
    return running_loss


class RawBoostArgs:
    """Container for RawBoost arguments from Hydra config."""

    def __init__(self, cfg: DictConfig):
        for key, value in cfg.items():
            setattr(self, key, value)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    set_random_seed(cfg.seed, cfg)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create output directories
    model_save_path = os.path.join(
        cfg.model_save_dir, f"{cfg.experiment_name}_seed{cfg.seed}"
    )
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(model_save_path, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    print(f"Config saved to {config_path}")

    # Create experiment manifest
    manifest = ExperimentManifest.create(
        cfg=cfg,
        experiment_name=cfg.experiment_name,
        model_save_path=model_save_path,
    )
    manifest.save()
    print(f"Experiment ID: {manifest.experiment_id}")

    # Create frontend
    print(f"\nInitializing {cfg.frontend.name} frontend...")
    frontend_cfg = dict(cfg.frontend)

    # Extract finetune config for MiMo, wav2vec2, or HuBERT
    finetune_config = None
    if cfg.frontend.name in ("mimo", "wav2vec2", "hubert") and "finetune" in cfg.frontend:
        finetune_config = OmegaConf.to_container(cfg.frontend.finetune, resolve=True)

    # Extract feature config for MiMo
    feature_type = "continuous"  # default
    feature_config = None
    if cfg.frontend.name == "mimo" and "feature" in cfg.frontend:
        feature_cfg = cfg.frontend.feature
        feature_type = feature_cfg.get("type", "continuous")
        # Get strategy-specific config
        if feature_type in feature_cfg:
            feature_config = OmegaConf.to_container(feature_cfg[feature_type], resolve=True)

    frontend = get_frontend(
        cfg.frontend.name,
        checkpoint=frontend_cfg.get("checkpoint"),
        model_path=frontend_cfg.get("model_path"),
        model_name=frontend_cfg.get("model_name"),
        freeze=frontend_cfg.get("freeze"),
        use_bfloat16=frontend_cfg.get("use_bfloat16", True),
        upsample_to_50hz=frontend_cfg.get("upsample_to_50hz", False),
        upsample_mode=frontend_cfg.get("upsample_mode", "linear"),
        native_50hz=frontend_cfg.get("native_50hz", False),
        feature_type=feature_type,
        feature_config=feature_config,
        finetune_config=finetune_config,
    )
    print(f"Frontend output dim: {frontend.out_dim}")
    print(f"Frontend sample rate: {frontend.sample_rate}")
    print(f"Frontend parameters: {frontend.num_params:,}")
    print(f"Frontend trainable parameters: {frontend.num_trainable_params:,}")

    # Create model with architecture config
    model_cfg = cfg.get("model", {})
    projection_cfg = model_cfg.get("projection", {})
    model = Model(
        frontend=frontend,
        # Backend capacity parameters
        filts_0=model_cfg.get("filts_0", 128),
        encoder_scale=model_cfg.get("encoder_scale", 1.0),
        # GAT and pooling
        gat_dims=list(model_cfg.get("gat_dims", [64, 32])),
        pool_ratios=list(model_cfg.get("pool_ratios", [0.5, 0.5, 0.5, 0.5])),
        temperatures=list(model_cfg.get("temperatures", [2.0, 2.0, 100.0, 100.0])),
        dropout=model_cfg.get("dropout", 0.5),
        dropout_way=model_cfg.get("dropout_way", 0.2),
        # Projection parameters
        projection_type=projection_cfg.get("type", "linear"),
        projection_hidden_dims=list(projection_cfg.get("hidden_dims", [512, 256])) if projection_cfg.get("hidden_dims") else None,
        projection_activation=projection_cfg.get("activation", "gelu"),
        projection_dropout=projection_cfg.get("dropout", 0.1),
        projection_use_batchnorm=projection_cfg.get("use_batchnorm", True),
    )
    model = model.to(device)
    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {nb_params:,}")

    # Optimizer with discriminative learning rates
    # encoder_lr: for MiMo encoder (should be very low, 1e-6 to 1e-7)
    # lr (backend_lr): for AASIST backend (can be higher, 1e-4 to 1e-5)
    encoder_lr = cfg.training.get('encoder_lr', None)
    backend_lr = cfg.training.lr

    if encoder_lr is not None:
        # Discriminative LR: separate encoder and backend parameters
        param_groups = []

        # Encoder parameters (from frontend)
        encoder_params = []
        if hasattr(model, 'frontend') and hasattr(model.frontend, 'get_trainable_params'):
            encoder_params = list(model.frontend.get_trainable_params())

        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': encoder_lr,
                'name': 'encoder'
            })
            print(f"Encoder params: {sum(p.numel() for p in encoder_params):,} @ LR={encoder_lr:.2e}")

        # Backend parameters (everything not in frontend)
        backend_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and 'frontend' not in n]
        if backend_params:
            param_groups.append({
                'params': backend_params,
                'lr': backend_lr,
                'name': 'backend'
            })
            print(f"Backend params: {sum(p.numel() for p in backend_params):,} @ LR={backend_lr:.2e}")

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=cfg.training.weight_decay,
        )
        print(f"Using discriminative LR: encoder={encoder_lr:.2e}, backend={backend_lr:.2e}")
    else:
        # Single LR for all parameters (default behavior)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

    # LR Scheduler
    scheduler = None
    scheduler_type = None
    scheduler_cfg = cfg.training.get("scheduler", {})
    if scheduler_cfg.get("enabled", False):
        scheduler_type = scheduler_cfg.get("type", "cosine")
        if scheduler_type == "cosine":
            T_max = scheduler_cfg.get("T_max", cfg.training.num_epochs)
            eta_min = float(scheduler_cfg.get("eta_min", 1e-7))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
            print(f"LR Scheduler: CosineAnnealing (T_max={T_max}, eta_min={eta_min})")
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("plateau_patience", 2),
                min_lr=float(scheduler_cfg.get("eta_min", 1e-7)),
            )
            print(f"LR Scheduler: ReduceLROnPlateau (factor={scheduler_cfg.factor}, patience={scheduler_cfg.plateau_patience})")

    # Resume from checkpoint if specified
    start_epoch = cfg.get("start_epoch", 0)
    if cfg.get("resume_from") is not None:
        print(f"\nResuming from checkpoint: {cfg.resume_from}")
        checkpoint = torch.load(cfg.resume_from, map_location=device, weights_only=False)

        # Handle both checkpoint formats:
        # 1. Optuna format: dict with 'model_state_dict' key
        # 2. train.py format: raw state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if "epoch" in checkpoint:
                print(f"  -> Checkpoint from epoch {checkpoint['epoch']}, best_eer={checkpoint.get('best_eer', 'N/A')}")
        else:
            state_dict = checkpoint

        # Load with strict=False to allow partial loading (e.g., frozen MiMo weights)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  -> Missing keys (will use initialized values): {len(missing)} keys")
        if unexpected:
            print(f"  -> Unexpected keys (ignored): {len(unexpected)} keys")
        print(f"  -> Loaded model weights, starting from epoch {start_epoch}")

    # Dataset paths
    track = cfg.track
    prefix = f"ASVspoof_{track}"
    prefix_2019 = f"ASVspoof2019.{track}"

    # RawBoost args
    rawboost_args = RawBoostArgs(cfg.rawboost)

    # Training data
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(
            cfg.dataset.protocols_path,
            f"{prefix}_cm_protocols/{prefix_2019}.cm.train.trn.txt",
        ),
        is_train=True,
        is_eval=False,
    )
    print(f"\nTraining trials: {len(file_train)}")

    train_set = Dataset_ASVspoof2019_train(
        args=rawboost_args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(
            cfg.dataset.database_path, f"ASVspoof2019_{track}_train/"
        ),
        algo=cfg.rawboost.algo,
        sample_rate=cfg.frontend.sample_rate,
        cut=cfg.frontend.cut,
        auto_scale_rawboost=cfg.rawboost.get("auto_scale", True),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
        drop_last=True,
    )

    del train_set, d_label_trn

    # Fast validation data (5% stratified subset of 2021 LA eval)
    fast_eval_trl = os.path.join(
        cfg.dataset.protocols_path,
        f"{prefix}_cm_protocols/ASVspoof2021.{track}.cm.eval.fast.trl.txt",
    )
    fast_eval_key = os.path.join(
        cfg.dataset.protocols_path,
        f"{prefix}_cm_protocols/ASVspoof2021.{track}.cm.eval.fast.key.txt",
    )

    # Check if fast eval subset exists
    if os.path.exists(fast_eval_trl) and os.path.exists(fast_eval_key):
        print("\nUsing fast 2021 LA eval subset for validation...")
        # Load file list
        file_dev = genSpoof_list(dir_meta=fast_eval_trl, is_train=False, is_eval=True)
        # Load labels
        fast_eval_labels = load_fast_eval_labels(fast_eval_key)
        print(f"Fast eval trials: {len(file_dev)}")
        print(f"  Bonafide: {sum(1 for l in fast_eval_labels.values() if l == 'bonafide')}")
        print(f"  Spoof: {sum(1 for l in fast_eval_labels.values() if l == 'spoof')}")

        dev_set = Dataset_ASVspoof2021_fast_eval(
            list_IDs=file_dev,
            labels=fast_eval_labels,
            base_dir=os.path.join(cfg.dataset.database_path, "ASVspoof2021_LA_eval/"),
            sample_rate=cfg.frontend.sample_rate,
            cut=cfg.frontend.cut,
        )
        use_fast_eval = True
    else:
        # Fallback to 2019 dev set
        print("\nFast eval subset not found, using 2019 dev set...")
        d_label_dev, file_dev = genSpoof_list(
            dir_meta=os.path.join(
                cfg.dataset.protocols_path,
                f"{prefix}_cm_protocols/{prefix_2019}.cm.dev.trl.txt",
            ),
            is_train=False,
            is_eval=False,
        )
        print(f"Validation trials: {len(file_dev)}")

        dev_set = Dataset_ASVspoof2019_train(
            args=rawboost_args,
            list_IDs=file_dev,
            labels=d_label_dev,
            base_dir=os.path.join(cfg.dataset.database_path, f"ASVspoof2019_{track}_dev/"),
            algo=cfg.rawboost.algo,
            sample_rate=cfg.frontend.sample_rate,
            cut=cfg.frontend.cut,
            auto_scale_rawboost=cfg.rawboost.get("auto_scale", True),
        )
        fast_eval_labels = None
        use_fast_eval = False

    dev_loader = DataLoader(
        dev_set,
        batch_size=cfg.dataset.eval_batch_size if hasattr(cfg.dataset, 'eval_batch_size') else cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False,
    )

    del dev_set

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.experiment_name))

    # Training loop
    train_start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"TRAINING START: {cfg.experiment_name}")
    logger.info(f"Epochs: {start_epoch} to {cfg.training.num_epochs}, Seed: {cfg.seed}")
    logger.info("=" * 60)
    best_eer = float("inf")
    best_val_loss = float("inf")

    # Track top-k checkpoints by EER: list of (eer, val_loss, epoch, path)
    top_k = 5
    top_checkpoints: list[tuple[float, float, int, str]] = []

    # Early stopping
    patience = cfg.training.get("patience", 0)
    epochs_without_improvement = 0
    best_metric_for_patience = float("inf")
    if patience > 0:
        print(f"Early stopping enabled with patience={patience}")

    final_epoch = start_epoch  # Track actual final epoch for manifest
    for epoch in range(start_epoch, cfg.training.num_epochs):
        # Handle gradual unfreezing if configured
        if hasattr(model, 'frontend') and hasattr(model.frontend, 'finetune_wrapper'):
            wrapper = model.frontend.finetune_wrapper
            if wrapper is not None and hasattr(wrapper, 'on_epoch_start'):
                n_unfrozen = wrapper.on_epoch_start(epoch)
                if epoch == start_epoch or n_unfrozen != getattr(wrapper, '_prev_unfrozen', -1):
                    print(f"Epoch {epoch}: {n_unfrozen} transformer layers unfrozen")
                    wrapper._prev_unfrozen = n_unfrozen

        running_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            device,
            cfg.training.loss_weights,
            epoch=epoch,
            total_epochs=cfg.training.num_epochs,
            max_grad_norm=float(cfg.training.get("max_grad_norm", 0.0)),
        )

        # Evaluate with EER if using fast eval, otherwise just loss
        if use_fast_eval:
            val_loss, eer = evaluate_fast_eer(dev_loader, model, device, fast_eval_labels)
        else:
            val_loss = evaluate_accuracy(dev_loader, model, device)
            eer = None

        # Logging
        writer.add_scalar("loss/train", running_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        if eer is not None:
            writer.add_scalar("eer/val", eer, epoch)

        # Determine if this is a new best
        is_best = (eer is not None and eer < best_eer) or (eer is None and val_loss < best_val_loss)
        best_marker = " ★ BEST" if is_best else ""

        if eer is not None:
            logger.info(f"Epoch {epoch:3d}/{cfg.training.num_epochs} | Train: {running_loss:.4f} | Val: {val_loss:.4f} | EER: {eer*100:.2f}%{best_marker}")
        else:
            logger.info(f"Epoch {epoch:3d}/{cfg.training.num_epochs} | Train: {running_loss:.6f} | Val: {val_loss:.6f}{best_marker}")

        # Keep top-k best checkpoints by EER (or val_loss if EER not available)
        metric = eer if eer is not None else val_loss
        checkpoint_path = os.path.join(
            model_save_path,
            f"epoch_{epoch}_eer_{eer*100:.2f}.pth" if eer is not None else f"epoch_{epoch}_loss_{val_loss:.6f}.pth"
        )

        if len(top_checkpoints) < top_k:
            # Still have room, save checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            top_checkpoints.append((metric, val_loss, epoch, checkpoint_path))
            top_checkpoints.sort(key=lambda x: x[0])  # Sort by metric ascending (lower is better)
        elif metric < top_checkpoints[-1][0]:
            # Better than worst in top-k, replace it
            _, _, _, worst_path = top_checkpoints.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)
            torch.save(model.state_dict(), checkpoint_path)
            top_checkpoints.append((metric, val_loss, epoch, checkpoint_path))
            top_checkpoints.sort(key=lambda x: x[0])

        # Track overall best
        if eer is not None and eer < best_eer:
            best_eer = eer
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Step LR scheduler
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler_metric = eer if eer is not None else val_loss
                scheduler.step(scheduler_metric)
            else:  # cosine or other epoch-based schedulers
                scheduler.step()

        # Update final epoch
        final_epoch = epoch + 1

        # Early stopping check
        if patience > 0:
            current_metric = eer if eer is not None else val_loss
            if current_metric < best_metric_for_patience:
                best_metric_for_patience = current_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > 1:
                    logger.info(f"  No improvement for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

    writer.close()

    # Training summary
    train_duration = datetime.now() - train_start_time
    train_minutes = train_duration.total_seconds() / 60

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    if best_eer < float("inf"):
        logger.info(f"Best EER: {best_eer*100:.2f}% (epoch {top_checkpoints[0][2] if top_checkpoints else '?'})")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"Duration: {train_minutes:.1f} minutes ({final_epoch - start_epoch} epochs)")
    logger.info(f"Models: {model_save_path}")
    logger.info("=" * 60)

    # Complete experiment manifest
    metrics = {
        "best_eer": best_eer if best_eer < float("inf") else None,
        "best_val_loss": best_val_loss,
        "final_train_loss": running_loss,
        "epochs_completed": final_epoch - start_epoch,
        "best_epoch": top_checkpoints[0][2] if top_checkpoints else 0,
        "early_stopped": patience > 0 and epochs_without_improvement >= patience,
        "train_duration_minutes": train_minutes,
    }
    manifest.complete(metrics, status="completed")
    manifest.save()

    # Save to results database
    try:
        results_db = ResultsDB()
        results_db.add_experiment(manifest)
        logger.info(f"Experiment saved to results database: {manifest.experiment_id}")
    except Exception as e:
        logger.warning(f"Failed to save to results database: {e}")


if __name__ == "__main__":
    main()
