"""
Evaluation script for audio deepfake detection.

Usage:
    # Evaluate with wav2vec2 frontend
    python evaluate.py frontend=wav2vec2 model_path=./experiments/models/best.pth

    # Evaluate with MiMo frontend
    python evaluate.py frontend=mimo model_path=./experiments/models/best.pth

Based on SSL_Anti-spoofing by Hemlata Tak.
"""

import os
import sys
from typing import Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.frontends import get_frontend
from src.model import Model
from src.data_utils import genSpoof_list, Dataset_ASVspoof2021_eval


def produce_evaluation_file(
    dataset: Dataset_ASVspoof2021_eval,
    model: nn.Module,
    device: str,
    save_path: str,
    batch_size: int = 10,
) -> None:
    """
    Produce evaluation scores file.

    Args:
        dataset: Evaluation dataset
        model: Trained model
        device: Device to run inference on
        save_path: Path to save scores
        batch_size: Batch size for inference
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()

    # Clear output file
    if os.path.exists(save_path):
        os.remove(save_path)

    pbar = tqdm(data_loader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for batch_x, utt_id in pbar:
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)

            # Get bonafide score (index 1)
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()

            # Write scores
            with open(save_path, "a") as fh:
                for f, cm in zip(utt_id, batch_score.tolist()):
                    fh.write(f"{f} {cm}\n")

    print(f"\nScores saved to {save_path}")


def compute_eer(scores_path: str, protocols_path: str, prefix_2021: str) -> tuple:
    """
    Compute EER and min t-DCF from scores file.

    Returns:
        (eer, min_tdcf)
    """
    # Try to import metrics
    try:
        sys.path.insert(0, "./SSL_Anti-spoofing")
        from eval_metric_LA import compute_eer_and_tdcf
        return compute_eer_and_tdcf(scores_path, protocols_path, prefix_2021)
    except ImportError:
        print("Warning: eval_metric_LA not found. Computing basic EER only.")

        # Basic EER computation
        import numpy as np
        from sklearn.metrics import roc_curve

        # Load scores
        scores = {}
        with open(scores_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                scores[parts[0]] = float(parts[1])

        # Load ground truth from protocols
        cm_key_file = os.path.join(
            protocols_path,
            f"ASVspoof_{prefix_2021.split('.')[1]}_cm_protocols",
            f"{prefix_2021}.cm.eval.trl.txt",
        )

        # Check for key file with labels
        labels_available = False
        y_true = []
        y_scores = []

        if os.path.exists(cm_key_file):
            with open(cm_key_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        key = parts[1]
                        label = parts[4]
                        if key in scores:
                            y_true.append(1 if label == "bonafide" else 0)
                            y_scores.append(scores[key])
                            labels_available = True

        if labels_available and len(y_true) > 0:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
            return eer, None
        else:
            print("Labels not available for EER computation")
            return None, None


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    print(OmegaConf.to_yaml(cfg))

    # Check model path
    model_path = cfg.get("model_path")
    if model_path is None:
        print("Error: model_path is required")
        print("Usage: python evaluate.py model_path=/path/to/model.pth")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create frontend
    print(f"\nInitializing {cfg.frontend.name} frontend...")
    frontend_cfg = dict(cfg.frontend)
    frontend = get_frontend(
        cfg.frontend.name,
        checkpoint=frontend_cfg.get("checkpoint"),
        model_path=frontend_cfg.get("model_path"),
        freeze=cfg.frontend.freeze,
        use_bfloat16=frontend_cfg.get("use_bfloat16", True),
    )
    print(f"Frontend output dim: {frontend.out_dim}")
    print(f"Frontend sample rate: {frontend.sample_rate}")

    # Create model
    model = Model(frontend=frontend)
    model = model.to(device)

    # Load weights
    print(f"\nLoading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)

    # Remap keys for baseline compatibility (ssl_model -> frontend)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("ssl_model.", "frontend.")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    print("Model loaded successfully")

    # Dataset paths
    track = cfg.track
    prefix = f"ASVspoof_{track}"
    prefix_2021 = f"ASVspoof2021.{track}"

    # Evaluation data
    file_eval = genSpoof_list(
        dir_meta=os.path.join(
            cfg.dataset.protocols_path,
            f"{prefix}_cm_protocols/{prefix_2021}.cm.eval.trl.txt",
        ),
        is_train=False,
        is_eval=True,
    )
    print(f"\nEvaluation trials: {len(file_eval)}")

    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=file_eval,
        base_dir=os.path.join(cfg.dataset.database_path, f"ASVspoof2021_{track}_eval/"),
        sample_rate=cfg.frontend.sample_rate,
        cut=cfg.frontend.cut,
    )

    # Output path
    output_dir = cfg.get("eval_output_dir", "./experiments/scores")
    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(
        output_dir, f"scores_{cfg.frontend.name}_{track}_{os.path.basename(model_path)}.txt"
    )

    # Produce evaluation file
    dataset_cfg = dict(cfg.dataset)
    eval_batch_size = dataset_cfg.get("eval_batch_size", cfg.dataset.batch_size)
    print(f"\nRunning inference with batch_size={eval_batch_size}...")
    produce_evaluation_file(
        eval_set, model, device, scores_path, batch_size=eval_batch_size
    )

    # Compute metrics
    print("\nComputing metrics...")
    eer, min_tdcf = compute_eer(scores_path, cfg.dataset.protocols_path, prefix_2021)

    if eer is not None:
        print(f"\n{'='*50}")
        print(f"Results for {cfg.frontend.name} on {track}:")
        print(f"  EER: {eer:.4f}%")
        if min_tdcf is not None:
            print(f"  min t-DCF: {min_tdcf:.4f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
