"""
Experiment manifest for tracking and reproducibility.

This module provides utilities to capture comprehensive experiment metadata
including git state, system info, configs, and results for paper-ready
experiment management.
"""

import json
import os
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


def get_git_info() -> Dict[str, Any]:
    """Get current git commit hash and dirty state."""
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        git_hash = result.stdout.strip() if result.returncode == 0 else "unknown"

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        git_dirty = bool(result.stdout.strip()) if result.returncode == 0 else False

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        git_branch = result.stdout.strip() if result.returncode == 0 else "unknown"

        return {
            "hash": git_hash,
            "dirty": git_dirty,
            "branch": git_branch,
        }
    except Exception:
        return {"hash": "unknown", "dirty": False, "branch": "unknown"}


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = line.split(", ")
                if len(parts) >= 2:
                    gpus.append({
                        "name": parts[0].strip(),
                        "vram_mb": int(parts[1].strip()),
                    })
            return {"gpus": gpus, "count": len(gpus)}
    except Exception:
        pass
    return {"gpus": [], "count": 0}


@dataclass
class ExperimentManifest:
    """
    Comprehensive experiment metadata for reproducibility.

    Captures all information needed to reproduce an experiment and
    track results for paper writing.
    """

    # Identification
    experiment_id: str = ""
    experiment_name: str = ""

    # Git state
    git_hash: str = ""
    git_dirty: bool = False
    git_branch: str = ""

    # Execution info
    command: str = ""
    python_version: str = ""
    working_dir: str = ""

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # System info
    hostname: str = ""
    gpu_info: Dict[str, Any] = field(default_factory=dict)

    # Config (stored as dict for JSON serialization)
    config: Dict[str, Any] = field(default_factory=dict)

    # Results
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running, completed, failed

    # Paths
    model_save_path: str = ""
    log_path: str = ""

    @classmethod
    def create(
        cls,
        cfg: DictConfig,
        experiment_name: str,
        model_save_path: str,
    ) -> "ExperimentManifest":
        """
        Create a new experiment manifest at the start of training.

        Args:
            cfg: Hydra config
            experiment_name: Name of the experiment
            model_save_path: Path where model checkpoints are saved

        Returns:
            Initialized ExperimentManifest
        """
        git_info = get_git_info()
        gpu_info = get_gpu_info()

        return cls(
            experiment_id=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            experiment_name=experiment_name,
            git_hash=git_info["hash"],
            git_dirty=git_info["dirty"],
            git_branch=git_info["branch"],
            command=" ".join(sys.argv),
            python_version=sys.version,
            working_dir=os.getcwd(),
            start_time=datetime.now().isoformat(),
            hostname=socket.gethostname(),
            gpu_info=gpu_info,
            config=OmegaConf.to_container(cfg, resolve=True),
            model_save_path=model_save_path,
            status="running",
        )

    def complete(
        self,
        metrics: Dict[str, Any],
        status: str = "completed",
    ) -> None:
        """
        Mark experiment as complete with final metrics.

        Args:
            metrics: Final metrics dict (best_eer, best_loss, epochs, etc.)
            status: Final status (completed, failed)
        """
        self.end_time = datetime.now().isoformat()
        if self.start_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            self.duration_seconds = (end - start).total_seconds()
        self.metrics = metrics
        self.status = status

    def save(self, path: Optional[str] = None) -> str:
        """
        Save manifest to JSON file.

        Args:
            path: Optional path, defaults to model_save_path/manifest.json

        Returns:
            Path where manifest was saved
        """
        if path is None:
            path = os.path.join(self.model_save_path, "manifest.json")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

        return path

    @classmethod
    def load(cls, path: str) -> "ExperimentManifest":
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"ID: {self.experiment_id}",
            f"Status: {self.status}",
            f"Git: {self.git_hash[:8]}{'*' if self.git_dirty else ''} ({self.git_branch})",
            f"Host: {self.hostname}",
        ]

        if self.gpu_info.get("gpus"):
            gpu = self.gpu_info["gpus"][0]
            lines.append(f"GPU: {gpu['name']} ({gpu['vram_mb']}MB)")

        if self.duration_seconds:
            hours = self.duration_seconds / 3600
            lines.append(f"Duration: {hours:.2f}h")

        if self.metrics:
            if "best_eer" in self.metrics:
                lines.append(f"Best EER: {self.metrics['best_eer']*100:.2f}%")
            if "best_val_loss" in self.metrics:
                lines.append(f"Best Val Loss: {self.metrics['best_val_loss']:.6f}")

        return "\n".join(lines)


@dataclass
class TrainingResult:
    """
    Result of a training run.

    Used to pass results between training function and callers (e.g., Optuna).
    """

    best_eer: Optional[float] = None
    best_val_loss: float = float("inf")
    final_train_loss: float = float("inf")
    epochs_completed: int = 0
    best_epoch: int = 0
    checkpoints: list = field(default_factory=list)
    manifest: Optional[ExperimentManifest] = None

    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert to metrics dict for manifest."""
        return {
            "best_eer": self.best_eer,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.final_train_loss,
            "epochs_completed": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "num_checkpoints": len(self.checkpoints),
        }
