"""Deterministic seeding helpers for future training runs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeedSettings:
    seed: int
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True


def seed_everything(settings: SeedSettings) -> None:
    """Seed Python, NumPy, and Torch if Torch is installed."""

    random.seed(settings.seed)
    np.random.seed(settings.seed)

    torch = _try_import_torch()
    if torch is None:
        return

    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(settings.seed)
    torch.backends.cudnn.deterministic = settings.cudnn_deterministic
    torch.backends.cudnn.benchmark = settings.cudnn_benchmark


def dataloader_worker_seed(base_seed: int, worker_id: int) -> int:
    """Return a stable 32-bit seed for a DataLoader worker."""

    if worker_id < 0:
        raise ValueError("worker_id must be non-negative")
    return (int(base_seed) + int(worker_id)) % 2**32


def seed_dataloader_worker(base_seed: int, worker_id: int) -> int:
    """Seed Python/NumPy and return the worker seed used.

    This function is easy to wrap for PyTorch:

    ```python
    worker_init_fn=lambda worker_id: seed_dataloader_worker(seed, worker_id)
    ```
    """

    seed = dataloader_worker_seed(base_seed, worker_id)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def torch_generator(seed: int) -> Any:
    """Create a seeded Torch generator for deterministic DataLoader shuffling."""

    torch = _try_import_torch()
    if torch is None:
        raise RuntimeError("Torch is required to create a torch.Generator")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch
