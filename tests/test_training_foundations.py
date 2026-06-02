import random

import numpy as np
import pytest

from mimodf.training.checkpoint import TopKCheckpointTracker, checkpoint_filename
from mimodf.training.seeding import (
    SeedSettings,
    dataloader_worker_seed,
    seed_dataloader_worker,
    seed_everything,
)


def test_dataloader_worker_seed_is_stable_32_bit():
    assert dataloader_worker_seed(100, 0) == 100
    assert dataloader_worker_seed(100, 7) == 107
    assert dataloader_worker_seed(2**32 - 1, 2) == 1

    with pytest.raises(ValueError, match="worker_id"):
        dataloader_worker_seed(1, -1)


def test_seed_dataloader_worker_seeds_python_and_numpy():
    seed = seed_dataloader_worker(123, 4)
    first_random = random.random()
    first_numpy = np.random.rand()

    assert seed == 127
    seed_dataloader_worker(123, 4)
    assert random.random() == first_random
    assert np.random.rand() == first_numpy


def test_seed_everything_seeds_python_and_numpy_without_requiring_torch():
    seed_everything(SeedSettings(seed=55))
    first_random = random.random()
    first_numpy = np.random.rand()

    seed_everything(SeedSettings(seed=55))
    assert random.random() == first_random
    assert np.random.rand() == first_numpy


def test_checkpoint_filename_is_stable():
    assert checkpoint_filename(3, "val eer", 0.123456) == "epoch_3_val_eer_0.1235.pth"

    with pytest.raises(ValueError, match="metric_name"):
        checkpoint_filename(1, " ", 0.1)


def test_top_k_checkpoint_tracker_saves_best_and_deletes_replaced(tmp_path):
    saved = []
    deleted = []

    def save_fn(state, path):
        saved.append((state, path.name))
        path.write_text(str(state))

    def delete_fn(path):
        deleted.append(path.name)
        path.unlink(missing_ok=True)

    tracker = TopKCheckpointTracker(tmp_path, k=2, save_fn=save_fn, delete_fn=delete_fn)

    assert (
        tracker.consider(state="a", epoch=1, metric=0.30, val_loss=1.0, metric_name="eer")
        is not None
    )
    assert (
        tracker.consider(state="b", epoch=2, metric=0.20, val_loss=0.9, metric_name="eer")
        is not None
    )
    assert (
        tracker.consider(state="c", epoch=3, metric=0.40, val_loss=0.8, metric_name="eer") is None
    )
    assert (
        tracker.consider(state="d", epoch=4, metric=0.10, val_loss=0.7, metric_name="eer")
        is not None
    )

    assert [record.epoch for record in tracker.records] == [4, 2]
    assert tracker.best is not None
    assert tracker.best.epoch == 4
    assert saved == [
        ("a", "epoch_1_eer_0.3000.pth"),
        ("b", "epoch_2_eer_0.2000.pth"),
        ("d", "epoch_4_eer_0.1000.pth"),
    ]
    assert deleted == ["epoch_1_eer_0.3000.pth"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "epoch_2_eer_0.2000.pth",
        "epoch_4_eer_0.1000.pth",
    ]


def test_top_k_checkpoint_tracker_rejects_invalid_k(tmp_path):
    with pytest.raises(ValueError, match="k must be"):
        TopKCheckpointTracker(tmp_path, k=0, save_fn=lambda state, path: None)
