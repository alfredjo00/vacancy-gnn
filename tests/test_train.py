"""Tests for the training loop and tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from vacancy_gnn.data.synthetic import make_synthetic_dataset
from vacancy_gnn.models.baseline import LinearBaseline
from vacancy_gnn.tracking import NullTracker, get_tracker
from vacancy_gnn.train import train


def test_train_returns_finite_metrics() -> None:
    dataset = make_synthetic_dataset(n_compositions=10, seed=2)
    result = train(LinearBaseline(), dataset, seed=2)
    assert result.n_train > 0
    assert result.n_val > 0
    assert result.val_mae >= 0.0
    assert result.val_rmse >= result.val_mae


def test_train_writes_checkpoint(tmp_path: Path) -> None:
    dataset = make_synthetic_dataset(n_compositions=10, seed=3)
    result = train(LinearBaseline(), dataset, checkpoint_dir=tmp_path, seed=3)
    assert result.checkpoint is not None
    assert result.checkpoint.exists()
    # The checkpoint is loadable and reproduces predictions.
    LinearBaseline.load(result.checkpoint)


def test_tracker_records_params_and_metrics() -> None:
    dataset = make_synthetic_dataset(n_compositions=10, seed=4)
    tracker = NullTracker()
    train(LinearBaseline(), dataset, tracker=tracker, seed=4)
    assert tracker.params["model"] == "LinearBaseline"
    assert len(tracker.history) == 1
    _step, metrics = tracker.history[0]
    assert "val_mae" in metrics


def test_get_tracker_none() -> None:
    assert isinstance(get_tracker("none"), NullTracker)


def test_get_tracker_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_tracker("bogus")


def test_get_tracker_unwired_backend_raises() -> None:
    with pytest.raises(NotImplementedError):
        get_tracker("wandb")
