"""Training loop.

Ties together a dataset, a composition-aware split, a model implementing
:class:`~vacancy_gnn.models.base.EnergyModel`, an experiment tracker, and
checkpointing (PLAN.md step 4). Kept model-agnostic: the same loop trains the
linear baseline today and the equivariant GNN once it lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vacancy_gnn.data.featurize import Graph, build_graph
from vacancy_gnn.data.schema import Dataset
from vacancy_gnn.data.splits import Split, composition_split
from vacancy_gnn.metrics import mae, rmse
from vacancy_gnn.models.base import EnergyModel
from vacancy_gnn.tracking import NullTracker, Tracker


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a training run."""

    val_mae: float
    val_rmse: float
    n_train: int
    n_val: int
    checkpoint: Path | None


def _graphs_and_energies(
    dataset: Dataset, indices: list[int], cutoff: float
) -> tuple[list[Graph], NDArray[np.float64]]:
    graphs = [build_graph(dataset.arrangements[i], cutoff=cutoff) for i in indices]
    energies = np.array(
        [dataset.arrangements[i].energy_ev for i in indices], dtype=np.float64
    )
    return graphs, energies


def train(
    model: EnergyModel,
    dataset: Dataset,
    *,
    cutoff: float = 5.0,
    split: Split | None = None,
    tracker: Tracker | None = None,
    checkpoint_dir: Path | None = None,
    seed: int = 0,
) -> TrainResult:
    """Fit ``model`` on ``dataset`` and evaluate on the held-out validation split.

    Args:
        model: Any object implementing :class:`EnergyModel`.
        dataset: The labeled dataset.
        cutoff: Edge distance cutoff for graph construction.
        split: A precomputed split; if ``None``, a composition-aware split is made.
        tracker: Experiment tracker; defaults to a :class:`NullTracker`.
        checkpoint_dir: If given, the fitted model is saved here.
        seed: Seed used for the default split.

    Returns:
        A :class:`TrainResult` with validation metrics and the checkpoint path.
    """
    tracker = tracker or NullTracker()
    split = split or composition_split(dataset, seed=seed)

    if len(split.train) == 0:
        raise ValueError("training split is empty")

    train_graphs, train_e = _graphs_and_energies(dataset, split.train, cutoff)
    val_graphs, val_e = _graphs_and_energies(dataset, split.val, cutoff)

    tracker.log_params(
        {
            "model": type(model).__name__,
            "cutoff": cutoff,
            "seed": seed,
            "n_train": len(split.train),
            "n_val": len(split.val),
        }
    )

    model.fit(train_graphs, train_e)

    if len(val_graphs) > 0:
        val_pred = model.predict(val_graphs)
        v_mae, v_rmse = mae(val_e, val_pred), rmse(val_e, val_pred)
    else:
        v_mae, v_rmse = float("nan"), float("nan")

    tracker.log_metrics({"val_mae": v_mae, "val_rmse": v_rmse}, step=0)

    checkpoint: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / "model.json"
        model.save(checkpoint)

    tracker.finish()
    return TrainResult(
        val_mae=v_mae,
        val_rmse=v_rmse,
        n_train=len(split.train),
        n_val=len(split.val),
        checkpoint=checkpoint,
    )
