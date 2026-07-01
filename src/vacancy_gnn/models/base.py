"""Common model interface.

Both the cluster-expansion baseline and the equivariant GNN implement this
protocol so training, evaluation, and inference code is model-agnostic
(PLAN.md Section 4). Models predict a per-arrangement energy from a featurized
:class:`~vacancy_gnn.data.featurize.Graph`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from vacancy_gnn.data.featurize import Graph


@runtime_checkable
class EnergyModel(Protocol):
    """A model mapping featurized arrangements to per-arrangement energies (eV)."""

    def fit(
        self,
        graphs: list[Graph],
        energies: NDArray[np.float64],
    ) -> None:
        """Fit the model to labeled arrangements."""
        ...

    def predict(self, graphs: list[Graph]) -> NDArray[np.float64]:
        """Predict energies (eV) for a list of arrangements."""
        ...

    def save(self, path: Path) -> None:
        """Persist the fitted model to ``path``."""
        ...

    @classmethod
    def load(cls, path: Path) -> EnergyModel:
        """Load a model previously written by :meth:`save`."""
        ...
