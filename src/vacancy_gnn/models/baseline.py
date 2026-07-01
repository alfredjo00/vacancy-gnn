"""Cluster-expansion-style linear baseline.

Ridge regression on the fixed-length invariant descriptors from
:mod:`vacancy_gnn.data.descriptors`. Closed-form, deterministic, and framework-free
(numpy only), so it needs no GPU and no torch. This is the interpretable reference
the equivariant GNN must beat (PLAN.md Section 6).

Features are standardized (zero mean, unit variance) before fitting; the ridge
penalty is applied to the standardized weights, excluding the intercept.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vacancy_gnn.data.descriptors import graph_descriptor
from vacancy_gnn.data.featurize import Graph


class LinearBaseline:
    """Ridge-regression energy model over invariant descriptors."""

    def __init__(self, regularization: float = 1e-3) -> None:
        if regularization < 0:
            raise ValueError("regularization must be >= 0")
        self.regularization = regularization
        self._weights: NDArray[np.float64] | None = None
        self._intercept: float = 0.0
        self._feature_mean: NDArray[np.float64] | None = None
        self._feature_std: NDArray[np.float64] | None = None

    def _design_matrix(self, graphs: list[Graph]) -> NDArray[np.float64]:
        return np.stack([graph_descriptor(g) for g in graphs], axis=0)

    def fit(self, graphs: list[Graph], energies: NDArray[np.float64]) -> None:
        """Fit ridge weights by the closed-form normal equations."""
        if len(graphs) == 0:
            raise ValueError("cannot fit on an empty dataset")
        x = self._design_matrix(graphs)
        y = np.asarray(energies, dtype=np.float64).ravel()
        if y.shape[0] != x.shape[0]:
            raise ValueError("number of energies must match number of graphs")

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0.0] = 1.0  # constant features carry no signal; avoid /0
        xs = (x - mean) / std

        n_features = xs.shape[1]
        a = xs.T @ xs + self.regularization * np.eye(n_features)
        b = xs.T @ (y - y.mean())
        weights = np.linalg.solve(a, b)

        self._weights = weights
        self._intercept = float(y.mean())
        self._feature_mean = mean
        self._feature_std = std

    def predict(self, graphs: list[Graph]) -> NDArray[np.float64]:
        """Predict energies for a list of arrangements."""
        if (
            self._weights is None
            or self._feature_mean is None
            or self._feature_std is None
        ):
            raise RuntimeError("model is not fitted; call fit() first")
        x = self._design_matrix(graphs)
        xs = (x - self._feature_mean) / self._feature_std
        result: NDArray[np.float64] = xs @ self._weights + self._intercept
        return result

    def save(self, path: Path) -> None:
        """Persist weights and standardization to a JSON file."""
        if (
            self._weights is None
            or self._feature_mean is None
            or self._feature_std is None
        ):
            raise RuntimeError("model is not fitted; call fit() first")
        payload = {
            "regularization": self.regularization,
            "weights": self._weights.tolist(),
            "intercept": self._intercept,
            "feature_mean": self._feature_mean.tolist(),
            "feature_std": self._feature_std.tolist(),
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> LinearBaseline:
        """Load a baseline previously written by :meth:`save`."""
        payload = json.loads(path.read_text())
        model = cls(regularization=payload["regularization"])
        model._weights = np.asarray(payload["weights"], dtype=np.float64)
        model._intercept = float(payload["intercept"])
        model._feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
        model._feature_std = np.asarray(payload["feature_std"], dtype=np.float64)
        return model
