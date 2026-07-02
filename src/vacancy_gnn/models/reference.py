"""Per-species linear energy reference.

Total energies differ across compositions by tens of eV for trivial reasons: the
per-element atomic reference terms. The quantity the models actually need to learn
is the per-arrangement *difference* within a composition (0.1-1 eV), so fitting
those trivial offsets wastes capacity and optimization signal. This module fits a
linear-in-composition reference ``E_ref = sum_species count_species * e_species``
by least squares on the training set; models train on the residual ``E - E_ref``
and add it back at predict time. This is standard MLIP practice and a large
data-efficiency win at small dataset sizes (PLAN.md Section 6).

The vacancy marker is one of the species, so a change in vacancy count shifts the
reference through the marker's fitted coefficient; the residual is therefore also
free of the leading linear-in-``v`` trend.

Framework-free (numpy only) so both the torch GNN and the linear baseline share
one implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vacancy_gnn.data.descriptors import DESCRIPTOR_SPECIES
from vacancy_gnn.data.featurize import Graph


def _species_counts(graph: Graph) -> NDArray[np.float64]:
    """Per-species node counts over :data:`DESCRIPTOR_SPECIES` (markers included)."""
    idx = {z: i for i, z in enumerate(DESCRIPTOR_SPECIES)}
    counts = np.zeros(len(DESCRIPTOR_SPECIES), dtype=np.float64)
    for z in graph.node_z.tolist():
        if z not in idx:
            raise ValueError(f"species {z} not in DESCRIPTOR_SPECIES")
        counts[idx[z]] += 1.0
    return counts


class CompositionReference:
    """Least-squares per-species energy reference, fit on training arrangements."""

    def __init__(self) -> None:
        self._coeffs: NDArray[np.float64] | None = None

    def fit(self, graphs: list[Graph], energies: NDArray[np.float64]) -> None:
        """Fit per-species energies so ``counts @ coeffs`` approximates ``energies``."""
        if len(graphs) == 0:
            raise ValueError("cannot fit reference on an empty dataset")
        x = np.stack([_species_counts(g) for g in graphs], axis=0)
        y = np.asarray(energies, dtype=np.float64).ravel()
        # Least squares; lstsq handles the rank-deficient case (species absent
        # from the training set get a zero coefficient via the minimum-norm soln).
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        self._coeffs = coeffs

    def predict(self, graphs: list[Graph]) -> NDArray[np.float64]:
        """Reference energy per graph."""
        if self._coeffs is None:
            raise RuntimeError("reference is not fitted; call fit() first")
        x = np.stack([_species_counts(g) for g in graphs], axis=0)
        result: NDArray[np.float64] = x @ self._coeffs
        return result

    @property
    def coeffs(self) -> NDArray[np.float64]:
        if self._coeffs is None:
            raise RuntimeError("reference is not fitted; call fit() first")
        return self._coeffs

    def to_list(self) -> list[float]:
        """Serialize coefficients for model checkpoints."""
        coeffs: list[float] = self.coeffs.tolist()
        return coeffs

    @classmethod
    def from_list(cls, coeffs: list[float]) -> CompositionReference:
        """Rebuild from :meth:`to_list` output."""
        ref = cls()
        ref._coeffs = np.asarray(coeffs, dtype=np.float64)
        return ref
