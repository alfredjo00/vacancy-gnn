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

With few training compositions relative to the number of species, the design
matrix can be rank-deficient or merely poorly conditioned along directions a held-
out composition's counts happen to stick out along (see IMPROVEMENTS.md P8): the
fit is then free to wander arbitrarily off training data outside the training
composition hull. ``fit`` optionally shrinks the solution toward a physically
motivated ``prior`` (see :func:`prior_from_e0s`) instead of the unconstrained
minimum-norm solution, so data determines the coefficients along directions it
constrains and the prior fills in the rest.

Framework-free (numpy only) so both the torch GNN and the linear baseline share
one implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vacancy_gnn.data.descriptors import DESCRIPTOR_SPECIES
from vacancy_gnn.data.featurize import VACANCY_MARKER_Z, Graph

#: Atomic-number -> element-symbol map for :data:`DESCRIPTOR_SPECIES` (excluding
#: the vacancy marker), used by :func:`prior_from_e0s` to read a symbol-keyed E0
#: table. Kept local to this module; nothing else needs a Z<->symbol mapping.
_SYMBOL_BY_Z: dict[int, str] = {
    3: "Li",
    12: "Mg",
    13: "Al",
    20: "Ca",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    40: "Zr",
    49: "In",
    50: "Sn",
}


def _species_counts(graph: Graph) -> NDArray[np.float64]:
    """Per-species node counts over :data:`DESCRIPTOR_SPECIES` (markers included)."""
    idx = {z: i for i, z in enumerate(DESCRIPTOR_SPECIES)}
    counts = np.zeros(len(DESCRIPTOR_SPECIES), dtype=np.float64)
    for z in graph.node_z.tolist():
        if z not in idx:
            raise ValueError(f"species {z} not in DESCRIPTOR_SPECIES")
        counts[idx[z]] += 1.0
    return counts


def prior_from_e0s(
    e0s_ev: dict[str, float], n_cations: int, n_oxygen_sites: int
) -> NDArray[np.float64]:
    """Build a :meth:`CompositionReference.fit` prior from isolated-atom energies.

    The reference has no oxygen column: occupied oxygen sites are not graph
    nodes, only vacancy markers are, so the oxygen count ``n_oxygen_sites - v``
    never appears explicitly. It is, however, an affine function of the constant
    node count and the marker count, so a per-element reference energy
    ``sum_cations E0_cation + (n_oxygen_sites - v) * E0_O`` can be reproduced
    exactly in this basis by folding the (constant, per-cell) oxygen contribution
    into every cation coefficient and the ``-v`` part into the marker
    coefficient:

    - ``prior[cation z] = E0_z + (n_oxygen_sites / n_cations) * E0_O``
    - ``prior[marker]   = -E0_O``

    Every arrangement has exactly ``n_cations`` cation sites occupied, so
    amortizing the oxygen term equally over them reproduces the total exactly:
    ``sum_cations prior[z] + v * prior[marker]``
    ``= sum_cations E0_z + n_cations * (n_oxygen_sites / n_cations) * E0_O - v * E0_O``
    ``= sum_cations E0_z + (n_oxygen_sites - v) * E0_O``.

    The marker direction (vacancy count) varies within every composition, so it
    is always well constrained by data; its prior value has negligible effect on
    the fit and mainly matters for the ``shrinkage -> inf`` limit.

    Args:
        e0s_ev: Isolated-atom reference energy (eV) per element symbol. Must
            contain ``"O"`` and every cation species present in
            :data:`~vacancy_gnn.data.descriptors.DESCRIPTOR_SPECIES`.
        n_cations: Number of cation sites per arrangement (e.g. 36 for the
            3x2x1 spinel supercell).
        n_oxygen_sites: Number of ideal oxygen sublattice sites per arrangement
            (e.g. 48 for the same supercell).

    Returns:
        A ``(len(DESCRIPTOR_SPECIES),)`` prior vector, ordered like
        :data:`~vacancy_gnn.data.descriptors.DESCRIPTOR_SPECIES`.

    Raises:
        ValueError: If ``e0s_ev`` is missing ``"O"`` or a required cation
            symbol, or if ``n_cations`` is not positive.
    """
    if n_cations <= 0:
        raise ValueError(f"n_cations must be positive, got {n_cations}")
    if "O" not in e0s_ev:
        raise ValueError("e0s_ev must contain an 'O' entry")
    e0_o = e0s_ev["O"]

    prior = np.zeros(len(DESCRIPTOR_SPECIES), dtype=np.float64)
    for i, z in enumerate(DESCRIPTOR_SPECIES):
        if z == VACANCY_MARKER_Z:
            prior[i] = -e0_o
            continue
        symbol = _SYMBOL_BY_Z[z]
        if symbol not in e0s_ev:
            raise ValueError(f"e0s_ev is missing an entry for {symbol!r} (Z={z})")
        prior[i] = e0s_ev[symbol] + (n_oxygen_sites / n_cations) * e0_o
    return prior


class CompositionReference:
    """Least-squares per-species energy reference, fit on training arrangements."""

    def __init__(self) -> None:
        self._coeffs: NDArray[np.float64] | None = None

    def fit(
        self,
        graphs: list[Graph],
        energies: NDArray[np.float64],
        prior: NDArray[np.float64] | None = None,
        shrinkage: float = 0.0,
    ) -> None:
        """Fit per-species energies so ``counts @ coeffs`` approximates ``energies``.

        Args:
            graphs: Training arrangements.
            energies: Total energy per graph (eV).
            prior: Optional per-species anchor (see :func:`prior_from_e0s`), same
                length and order as
                :data:`~vacancy_gnn.data.descriptors.DESCRIPTOR_SPECIES`. Ignored
                if ``shrinkage`` is 0.
            shrinkage: Ridge penalty applied toward ``prior`` (rather than
                toward zero) via augmented rows: ``min ||X b - y||^2 +
                shrinkage * ||b - prior||^2``. Directions the data constrains
                are dominated by the data term; directions it does not (e.g. a
                held-out composition's counts sticking outside the training
                hull) fall back to ``prior`` instead of an arbitrary minimum-
                norm value. ``0.0`` (the default) reproduces the plain,
                unconstrained least-squares fit exactly.

        Raises:
            ValueError: If ``graphs`` is empty, or ``shrinkage`` is nonzero
                without a ``prior``.
        """
        if len(graphs) == 0:
            raise ValueError("cannot fit reference on an empty dataset")
        if shrinkage != 0.0 and prior is None:
            raise ValueError("shrinkage requires a prior")
        x = np.stack([_species_counts(g) for g in graphs], axis=0)
        y = np.asarray(energies, dtype=np.float64).ravel()
        if shrinkage == 0.0:
            # Plain least squares; lstsq handles the rank-deficient case (species
            # absent from training get a zero coefficient via the minimum-norm
            # solution).
            coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        else:
            n_species = x.shape[1]
            lam_sqrt = np.sqrt(shrinkage)
            x_aug = np.vstack([x, lam_sqrt * np.eye(n_species)])
            y_aug = np.concatenate([y, lam_sqrt * np.asarray(prior, dtype=np.float64)])
            coeffs, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
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


def out_of_span_norm(train_graphs: list[Graph], graph: Graph) -> float:
    """Norm of ``graph``'s count vector orthogonal to the training row space.

    Quantifies how far a composition's per-species counts sit outside the span
    of the training compositions' counts: 0 means the count vector is a linear
    combination of training rows (the reference can interpolate); a large value
    means the fit has to extrapolate along a direction the training data never
    constrained (IMPROVEMENTS.md P8). Held-out error was found to track this
    quantity roughly linearly (~10 eV per unit of norm) on the real factory
    data, so it is a cheap, honest leverage flag for :mod:`vacancy_gnn.evaluate`.

    Args:
        train_graphs: Training arrangements (any composition mix).
        graph: The arrangement to score.

    Returns:
        The Euclidean norm of the component of ``graph``'s count vector that
        lies outside the row space spanned by ``train_graphs``' count vectors.
    """
    x_train = np.stack([_species_counts(g) for g in train_graphs], axis=0)
    x = _species_counts(graph)
    _, s, vt = np.linalg.svd(x_train, full_matrices=False)
    basis = vt[s > 1e-8 * s[0]] if s.size else vt[:0]
    projection = basis.T @ (basis @ x)
    return float(np.linalg.norm(x - projection))
