"""Local-environment graph construction for a vacancy arrangement.

The featurizer turns an :class:`~vacancy_gnn.data.schema.Arrangement` into a graph
whose nodes are cations and whose edges connect cations within a distance cutoff.
Node and edge features are built so that any downstream energy readout is invariant
to global rotation, translation, and cation-index permutation, which is the
physically required symmetry of a total energy (PLAN.md Section 6).

This module is deliberately framework-free (numpy only): it produces plain arrays
that either the equivariant GNN or the linear baseline can consume, and it is the
target of the equivariance/permutation unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vacancy_gnn.data.schema import Arrangement


@dataclass(frozen=True)
class Graph:
    """A featurized arrangement.

    Attributes:
        node_z: Atomic number per cation node, shape ``(n_nodes,)``.
        node_vacancy_count: Number of vacant oxygen sites within ``cutoff`` of each
            cation, shape ``(n_nodes,)``. This is the invariant scalar that carries
            the vacancy information into the model.
        edge_index: Directed edges as a ``(2, n_edges)`` int array.
        edge_vec: Cartesian displacement per edge, shape ``(n_edges, 3)``. Rotates
            with the structure; used by the equivariant model. The baseline uses
            only its norm.
        edge_dist: Edge lengths, shape ``(n_edges,)``. Rotation/translation
            invariant.
    """

    node_z: NDArray[np.int64]
    node_vacancy_count: NDArray[np.int64]
    edge_index: NDArray[np.int64]
    edge_vec: NDArray[np.float64]
    edge_dist: NDArray[np.float64]

    @property
    def n_nodes(self) -> int:
        return int(self.node_z.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def _pairwise_distances(pos: NDArray[np.float64]) -> NDArray[np.float64]:
    diff = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_graph(
    arrangement: Arrangement,
    cutoff: float,
    vacancy_positions: NDArray[np.float64] | None = None,
    vacancy_cutoff: float | None = None,
) -> Graph:
    """Build a local-environment graph from an arrangement.

    Args:
        arrangement: The labeled arrangement to featurize.
        cutoff: Cation-cation edge distance cutoff (same length unit as positions).
        vacancy_positions: Optional ``(n_o_sites, 3)`` coordinates of the oxygen
            sublattice, indexed to match ``arrangement.vacancy_sites``. When given,
            each cation gets a count of nearby vacancies as a node feature.
        vacancy_cutoff: Distance cutoff for counting nearby vacancies; defaults to
            ``cutoff`` when not provided.

    Returns:
        A :class:`Graph`.

    Raises:
        ValueError: If ``cutoff`` is not positive, or a referenced vacancy site is
            out of range for ``vacancy_positions``.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    pos = arrangement.positions_array()
    z = arrangement.species_array()
    n = pos.shape[0]

    dmat = _pairwise_distances(pos)
    # Directed edges within cutoff, excluding self-loops.
    mask = (dmat <= cutoff) & ~np.eye(n, dtype=bool)
    src, dst = np.nonzero(mask)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_vec = (pos[dst] - pos[src]).astype(np.float64)
    edge_dist = dmat[src, dst].astype(np.float64)

    node_vacancy_count = _vacancy_counts(
        pos, arrangement.vacancy_sites, vacancy_positions, vacancy_cutoff or cutoff
    )

    return Graph(
        node_z=z,
        node_vacancy_count=node_vacancy_count,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
    )


def _vacancy_counts(
    cation_pos: NDArray[np.float64],
    vacancy_sites: list[int],
    vacancy_positions: NDArray[np.float64] | None,
    vacancy_cutoff: float,
) -> NDArray[np.int64]:
    n = cation_pos.shape[0]
    if vacancy_positions is None or len(vacancy_sites) == 0:
        return np.zeros(n, dtype=np.int64)

    n_sites = vacancy_positions.shape[0]
    if any(s >= n_sites for s in vacancy_sites):
        raise ValueError("a vacancy site index is out of range for vacancy_positions")

    vac_pos = vacancy_positions[np.asarray(vacancy_sites, dtype=np.int64)]
    diff = cation_pos[:, None, :] - vac_pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return (dist <= vacancy_cutoff).sum(axis=1).astype(np.int64)
