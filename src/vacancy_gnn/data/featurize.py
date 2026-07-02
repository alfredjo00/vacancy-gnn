"""Local-environment graph construction for a vacancy arrangement.

The featurizer turns an :class:`~vacancy_gnn.data.schema.Arrangement` into a graph
whose nodes are the cations *and* a marker node at each vacant oxygen site, with
edges connecting nodes within a distance cutoff under the minimum-image
convention. Representing vacancies as their own nodes (rather than a per-cation
count) is what lets the model see *which* cations surround each vacancy and how
vacancies sit relative to one another, which is the physical signal the project
exists to capture (PLAN.md Sections 2 and 6).

Node and edge features are built so that any downstream energy readout is
invariant to global rotation, translation, and node permutation, which is the
physically required symmetry of a total energy. Edges use minimum-image
displacements from ``arrangement.cell`` so a periodic supercell is featurized
without boundary-truncation artifacts.

This module is deliberately framework-free (numpy only): it produces plain arrays
that either the equivariant GNN or the linear baseline can consume, and it is the
target of the equivariance/permutation unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vacancy_gnn.data.schema import Arrangement

#: Reserved pseudo-species (atomic number) used for vacancy-marker nodes. Zero is
#: not a real element, so it never collides with a cation species and can carry
#: its own learned embedding row / descriptor slot.
VACANCY_MARKER_Z = 0


@dataclass(frozen=True)
class Graph:
    """A featurized arrangement.

    Nodes are cations followed by one marker per vacant oxygen site; a node is a
    vacancy marker iff ``node_z == VACANCY_MARKER_Z``.

    Attributes:
        node_z: Atomic number per node, shape ``(n_nodes,)``. Cation nodes carry
            their real atomic number; vacancy-marker nodes carry
            :data:`VACANCY_MARKER_Z`.
        node_is_vacancy: Boolean mask, ``True`` for vacancy-marker nodes, shape
            ``(n_nodes,)``.
        edge_index: Directed edges as a ``(2, n_edges)`` int array.
        edge_vec: Minimum-image Cartesian displacement per edge, shape
            ``(n_edges, 3)``. Rotates with the structure; used by the equivariant
            model. The baseline uses only its norm.
        edge_dist: Edge lengths, shape ``(n_edges,)``. Rotation/translation
            invariant.
    """

    node_z: NDArray[np.int64]
    node_is_vacancy: NDArray[np.bool_]
    edge_index: NDArray[np.int64]
    edge_vec: NDArray[np.float64]
    edge_dist: NDArray[np.float64]

    @property
    def n_nodes(self) -> int:
        return int(self.node_z.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def _minimum_image_displacements(
    pos: NDArray[np.float64], cell: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pairwise displacements ``pos[j] - pos[i]`` under the minimum-image convention.

    Returns an ``(n, n, 3)`` array; entry ``[i, j]`` is the shortest displacement
    from node ``i`` to node ``j`` across periodic images of ``cell``.
    """
    diff = pos[None, :, :] - pos[:, None, :]  # (n, n, 3): pos[j] - pos[i]
    # Wrap each displacement into the primitive cell: subtract the nearest lattice
    # translation. frac = diff @ inv(cell); diff -= round(frac) @ cell.
    inv_cell = np.linalg.inv(cell)
    frac = diff @ inv_cell
    frac -= np.round(frac)
    return frac @ cell


def build_graph(arrangement: Arrangement, cutoff: float) -> Graph:
    """Build a local-environment graph from an arrangement.

    Args:
        arrangement: The labeled arrangement to featurize.
        cutoff: Node-node edge distance cutoff (same length unit as positions).

    Returns:
        A :class:`Graph` whose nodes are the cations followed by one marker per
        vacant oxygen site, with minimum-image edges within ``cutoff``.

    Raises:
        ValueError: If ``cutoff`` is not positive.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    cation_pos = arrangement.positions_array()
    cation_z = arrangement.species_array()

    oxygen_pos = arrangement.oxygen_positions_array()
    vac_idx = np.asarray(arrangement.vacancy_sites, dtype=np.int64)
    vac_pos = (
        oxygen_pos[vac_idx] if vac_idx.size else np.empty((0, 3), dtype=np.float64)
    )

    pos = np.concatenate([cation_pos, vac_pos], axis=0)
    node_z = np.concatenate(
        [cation_z, np.full(vac_pos.shape[0], VACANCY_MARKER_Z, dtype=np.int64)]
    )
    node_is_vacancy = np.concatenate(
        [
            np.zeros(cation_pos.shape[0], dtype=np.bool_),
            np.ones(vac_pos.shape[0], dtype=np.bool_),
        ]
    )

    cell = arrangement.cell_array()
    disp = _minimum_image_displacements(pos, cell)  # (n, n, 3), disp[i, j]
    dmat = np.linalg.norm(disp, axis=-1)

    n = pos.shape[0]
    mask = (dmat <= cutoff) & ~np.eye(n, dtype=bool)
    src, dst = np.nonzero(mask)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_vec = disp[src, dst].astype(np.float64)
    edge_dist = dmat[src, dst].astype(np.float64)

    return Graph(
        node_z=node_z,
        node_is_vacancy=node_is_vacancy,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
    )
