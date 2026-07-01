"""Fixed-length invariant descriptors for the cluster-expansion baseline.

The equivariant GNN consumes the raw :class:`~vacancy_gnn.data.featurize.Graph`
directly, but the linear baseline needs a fixed-length feature vector. This module
pools the graph into such a vector using only rotation-, translation-, and
permutation-invariant quantities, so the baseline respects the same physical
symmetry as the total energy (PLAN.md Section 6).

The representation is a small, interpretable cluster-expansion-style descriptor:
per-element counts, element-pair bond counts within the cutoff, and vacancy-count
aggregates. It is deliberately transparent so the baseline is a meaningful, legible
reference the GNN must beat.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vacancy_gnn.data.featurize import Graph

#: Cation atomic numbers the descriptor is defined over (the FeMnAl family plus
#: common spinel B-site cations). Fixed so descriptor length is constant.
DESCRIPTOR_SPECIES: tuple[int, ...] = (3, 12, 13, 22, 23, 24, 25, 26, 27, 28, 29, 30)


def descriptor_length() -> int:
    """Length of the descriptor vector produced by :func:`graph_descriptor`."""
    n = len(DESCRIPTOR_SPECIES)
    n_pairs = n * (n + 1) // 2  # unordered element pairs, with self-pairs
    # per-element counts + element-pair bond counts + 3 vacancy aggregates
    return n + n_pairs + 3


def _species_index() -> dict[int, int]:
    return {z: i for i, z in enumerate(DESCRIPTOR_SPECIES)}


def graph_descriptor(graph: Graph) -> NDArray[np.float64]:
    """Pool a graph into a fixed-length invariant descriptor vector.

    Args:
        graph: The featurized arrangement.

    Returns:
        A ``(descriptor_length(),)`` float vector.

    Raises:
        ValueError: If the graph contains a species outside
            :data:`DESCRIPTOR_SPECIES`.
    """
    idx = _species_index()
    n_species = len(DESCRIPTOR_SPECIES)

    # Per-element node counts.
    counts = np.zeros(n_species, dtype=np.float64)
    node_slot = np.empty(graph.n_nodes, dtype=np.int64)
    for node, z in enumerate(graph.node_z.tolist()):
        if z not in idx:
            raise ValueError(f"species {z} not in DESCRIPTOR_SPECIES")
        node_slot[node] = idx[z]
        counts[idx[z]] += 1.0

    # Element-pair bond counts within the cutoff (undirected: halve directed edges).
    pair_index = _pair_index_map(n_species)
    pair_counts = np.zeros(len(pair_index), dtype=np.float64)
    for e in range(graph.n_edges):
        i, j = int(graph.edge_index[0, e]), int(graph.edge_index[1, e])
        a, b = int(node_slot[i]), int(node_slot[j])
        key = (a, b) if a <= b else (b, a)
        pair_counts[pair_index[key]] += 0.5

    # Vacancy aggregates: total, mean per node, max per node.
    vac = graph.node_vacancy_count.astype(np.float64)
    vac_aggr = np.array(
        [vac.sum(), vac.mean() if vac.size else 0.0, vac.max() if vac.size else 0.0]
    )

    return np.concatenate([counts, pair_counts, vac_aggr])


def _pair_index_map(n_species: int) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    k = 0
    for a in range(n_species):
        for b in range(a, n_species):
            mapping[(a, b)] = k
            k += 1
    return mapping
