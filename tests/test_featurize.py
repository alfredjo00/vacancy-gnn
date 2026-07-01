"""Featurization tests, including the symmetry properties a total energy requires.

The standout tests here assert that the *invariant* graph features (distances,
vacancy counts, sorted degree) do not change under global rotation, translation,
or cation-index permutation, and that the *equivariant* edge vectors rotate with
the structure. A downstream energy readout built on these features therefore
inherits the correct physical symmetry (PLAN.md Section 6).
"""

from __future__ import annotations

import numpy as np

from vacancy_gnn.data.featurize import build_graph
from vacancy_gnn.data.schema import Arrangement

from .conftest import ArrangementFactory


def _rotation_matrix(seed: int = 0) -> np.ndarray:
    # A proper rotation via QR of a random matrix (det +1 enforced).
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _with_positions(a: Arrangement, pos: np.ndarray) -> Arrangement:
    return a.model_copy(update={"cation_positions": pos.tolist()})


def test_edge_distances_invariant_under_rotation_and_translation(
    make_arrangement: ArrangementFactory,
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0, 1], seed=3)
    g = build_graph(a, cutoff=5.0)

    rot = _rotation_matrix(seed=1)
    shift = np.array([10.0, -3.0, 7.0])
    pos2 = a.positions_array() @ rot.T + shift
    g2 = build_graph(_with_positions(a, pos2), cutoff=5.0)

    assert g.n_edges == g2.n_edges
    np.testing.assert_allclose(np.sort(g.edge_dist), np.sort(g2.edge_dist), atol=1e-9)


def test_edge_vectors_rotate_with_structure(
    make_arrangement: ArrangementFactory,
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0], seed=4)
    g = build_graph(a, cutoff=5.0)

    rot = _rotation_matrix(seed=2)
    pos2 = a.positions_array() @ rot.T
    g2 = build_graph(_with_positions(a, pos2), cutoff=5.0)

    # Same connectivity, and edge vectors of g2 equal rotated edge vectors of g.
    np.testing.assert_array_equal(g.edge_index, g2.edge_index)
    np.testing.assert_allclose(g.edge_vec @ rot.T, g2.edge_vec, atol=1e-9)


def test_invariant_features_unchanged_under_permutation(
    make_arrangement: ArrangementFactory,
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0, 2], seed=5)
    g = build_graph(a, cutoff=5.0)

    perm = np.array([2, 0, 3, 1])
    pos_p = a.positions_array()[perm]
    species_p = a.species_array()[perm].tolist()
    a_p = a.model_copy(
        update={
            "cation_positions": pos_p.tolist(),
            "cation_species": species_p,
        }
    )
    g_p = build_graph(a_p, cutoff=5.0)

    # Multiset of edge distances and node degrees are permutation invariant.
    np.testing.assert_allclose(np.sort(g.edge_dist), np.sort(g_p.edge_dist), atol=1e-9)
    deg = np.bincount(g.edge_index[0], minlength=g.n_nodes)
    deg_p = np.bincount(g_p.edge_index[0], minlength=g_p.n_nodes)
    np.testing.assert_array_equal(np.sort(deg), np.sort(deg_p))


def test_cutoff_controls_connectivity(make_arrangement: ArrangementFactory) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[], seed=6)
    sparse = build_graph(a, cutoff=0.5)
    dense = build_graph(a, cutoff=100.0)
    assert sparse.n_edges <= dense.n_edges
    # Large cutoff connects every ordered pair (no self-loops): n*(n-1).
    assert dense.n_edges == dense.n_nodes * (dense.n_nodes - 1)


def test_no_self_loops(make_arrangement: ArrangementFactory) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[], seed=7)
    g = build_graph(a, cutoff=100.0)
    assert not np.any(g.edge_index[0] == g.edge_index[1])


def test_vacancy_counts_use_oxygen_positions(
    make_arrangement: ArrangementFactory, oxygen_sites: np.ndarray
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0, 1], seed=8)
    g = build_graph(
        a,
        cutoff=5.0,
        vacancy_positions=oxygen_sites,
        vacancy_cutoff=100.0,
    )
    # With a huge vacancy cutoff, every cation "sees" both vacancies.
    assert np.all(g.node_vacancy_count == 2)


def test_no_vacancy_positions_gives_zero_counts(
    make_arrangement: ArrangementFactory,
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0, 1], seed=9)
    g = build_graph(a, cutoff=5.0)
    assert np.all(g.node_vacancy_count == 0)
