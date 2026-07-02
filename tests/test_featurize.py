"""Featurization tests, including the symmetry properties a total energy requires.

The standout tests here assert that the *invariant* graph features (distances,
sorted degree) do not change under global rotation, translation, or node-index
permutation, and that the *equivariant* edge vectors rotate with the structure. A
downstream energy readout built on these features therefore inherits the correct
physical symmetry (PLAN.md Section 6). Vacancies are their own nodes, so the tests
also cover marker nodes and minimum-image periodicity.
"""

from __future__ import annotations

import numpy as np

from vacancy_gnn.data.featurize import VACANCY_MARKER_Z, build_graph
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


def _transform(a: Arrangement, rot: np.ndarray, shift: np.ndarray) -> Arrangement:
    """Rotate+translate cations, oxygen sublattice, and cell together."""
    pos = a.positions_array() @ rot.T + shift
    oxy = a.oxygen_positions_array() @ rot.T + shift
    cell = a.cell_array() @ rot.T
    return a.model_copy(
        update={
            "cation_positions": pos.tolist(),
            "oxygen_positions": oxy.tolist(),
            "cell": cell.tolist(),
        }
    )


def test_edge_distances_invariant_under_rotation_and_translation(
    make_arrangement: ArrangementFactory,
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0, 1], seed=3)
    g = build_graph(a, cutoff=5.0)

    rot = _rotation_matrix(seed=1)
    shift = np.array([10.0, -3.0, 7.0])
    g2 = build_graph(_transform(a, rot, shift), cutoff=5.0)

    assert g.n_edges == g2.n_edges
    np.testing.assert_allclose(np.sort(g.edge_dist), np.sort(g2.edge_dist), atol=1e-9)


def test_edge_vectors_rotate_with_structure(
    make_arrangement: ArrangementFactory,
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0], seed=4)
    g = build_graph(a, cutoff=5.0)

    rot = _rotation_matrix(seed=2)
    g2 = build_graph(_transform(a, rot, np.zeros(3)), cutoff=5.0)

    # Same connectivity, and edge vectors of g2 equal rotated edge vectors of g.
    np.testing.assert_array_equal(g.edge_index, g2.edge_index)
    np.testing.assert_allclose(g.edge_vec @ rot.T, g2.edge_vec, atol=1e-9)


def test_invariant_features_unchanged_under_permutation(
    make_arrangement: ArrangementFactory,
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0, 2], seed=5)
    g = build_graph(a, cutoff=5.0)

    perm = np.array([2, 0, 3, 1])  # cation-only permutation
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


def test_vacancies_become_marker_nodes(
    make_arrangement: ArrangementFactory,
) -> None:
    a0 = make_arrangement("C", "F", vacancy_sites=[], seed=8)
    a2 = make_arrangement("C", "F", vacancy_sites=[0, 1], seed=8)
    g0 = build_graph(a0, cutoff=5.0)
    g2 = build_graph(a2, cutoff=5.0)

    # Two vacancies add two marker nodes carrying the reserved species.
    assert g2.n_nodes == g0.n_nodes + 2
    assert int(g2.node_is_vacancy.sum()) == 2
    assert np.all(g2.node_z[g2.node_is_vacancy] == VACANCY_MARKER_Z)
    # Cation nodes are unchanged.
    np.testing.assert_array_equal(g2.node_z[~g2.node_is_vacancy], a2.species_array())


def test_minimum_image_wraps_across_boundary() -> None:
    # Two cations near opposite faces of a 10 A cubic cell: the true separation is
    # 2 A through the periodic boundary, not 8 A across the interior.
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    a = Arrangement(
        composition="C",
        family="F",
        v=0,
        cation_species=[26, 25],
        cation_positions=[[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
        oxygen_positions=[[5.0, 5.0, 5.0]],
        vacancy_sites=[],
        cell=cell,
        energy_ev=0.0,
    )
    g = build_graph(a, cutoff=5.0)
    # A cutoff of 5 A must connect them (2 A minimum image), which an open-boundary
    # 8 A distance would have excluded.
    assert g.n_edges == 2
    np.testing.assert_allclose(np.sort(g.edge_dist), [2.0, 2.0], atol=1e-9)
