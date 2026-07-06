"""Tests for the per-species composition reference (IMPROVEMENTS.md P8)."""

from __future__ import annotations

import numpy as np
import pytest

from vacancy_gnn.data.descriptors import DESCRIPTOR_SPECIES
from vacancy_gnn.data.featurize import VACANCY_MARKER_Z, Graph
from vacancy_gnn.models.reference import (
    CompositionReference,
    out_of_span_norm,
    prior_from_e0s,
)


def _graph(species: list[int]) -> Graph:
    return Graph(
        node_z=np.array(species, dtype=np.int64),
        node_is_vacancy=np.array(
            [z == VACANCY_MARKER_Z for z in species], dtype=np.bool_
        ),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_vec=np.zeros((0, 3), dtype=np.float64),
        edge_dist=np.zeros((0,), dtype=np.float64),
    )


def test_plain_fit_reproduces_consistent_energies() -> None:
    graphs = [_graph([26, 26, 13]), _graph([26, 13, 13]), _graph([13, 13, 13])]
    energies = np.array([-10.0, -8.0, -6.0])

    ref = CompositionReference()
    ref.fit(graphs, energies)

    np.testing.assert_allclose(ref.predict(graphs), energies, atol=1e-8)


def test_shrinkage_zero_is_byte_identical_to_plain_fit() -> None:
    graphs = [_graph([26, 26, 13]), _graph([26, 13, 13]), _graph([13, 13, 13])]
    energies = np.array([-10.0, -8.0, -6.0])
    prior = np.zeros(len(DESCRIPTOR_SPECIES))

    plain = CompositionReference()
    plain.fit(graphs, energies)

    anchored = CompositionReference()
    anchored.fit(graphs, energies, prior=prior, shrinkage=0.0)

    np.testing.assert_array_equal(plain.coeffs, anchored.coeffs)


def test_shrinkage_requires_a_prior() -> None:
    graphs = [_graph([26, 13])]
    energies = np.array([-10.0])

    ref = CompositionReference()
    with pytest.raises(ValueError, match="prior"):
        ref.fit(graphs, energies, shrinkage=1.0)


def test_anchored_fit_uses_prior_along_unconstrained_direction() -> None:
    """Rank-deficient design (single composition): data can't separate the two
    cation species' coefficients, only their sum. A large shrinkage should pin
    the fit near the prior instead of the arbitrary minimum-norm split."""
    graphs = [_graph([26, 13]), _graph([26, 13])]  # same composition, twice
    energies = np.array([-10.0, -10.0])

    idx_fe = DESCRIPTOR_SPECIES.index(26)
    idx_al = DESCRIPTOR_SPECIES.index(13)
    prior = np.zeros(len(DESCRIPTOR_SPECIES))
    prior[idx_fe] = -3.0
    prior[idx_al] = -7.0  # sums to -10, matches the data's constrained total

    ref = CompositionReference()
    ref.fit(graphs, energies, prior=prior, shrinkage=1e6)

    assert ref.coeffs[idx_fe] == pytest.approx(-3.0, abs=1e-2)
    assert ref.coeffs[idx_al] == pytest.approx(-7.0, abs=1e-2)
    # Constrained direction (the sum) still matches the data exactly.
    np.testing.assert_allclose(ref.predict(graphs), energies, atol=1e-6)


_ALL_SYMBOLS = (
    "Li",
    "Mg",
    "Al",
    "Ca",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Zr",
    "In",
    "Sn",
)


def _all_e0s(**overrides: float) -> dict[str, float]:
    """A complete E0 table (every DESCRIPTOR_SPECIES cation plus O)."""
    e0s = {symbol: -5.0 for symbol in _ALL_SYMBOLS}
    e0s["O"] = -1.0
    e0s.update(overrides)
    return e0s


def test_prior_from_e0s_reproduces_reference_energy() -> None:
    e0s = _all_e0s(Fe=-3.0, Al=-2.0)
    n_cations = 4
    n_oxygen_sites = 8
    prior = prior_from_e0s(e0s, n_cations=n_cations, n_oxygen_sites=n_oxygen_sites)

    idx_fe = DESCRIPTOR_SPECIES.index(26)
    idx_al = DESCRIPTOR_SPECIES.index(13)
    idx_marker = DESCRIPTOR_SPECIES.index(VACANCY_MARKER_Z)

    # 3 Fe, 1 Al cations, v vacancies out of n_oxygen_sites ideal O sites.
    for v in (0, 2, 5):
        expected = (3 * e0s["Fe"] + 1 * e0s["Al"]) + (n_oxygen_sites - v) * e0s["O"]
        got = 3 * prior[idx_fe] + 1 * prior[idx_al] + v * prior[idx_marker]
        assert got == pytest.approx(expected)


def test_prior_from_e0s_requires_oxygen() -> None:
    with pytest.raises(ValueError, match="O"):
        prior_from_e0s({"Fe": -3.0}, n_cations=4, n_oxygen_sites=8)


def test_prior_from_e0s_requires_every_cation_species() -> None:
    incomplete = _all_e0s()
    del incomplete["Fe"]
    with pytest.raises(ValueError, match="Fe"):
        prior_from_e0s(incomplete, n_cations=4, n_oxygen_sites=8)


def test_prior_from_e0s_rejects_nonpositive_n_cations() -> None:
    with pytest.raises(ValueError, match="n_cations"):
        prior_from_e0s({"O": -1.0}, n_cations=0, n_oxygen_sites=8)


def test_coeffs_round_trip_through_to_list_and_from_list() -> None:
    graphs = [_graph([26, 26, 13]), _graph([26, 13, 13]), _graph([13, 13, 13])]
    energies = np.array([-10.0, -8.0, -6.0])
    ref = CompositionReference()
    ref.fit(graphs, energies)

    restored = CompositionReference.from_list(ref.to_list())

    np.testing.assert_array_equal(ref.coeffs, restored.coeffs)
    np.testing.assert_allclose(restored.predict(graphs), ref.predict(graphs))


def test_out_of_span_norm_is_zero_in_span() -> None:
    train = [_graph([26, 26, 13]), _graph([26, 13, 13])]
    in_span = _graph([26, 13, 13])  # identical to a training graph

    assert out_of_span_norm(train, in_span) == pytest.approx(0.0, abs=1e-8)


def test_out_of_span_norm_is_positive_out_of_span() -> None:
    train = [_graph([26, 26, 13]), _graph([26, 26, 13])]  # only ever this composition
    out_of_span = _graph([13, 13, 13])  # a composition never seen in training

    assert out_of_span_norm(train, out_of_span) > 1.0
