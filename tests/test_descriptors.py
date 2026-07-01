"""Tests for the invariant descriptor used by the baseline."""

from __future__ import annotations

import numpy as np
import pytest

from vacancy_gnn.data.descriptors import (
    DESCRIPTOR_SPECIES,
    descriptor_length,
    graph_descriptor,
)
from vacancy_gnn.data.featurize import build_graph

from .conftest import ArrangementFactory


def test_descriptor_has_fixed_length(make_arrangement: ArrangementFactory) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0, 1], seed=1)
    d = graph_descriptor(build_graph(a, cutoff=5.0))
    assert d.shape == (descriptor_length(),)


def test_descriptor_invariant_under_permutation(
    make_arrangement: ArrangementFactory,
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0, 2], seed=2)
    d = graph_descriptor(build_graph(a, cutoff=5.0))

    perm = np.array([2, 0, 3, 1])
    a_p = a.model_copy(
        update={
            "cation_positions": a.positions_array()[perm].tolist(),
            "cation_species": a.species_array()[perm].tolist(),
        }
    )
    d_p = graph_descriptor(build_graph(a_p, cutoff=5.0))
    np.testing.assert_allclose(d, d_p, atol=1e-9)


def test_descriptor_invariant_under_rotation(
    make_arrangement: ArrangementFactory,
) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[0], seed=3)
    d = graph_descriptor(build_graph(a, cutoff=5.0))

    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    a_r = a.model_copy(
        update={"cation_positions": (a.positions_array() @ q.T).tolist()}
    )
    d_r = graph_descriptor(build_graph(a_r, cutoff=5.0))
    np.testing.assert_allclose(d, d_r, atol=1e-9)


def test_unknown_species_raises(make_arrangement: ArrangementFactory) -> None:
    a = make_arrangement("C", "F", vacancy_sites=[], seed=4)
    a_bad = a.model_copy(update={"cation_species": [999, 999, 999, 999]})
    assert 999 not in DESCRIPTOR_SPECIES
    with pytest.raises(ValueError):
        graph_descriptor(build_graph(a_bad, cutoff=5.0))
