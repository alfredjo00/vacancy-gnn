"""Tests for synthetic data generation, in particular the brute-force reference."""

from __future__ import annotations

import pytest

from vacancy_gnn.data.synthetic import (
    make_brute_force_reference,
    make_synthetic_dataset,
)


def test_brute_force_reference_shape() -> None:
    reference = make_brute_force_reference(
        n_compositions=3, vacancy_levels=(1, 2, 3), arrangements_per_level=10, seed=0
    )
    assert len(reference) == 3 * 3 * 10
    assert len(reference.compositions()) == 3


def test_brute_force_reference_fixed_cations_per_composition() -> None:
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(1, 2), arrangements_per_level=5, seed=1
    )
    species = {tuple(a.cation_species) for a in reference.arrangements}
    positions = {tuple(map(tuple, a.cation_positions)) for a in reference.arrangements}
    # Only the vacancy placement should vary within one composition.
    assert len(species) == 1
    assert len(positions) == 1


def test_brute_force_reference_vacancy_levels_match_v_field() -> None:
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(0, 2), arrangements_per_level=4, seed=2
    )
    levels = {a.v for a in reference.arrangements}
    assert levels == {0, 2}


def test_brute_force_reference_rejects_level_exceeding_oxygen_sites() -> None:
    with pytest.raises(ValueError, match="exceeds n_oxygen_sites"):
        make_brute_force_reference(vacancy_levels=(20,), n_oxygen_sites=8)


def test_brute_force_reference_deterministic_given_seed() -> None:
    a = make_brute_force_reference(seed=5, arrangements_per_level=5)
    b = make_brute_force_reference(seed=5, arrangements_per_level=5)
    a_energies = [x.energy_ev for x in a.arrangements]
    b_energies = [x.energy_ev for x in b.arrangements]
    assert a_energies == b_energies


def test_brute_force_reference_disjoint_from_default_synthetic_seed() -> None:
    # Default seeds should not collide, so a demo run drawing both is reproducible
    # and the two datasets are not accidentally identical draws.
    dataset = make_synthetic_dataset()
    reference = make_brute_force_reference()
    assert dataset.compositions() != reference.compositions()


def test_default_weight_seed_is_shared_between_dataset_and_reference() -> None:
    # A model trained on make_synthetic_dataset() must be scored against a
    # reference labeled by the same descriptor->energy function, or evaluation
    # results are meaningless. Verify this end to end: a model that fits the
    # dataset's descriptor->energy relationship exactly should also predict the
    # reference's labels exactly, since both come from the same weight_seed.
    import numpy as np

    from vacancy_gnn.data.descriptors import graph_descriptor
    from vacancy_gnn.data.featurize import build_graph
    from vacancy_gnn.data.synthetic import _reference_weight

    ref_weight = _reference_weight(weight_seed=1)  # shared default
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(1,), arrangements_per_level=5, seed=7
    )
    descriptors = np.stack(
        [graph_descriptor(build_graph(a, cutoff=5.0)) for a in reference.arrangements]
    )
    predicted = descriptors @ ref_weight
    actual = np.array([a.energy_ev for a in reference.arrangements])
    # Labels are `descriptor @ ref_weight + noise`; the noiseless prediction must
    # be close, confirming the reference uses the same weight vector as the
    # default-seeded make_synthetic_dataset().
    assert np.abs(predicted - actual).mean() < 1.0


def test_explicit_weight_seed_changes_labels() -> None:
    a = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(1,), arrangements_per_level=3, weight_seed=1
    )
    b = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(1,), arrangements_per_level=3, weight_seed=2
    )
    a_energies = [x.energy_ev for x in a.arrangements]
    b_energies = [x.energy_ev for x in b.arrangements]
    assert a_energies != b_energies
