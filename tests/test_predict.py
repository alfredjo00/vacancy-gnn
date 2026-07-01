"""Tests for inference: candidate arrangements -> Boltzmann-averaged G(v)."""

from __future__ import annotations

import pytest

from vacancy_gnn.data.synthetic import (
    make_brute_force_reference,
    make_synthetic_dataset,
)
from vacancy_gnn.models.baseline import LinearBaseline
from vacancy_gnn.physics.boltzmann import free_energy
from vacancy_gnn.predict import predict_free_energy
from vacancy_gnn.train import train


def _fitted_baseline(seed: int = 0) -> LinearBaseline:
    dataset = make_synthetic_dataset(n_compositions=10, seed=seed)
    model = LinearBaseline()
    train(model, dataset, seed=seed)
    return model


def test_predict_free_energy_matches_manual_computation() -> None:
    model = _fitted_baseline()
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(2,), arrangements_per_level=15, seed=1
    )
    arrangements = list(reference.arrangements)

    g = predict_free_energy(model, arrangements, temperature=1323.0)

    from vacancy_gnn.data.featurize import build_graph

    graphs = [build_graph(a, cutoff=5.0) for a in arrangements]
    expected = free_energy(model.predict(graphs), 1323.0)
    assert g == pytest.approx(expected)


def test_predict_free_energy_rejects_empty() -> None:
    model = _fitted_baseline()
    with pytest.raises(ValueError, match="non-empty"):
        predict_free_energy(model, [], temperature=1323.0)


def test_predict_free_energy_rejects_mixed_composition() -> None:
    model = _fitted_baseline()
    reference = make_brute_force_reference(
        n_compositions=2, vacancy_levels=(1,), arrangements_per_level=2, seed=2
    )
    with pytest.raises(ValueError, match="multiple compositions"):
        predict_free_energy(model, list(reference.arrangements), temperature=1323.0)


def test_predict_free_energy_rejects_mixed_vacancy_count() -> None:
    model = _fitted_baseline()
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(1, 2), arrangements_per_level=2, seed=3
    )
    with pytest.raises(ValueError, match="multiple vacancy counts"):
        predict_free_energy(model, list(reference.arrangements), temperature=1323.0)
