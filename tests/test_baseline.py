"""Tests for the linear cluster-expansion baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vacancy_gnn.data.featurize import build_graph
from vacancy_gnn.data.synthetic import make_synthetic_dataset
from vacancy_gnn.models.base import EnergyModel
from vacancy_gnn.models.baseline import LinearBaseline


def _graphs_energies(cutoff: float = 5.0):
    ds = make_synthetic_dataset(n_compositions=8, seed=1)
    graphs = [build_graph(a, cutoff=cutoff) for a in ds.arrangements]
    energies = np.array([a.energy_ev for a in ds.arrangements])
    return graphs, energies


def test_baseline_satisfies_protocol() -> None:
    assert isinstance(LinearBaseline(), EnergyModel)


def test_fit_predict_recovers_signal() -> None:
    graphs, energies = _graphs_energies()
    model = LinearBaseline(regularization=1e-4)
    model.fit(graphs, energies)
    pred = model.predict(graphs)
    # Synthetic labels are a linear function of the descriptor plus small noise,
    # so the in-sample fit should track the labels closely.
    corr = np.corrcoef(pred, energies)[0, 1]
    assert corr > 0.9


def test_predict_before_fit_raises() -> None:
    graphs, _ = _graphs_energies()
    with pytest.raises(RuntimeError):
        LinearBaseline().predict(graphs)


def test_fit_rejects_empty() -> None:
    with pytest.raises(ValueError):
        LinearBaseline().fit([], np.array([]))


def test_fit_rejects_length_mismatch() -> None:
    graphs, energies = _graphs_energies()
    with pytest.raises(ValueError):
        LinearBaseline().fit(graphs, energies[:-1])


def test_negative_regularization_raises() -> None:
    with pytest.raises(ValueError):
        LinearBaseline(regularization=-1.0)


def test_save_load_round_trip(tmp_path: Path) -> None:
    graphs, energies = _graphs_energies()
    model = LinearBaseline(regularization=1e-3)
    model.fit(graphs, energies)
    before = model.predict(graphs)

    path = tmp_path / "model.json"
    model.save(path)
    restored = LinearBaseline.load(path)
    after = restored.predict(graphs)

    np.testing.assert_allclose(before, after, atol=1e-10)


def test_save_before_fit_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        LinearBaseline().save(tmp_path / "model.json")
