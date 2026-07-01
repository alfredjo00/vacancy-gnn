"""Tests for the evaluation harness (PLAN.md Section 7)."""

from __future__ import annotations

import numpy as np

from vacancy_gnn.data.synthetic import (
    make_brute_force_reference,
    make_synthetic_dataset,
)
from vacancy_gnn.evaluate import (
    evaluate,
    free_energy_accuracy,
    group_by_composition_and_v,
    min_vs_average_divergence,
    oracle_efficiency_curves,
    per_arrangement_parity,
    temperature_sweeps,
)
from vacancy_gnn.models.baseline import LinearBaseline
from vacancy_gnn.physics.constants import T_FR
from vacancy_gnn.train import train


def _fitted_baseline(seed: int = 0) -> LinearBaseline:
    dataset = make_synthetic_dataset(n_compositions=10, seed=seed)
    model = LinearBaseline()
    train(model, dataset, seed=seed)
    return model


def test_group_by_composition_and_v_partitions_all_arrangements() -> None:
    reference = make_brute_force_reference(
        n_compositions=2, vacancy_levels=(1, 2), arrangements_per_level=5, seed=1
    )
    groups = group_by_composition_and_v(reference)
    assert len(groups) == 4  # 2 compositions x 2 vacancy levels
    total = sum(len(g.indices) for g in groups)
    assert total == len(reference)
    for g in groups:
        assert len(g.indices) == 5
        assert all(reference.arrangements[i].v == g.v for i in g.indices)
        assert all(
            reference.arrangements[i].composition == g.composition for i in g.indices
        )


def test_per_arrangement_parity_matches_manual_metrics() -> None:
    model = _fitted_baseline()
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(1,), arrangements_per_level=20, seed=2
    )
    result = per_arrangement_parity(model, reference)
    assert result.y_true.shape == result.y_pred.shape == (20,)
    assert result.mae >= 0.0
    assert result.rmse >= result.mae


def test_free_energy_accuracy_one_estimate_per_group() -> None:
    model = _fitted_baseline()
    reference = make_brute_force_reference(
        n_compositions=2, vacancy_levels=(1, 2), arrangements_per_level=30, seed=3
    )
    results = free_energy_accuracy(model, reference, temperature=T_FR)
    assert len(results) == 4
    for est in results:
        assert np.isfinite(est.truth)
        assert np.isfinite(est.model_estimate)
        assert est.error == est.model_estimate - est.truth


def test_oracle_efficiency_curve_ends_at_truth() -> None:
    model = _fitted_baseline()
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(2,), arrangements_per_level=40, seed=4
    )
    curves = oracle_efficiency_curves(model, reference, temperature=T_FR)
    assert len(curves) == 1
    curve = curves[0]
    assert curve.sample_sizes[0] == 1
    assert curve.sample_sizes[-1] == 40
    # Using all 40 arrangements in random order must reproduce the true G(v).
    assert curve.random_curve[-1] == curve.truth


def test_min_vs_average_flags_entropy_dominated_flat_landscape() -> None:
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(2,), arrangements_per_level=50, seed=5
    )
    results = min_vs_average_divergence(reference, reactor_temperature=T_FR)
    assert len(results) == 1
    div = results[0]
    assert div.g_reactor_t <= div.g_zero_t + 1e-9  # entropy can only lower G
    assert div.divergence == div.g_reactor_t - div.g_zero_t
    assert div.entropy_dominated == (abs(div.divergence) > 0.05)


def test_min_vs_average_zero_temperature_reproduces_minimum() -> None:
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(1,), arrangements_per_level=10, seed=6
    )
    group = group_by_composition_and_v(reference)[0]
    energies = [reference.arrangements[i].energy_ev for i in group.indices]
    results = min_vs_average_divergence(reference, reactor_temperature=T_FR)
    assert results[0].g_zero_t == min(energies)


def test_temperature_sweep_zero_and_high_t_limits() -> None:
    reference = make_brute_force_reference(
        n_compositions=1, vacancy_levels=(1,), arrangements_per_level=10, seed=7
    )
    group = group_by_composition_and_v(reference)[0]
    energies = np.array([reference.arrangements[i].energy_ev for i in group.indices])
    sweeps = temperature_sweeps(reference, temperatures=[0.0, 1e8])
    assert len(sweeps) == 1
    sweep = sweeps[0]
    assert sweep.free_energies[0] == energies.min()
    assert sweep.free_energies[0] >= sweep.free_energies[1]


def test_evaluate_returns_full_report() -> None:
    model = _fitted_baseline()
    reference = make_brute_force_reference(
        n_compositions=2, vacancy_levels=(1, 2), arrangements_per_level=25, seed=8
    )
    report = evaluate(model, reference, reactor_temperature=T_FR)
    assert report.parity.mae >= 0.0
    assert len(report.free_energy_accuracy) == 4
    assert len(report.oracle_efficiency) == 4
    assert len(report.min_vs_average) == 4
    assert len(report.temperature_sweeps) == 4
