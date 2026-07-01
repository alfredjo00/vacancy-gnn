"""Tests for regression metrics."""

from __future__ import annotations

import numpy as np
import pytest

from vacancy_gnn.metrics import free_energy_convergence, mae, rmse
from vacancy_gnn.physics.boltzmann import free_energy


def test_mae_zero_for_exact() -> None:
    y = np.array([1.0, -2.0, 3.5])
    assert mae(y, y) == 0.0


def test_rmse_zero_for_exact() -> None:
    y = np.array([1.0, -2.0, 3.5])
    assert rmse(y, y) == 0.0


def test_mae_known_value() -> None:
    assert mae([0.0, 0.0], [1.0, 3.0]) == pytest.approx(2.0)


def test_rmse_known_value() -> None:
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_rmse_at_least_mae() -> None:
    yt = np.array([0.0, 0.0, 0.0])
    yp = np.array([1.0, 2.0, 3.0])
    assert rmse(yt, yp) >= mae(yt, yp)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        rmse([], [])


def test_free_energy_convergence_default_sizes_span_full_range() -> None:
    energies = [-4.0, -3.5, -3.0, -1.0, 0.5]
    sizes, estimates = free_energy_convergence(energies, 1323.0)
    assert sizes.tolist() == [1, 2, 3, 4, 5]
    assert estimates.shape == (5,)


def test_free_energy_convergence_full_sample_matches_free_energy() -> None:
    energies = [-4.0, -3.5, -3.0, -1.0, 0.5]
    _, estimates = free_energy_convergence(energies, 1323.0)
    assert estimates[-1] == pytest.approx(free_energy(energies, 1323.0))


def test_free_energy_convergence_single_sample_is_one_of_the_energies() -> None:
    energies = [-4.0, -3.5, -3.0]
    _, estimates = free_energy_convergence(energies, 1323.0)
    assert estimates[0] in energies


def test_free_energy_convergence_custom_sample_sizes() -> None:
    energies = [-4.0, -3.5, -3.0, -1.0, 0.5]
    sizes, estimates = free_energy_convergence(energies, 1323.0, sample_sizes=[1, 5])
    assert sizes.tolist() == [1, 5]
    assert estimates.shape == (2,)


def test_free_energy_convergence_rejects_oversized_sample() -> None:
    with pytest.raises(ValueError, match="exceeds available"):
        free_energy_convergence([-1.0, -2.0], 1323.0, sample_sizes=[3])


def test_free_energy_convergence_rejects_empty_energies() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        free_energy_convergence([], 1323.0)


def test_free_energy_convergence_deterministic_given_seed() -> None:
    energies = [-4.0, -3.5, -3.0, -1.0, 0.5]
    _, e1 = free_energy_convergence(energies, 1323.0, seed=42)
    _, e2 = free_energy_convergence(energies, 1323.0, seed=42)
    np.testing.assert_array_equal(e1, e2)
