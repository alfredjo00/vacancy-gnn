"""Analytic tests for the configurational free energy.

These encode the physics argument in PLAN.md Section 2.1: the lowest-energy
arrangement is the ``T -> 0`` limit of the Boltzmann average, and the uniform
mean is the ``T -> inf`` limit. If these ever break, the central claim of the
project is wrong, so they are the highest-value tests in the suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vacancy_gnn.physics import boltzmann_weights, free_energy, free_energy_sweep
from vacancy_gnn.physics.constants import K_B_EV


def test_zero_temperature_equals_minimum() -> None:
    energies = [-3.1, -5.0, -4.2, -1.0]
    assert free_energy(energies, 0.0) == pytest.approx(min(energies))


def test_low_temperature_approaches_minimum() -> None:
    # A well-separated ground state: at low T the sum collapses onto it.
    energies = [-5.0, -1.0, 0.0]
    assert free_energy(energies, 1.0) == pytest.approx(-5.0, abs=1e-6)


def test_high_temperature_approaches_uniform_mean() -> None:
    # As T -> inf, G -> <E> - kT ln N; subtracting the entropy term leaves the
    # uniform arithmetic mean of the energies.
    energies = np.array([-2.0, 0.0, 1.0, 3.0])
    t = 1e8
    n = energies.size
    g = free_energy(energies, t)
    recovered_mean = g + K_B_EV * t * math.log(n)
    assert recovered_mean == pytest.approx(energies.mean(), abs=1e-3)


def test_two_state_system_matches_closed_form() -> None:
    # Analytic check: G = e0 - kT ln(1 + exp(-(e1-e0)/kT)) for two states e0<e1.
    e0, e1 = -4.0, -3.5
    t = 1000.0
    beta = 1.0 / (K_B_EV * t)
    expected = e0 - K_B_EV * t * math.log(1.0 + math.exp(-beta * (e1 - e0)))
    assert free_energy([e1, e0], t) == pytest.approx(expected)


def test_free_energy_is_below_or_equal_to_min() -> None:
    # Configurational entropy can only lower (or leave) the free energy.
    energies = [-2.0, -1.5, -1.0, 0.5]
    assert free_energy(energies, 1323.0) <= min(energies) + 1e-12


def test_numerical_stability_large_magnitude() -> None:
    # Anchoring to the minimum must prevent overflow for large |E|/kT.
    energies = [-500.0, -499.9, -499.5]
    g = free_energy(energies, 5.0)
    assert math.isfinite(g)
    assert g <= min(energies) + 1e-9


def test_weights_sum_to_one_and_favor_low_energy() -> None:
    energies = [-3.0, -2.0, 0.0]
    w = boltzmann_weights(energies, 1323.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]


def test_sweep_is_monotone_nonincreasing_toward_min_at_low_t() -> None:
    energies = [-4.0, -2.0, 1.0]
    temps = [1e-9, 300.0, 1323.0, 5000.0]
    g = free_energy_sweep(energies, temps)
    assert g[0] == pytest.approx(min(energies), abs=1e-6)
    # Lower T -> closer to (and not below) the ground state.
    assert g[0] >= g[1] >= g[2] >= g[3]


@pytest.mark.parametrize(
    ("bad_energies", "temperature"),
    [([], 1000.0), ([1.0], -1.0)],
)
def test_invalid_inputs_raise(bad_energies: list[float], temperature: float) -> None:
    with pytest.raises(ValueError):
        free_energy(bad_energies, temperature)


def test_weights_reject_zero_temperature() -> None:
    with pytest.raises(ValueError):
        boltzmann_weights([1.0, 2.0], 0.0)
