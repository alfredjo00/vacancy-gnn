"""Configurational free energy over vacancy arrangements.

This module is the scientifically load-bearing core of the package and is kept
pure (numpy only, no ML dependency). It computes the canonical configurational
free energy at level ``v``,

    G(T) = -k_B T ln  sum_i exp(-E_i / k_B T),

over the arrangement energies ``E_i``. See PLAN.md Section 2.1 for why this
Boltzmann-weighted average, rather than the single lowest-energy arrangement, is
the physically correct quantity at reactor temperatures.

The lowest-energy arrangement is recovered exactly as the ``T -> 0`` limit; the
uniform mean is recovered as ``T -> inf``. Both limits are asserted in the tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vacancy_gnn.physics.constants import K_B_EV


def free_energy(energies: ArrayLike, temperature: float) -> float:
    """Configurational free energy ``G(T)`` over a set of arrangement energies.

    Computed in a numerically stable way via the log-sum-exp identity, so the
    result is anchored to ``min(energies)`` and does not overflow for large
    ``|E_i| / k_B T``.

    Args:
        energies: 1-D array of arrangement energies ``E_i`` (eV).
        temperature: Temperature in kelvin. Must be ``>= 0``.

    Returns:
        The free energy ``G(T)`` in eV.

    Raises:
        ValueError: If ``energies`` is empty or ``temperature`` is negative.
    """
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size == 0:
        raise ValueError("energies must be non-empty")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    e_min = float(e.min())
    if temperature == 0.0:
        # T -> 0 limit: the free energy collapses onto the lowest arrangement.
        return e_min

    beta = 1.0 / (K_B_EV * temperature)
    # G = e_min - kT * ln sum_i exp(-beta (E_i - e_min))
    shifted = -beta * (e - e_min)
    log_sum = float(np.logaddexp.reduce(shifted))
    return e_min - (1.0 / beta) * log_sum


def boltzmann_weights(energies: ArrayLike, temperature: float) -> NDArray[np.float64]:
    """Normalized Boltzmann weights ``p_i = exp(-E_i/kT) / Z`` for each arrangement.

    Args:
        energies: 1-D array of arrangement energies (eV).
        temperature: Temperature in kelvin. Must be ``> 0``.

    Returns:
        Array of weights summing to 1, same length as ``energies``.

    Raises:
        ValueError: If ``energies`` is empty or ``temperature`` is not positive.
    """
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size == 0:
        raise ValueError("energies must be non-empty")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 for weights, got {temperature}")

    beta = 1.0 / (K_B_EV * temperature)
    shifted = -beta * (e - e.min())
    w = np.exp(shifted)
    normalized: NDArray[np.float64] = w / w.sum()
    return normalized


def free_energy_sweep(
    energies: ArrayLike, temperatures: ArrayLike
) -> NDArray[np.float64]:
    """Vectorized ``free_energy`` over a temperature grid.

    This is the ``T``-sweep used in the evaluation harness (PLAN.md Section 7) to
    show ``G(T)`` interpolating between the lowest-arrangement value at ``T -> 0``
    and the entropy-lowered value at reactor temperature.

    Args:
        energies: 1-D array of arrangement energies (eV).
        temperatures: 1-D array of temperatures (K).

    Returns:
        Array of ``G(T)`` values, one per temperature.
    """
    temps = np.asarray(temperatures, dtype=np.float64).ravel()
    return np.array([free_energy(energies, float(t)) for t in temps])
