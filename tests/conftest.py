"""Shared synthetic fixtures.

These build small, valid arrangements without any interatomic-potential call, so
the data-layer tests run fast and offline (PLAN.md step 3).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from vacancy_gnn.data.schema import Arrangement, Dataset

ArrangementFactory = Callable[..., Arrangement]

# A small oxygen sublattice shared across arrangements (indexable vacancy sites).
OXYGEN_SITES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _make_arrangement(
    composition: str,
    family: str,
    vacancy_sites: list[int],
    seed: int = 0,
) -> Arrangement:
    """Build a valid arrangement with random cation positions."""
    rng = np.random.default_rng(seed)
    n_cations = 4
    positions = rng.uniform(-2.0, 2.0, size=(n_cations, 3))
    species = [26, 25, 13, 26]  # Fe, Mn, Al, Fe
    return Arrangement(
        composition=composition,
        family=family,
        v=len(vacancy_sites),
        cation_species=species,
        cation_positions=positions.tolist(),
        vacancy_sites=vacancy_sites,
        energy_ev=float(rng.normal(-100.0, 1.0)),
        source_run="synthetic",
    )


@pytest.fixture
def make_arrangement() -> ArrangementFactory:
    """Factory fixture: build a valid arrangement in a test."""
    return _make_arrangement


@pytest.fixture
def oxygen_sites() -> np.ndarray:
    return OXYGEN_SITES


@pytest.fixture
def small_dataset() -> Dataset:
    """Six compositions, two arrangements each (12 arrangements)."""
    arrangements = []
    for c in range(6):
        comp = f"Comp{c}"
        for k in range(2):
            arrangements.append(
                _make_arrangement(
                    composition=comp,
                    family="FeMnAl",
                    vacancy_sites=[k, k + 1],
                    seed=100 * c + k,
                )
            )
    return Dataset(arrangements=arrangements)
