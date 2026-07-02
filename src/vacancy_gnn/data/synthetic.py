"""Synthetic dataset generation.

Produces small, physically plausible-looking arrangements without any interatomic
potential, so the training loop and CLI have something real to run on before the
data factory export exists (PLAN.md step 4). The energy label is a deterministic
function of the descriptor plus noise, so a fitted model can actually learn signal.
"""

from __future__ import annotations

import numpy as np

from vacancy_gnn.data.descriptors import descriptor_length, graph_descriptor
from vacancy_gnn.data.featurize import build_graph
from vacancy_gnn.data.schema import Arrangement, Dataset

_FAMILY_SPECIES: tuple[int, ...] = (26, 25, 13)  # Fe, Mn, Al

# A cubic cell large relative to the cutoff so the synthetic point clouds are
# featurized with well-defined (rarely wrapping) minimum-image edges.
_CELL: list[list[float]] = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]]


def _oxygen_sublattice(rng: np.random.Generator, n_sites: int) -> list[list[float]]:
    """Random ideal oxygen sublattice inside the synthetic cell."""
    sites: list[list[float]] = rng.uniform(-3.0, 3.0, size=(n_sites, 3)).tolist()
    return sites


def _reference_weight(weight_seed: int) -> np.ndarray:
    """The fixed random linear response over the descriptor that defines labels.

    Shared by :func:`make_synthetic_dataset` and :func:`make_brute_force_reference`
    (via the same default ``weight_seed``) so a model trained on one is scored
    against a reference drawn from the *same* underlying energy function, not a
    different random one (PLAN.md Section 7 requires the two to be comparable).
    """
    return np.random.default_rng(weight_seed).normal(size=descriptor_length())


def make_synthetic_dataset(
    n_compositions: int = 12,
    arrangements_per_composition: int = 8,
    n_cations: int = 6,
    cutoff: float = 5.0,
    seed: int = 0,
    weight_seed: int = 1,
) -> Dataset:
    """Generate a synthetic labeled dataset with learnable energy structure.

    Args:
        n_compositions: Number of distinct compositions.
        arrangements_per_composition: Vacancy arrangements per composition.
        n_cations: Cation sites per structure.
        cutoff: Cutoff used to compute the descriptor the label is derived from.
        seed: Master RNG seed for composition/arrangement sampling.
        weight_seed: Seed for the fixed descriptor->energy response. Share this
            value with :func:`make_brute_force_reference` (the default already
            does) so the two datasets are labeled by the same ground truth.

    Returns:
        A validated :class:`Dataset`.
    """
    rng = np.random.default_rng(seed)
    ref_weight = _reference_weight(weight_seed)

    arrangements: list[Arrangement] = []

    n_oxygen_sites = 8
    for c in range(n_compositions):
        species = rng.choice(_FAMILY_SPECIES, size=n_cations).tolist()
        oxygen_positions = _oxygen_sublattice(rng, n_oxygen_sites)
        comp = f"FeMnAl-{c:03d}"
        for _ in range(arrangements_per_composition):
            positions = rng.uniform(-3.0, 3.0, size=(n_cations, 3))
            v = int(rng.integers(0, 4))
            vacancy_sites = sorted(
                rng.choice(n_oxygen_sites, size=v, replace=False).tolist()
            )
            base = Arrangement(
                composition=comp,
                family="FeMnAl",
                v=v,
                cation_species=species,
                cation_positions=positions.tolist(),
                oxygen_positions=oxygen_positions,
                vacancy_sites=vacancy_sites,
                cell=_CELL,
                energy_ev=0.0,  # placeholder; overwritten below
                source_run="synthetic",
            )
            desc = graph_descriptor(build_graph(base, cutoff=cutoff))
            energy = float(desc @ ref_weight + rng.normal(0.0, 0.5))
            arrangements.append(base.model_copy(update={"energy_ev": energy}))

    return Dataset(arrangements=arrangements)


def make_brute_force_reference(
    n_compositions: int = 4,
    vacancy_levels: tuple[int, ...] = (1, 2, 3),
    arrangements_per_level: int = 200,
    n_cations: int = 6,
    n_oxygen_sites: int = 8,
    cutoff: float = 5.0,
    seed: int = 1000,
    weight_seed: int = 1,
) -> Dataset:
    """Generate a brute-force reference: many arrangements per (composition, v).

    This stands in for the "hundreds of CHGNet-labeled arrangements per v" that
    PLAN.md Section 7 calls for. It shares its default ``weight_seed`` with
    :func:`make_synthetic_dataset`, so a model trained on the synthetic dataset is
    scored here against the *same* underlying energy function rather than an
    unrelated one; only ``seed`` (composition/arrangement sampling) differs, so the
    two datasets do not share draws. Each composition has a fixed cation
    decoration; only the vacancy placement varies within it, which is what lets
    ``G(v)`` be estimated from many draws at a fixed ``v``.

    Args:
        n_compositions: Number of distinct compositions.
        vacancy_levels: Vacancy counts to enumerate for every composition.
        arrangements_per_level: Number of distinct vacancy arrangements sampled per
            ``(composition, v)`` pair.
        n_cations: Cation sites per structure.
        n_oxygen_sites: Size of the oxygen sublattice vacancies are drawn from.
        cutoff: Cutoff used to compute the descriptor the label is derived from.
        seed: Master RNG seed for composition/arrangement sampling.
        weight_seed: Seed for the fixed descriptor->energy response; see
            :func:`make_synthetic_dataset`.

    Returns:
        A validated :class:`Dataset` with ``arrangements_per_level`` samples for
        every ``(composition, v)`` pair.
    """
    rng = np.random.default_rng(seed)
    ref_weight = _reference_weight(weight_seed)

    arrangements: list[Arrangement] = []

    for c in range(n_compositions):
        species = rng.choice(_FAMILY_SPECIES, size=n_cations).tolist()
        positions = rng.uniform(-3.0, 3.0, size=(n_cations, 3))
        oxygen_positions = _oxygen_sublattice(rng, n_oxygen_sites)
        comp = f"FeMnAl-ref-{c:03d}"
        for v in vacancy_levels:
            if v > n_oxygen_sites:
                raise ValueError(
                    f"vacancy level {v} exceeds n_oxygen_sites={n_oxygen_sites}"
                )
            for _ in range(arrangements_per_level):
                vacancy_sites = sorted(
                    rng.choice(n_oxygen_sites, size=v, replace=False).tolist()
                )
                base = Arrangement(
                    composition=comp,
                    family="FeMnAl",
                    v=v,
                    cation_species=species,
                    cation_positions=positions.tolist(),
                    oxygen_positions=oxygen_positions,
                    vacancy_sites=vacancy_sites,
                    cell=_CELL,
                    energy_ev=0.0,  # placeholder; overwritten below
                    source_run="synthetic-brute-force",
                )
                desc = graph_descriptor(build_graph(base, cutoff=cutoff))
                energy = float(desc @ ref_weight + rng.normal(0.0, 0.5))
                arrangements.append(base.model_copy(update={"energy_ev": energy}))

    return Dataset(arrangements=arrangements)
