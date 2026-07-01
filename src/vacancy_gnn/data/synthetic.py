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


def make_synthetic_dataset(
    n_compositions: int = 12,
    arrangements_per_composition: int = 8,
    n_cations: int = 6,
    cutoff: float = 5.0,
    seed: int = 0,
) -> Dataset:
    """Generate a synthetic labeled dataset with learnable energy structure.

    Args:
        n_compositions: Number of distinct compositions.
        arrangements_per_composition: Vacancy arrangements per composition.
        n_cations: Cation sites per structure.
        cutoff: Cutoff used to compute the descriptor the label is derived from.
        seed: Master RNG seed.

    Returns:
        A validated :class:`Dataset`.
    """
    rng = np.random.default_rng(seed)
    # A fixed random linear response over the descriptor gives learnable labels.
    weight_seed = np.random.default_rng(seed + 1)
    ref_weight = weight_seed.normal(size=descriptor_length())

    arrangements: list[Arrangement] = []

    for c in range(n_compositions):
        species = rng.choice(_FAMILY_SPECIES, size=n_cations).tolist()
        comp = f"FeMnAl-{c:03d}"
        for _ in range(arrangements_per_composition):
            positions = rng.uniform(-3.0, 3.0, size=(n_cations, 3))
            v = int(rng.integers(0, 4))
            vacancy_sites = sorted(rng.choice(range(8), size=v, replace=False).tolist())
            base = Arrangement(
                composition=comp,
                family="FeMnAl",
                v=v,
                cation_species=species,
                cation_positions=positions.tolist(),
                vacancy_sites=vacancy_sites,
                energy_ev=0.0,  # placeholder; overwritten below
                source_run="synthetic",
            )
            desc = graph_descriptor(build_graph(base, cutoff=cutoff))
            energy = float(desc @ ref_weight + rng.normal(0.0, 0.5))
            arrangements.append(base.model_copy(update={"energy_ev": energy}))

    return Dataset(arrangements=arrangements)
