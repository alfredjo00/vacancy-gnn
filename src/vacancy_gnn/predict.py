"""Inference: score candidate arrangements and Boltzmann-average into G(v).

This is the production entry point the training/evaluation machinery builds
toward: given a fitted model and a set of candidate vacancy arrangements for one
composition and vacancy count, predict each arrangement's energy and combine them
into the low-noise ``G(v)`` (PLAN.md Section 4/7). No CHGNet call is made here;
the model was trained on labels exported offline.
"""

from __future__ import annotations

from vacancy_gnn.data.featurize import build_graph
from vacancy_gnn.data.schema import Arrangement
from vacancy_gnn.models.base import EnergyModel
from vacancy_gnn.physics.boltzmann import free_energy


def predict_free_energy(
    model: EnergyModel,
    arrangements: list[Arrangement],
    *,
    temperature: float,
    cutoff: float = 5.0,
) -> float:
    """Boltzmann-averaged free energy over model-predicted arrangement energies.

    Args:
        model: A fitted model implementing :class:`EnergyModel`.
        arrangements: Candidate vacancy arrangements to score. All must share the
            same composition and vacancy count for the result to correspond to a
            single ``G(v)``.
        temperature: Temperature in kelvin.
        cutoff: Edge distance cutoff for graph construction; must match training.

    Returns:
        The predicted ``G(v)`` in eV.

    Raises:
        ValueError: If ``arrangements`` is empty, or its entries mix compositions
            or vacancy counts.
    """
    if not arrangements:
        raise ValueError("arrangements must be non-empty")
    compositions = {a.composition for a in arrangements}
    levels = {a.v for a in arrangements}
    if len(compositions) > 1:
        raise ValueError(f"arrangements span multiple compositions: {compositions}")
    if len(levels) > 1:
        raise ValueError(f"arrangements span multiple vacancy counts: {levels}")

    graphs = [build_graph(a, cutoff=cutoff) for a in arrangements]
    energies = model.predict(graphs)
    return free_energy(energies, temperature)
