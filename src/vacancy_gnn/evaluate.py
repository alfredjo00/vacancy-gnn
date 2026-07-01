"""Evaluation harness: model vs brute-force reference (PLAN.md Section 7).

Every quantity here is derived from a fitted :class:`~vacancy_gnn.models.base.
EnergyModel` scored against a brute-force reference :class:`~vacancy_gnn.data.
schema.Dataset` that has many labeled arrangements per ``(composition, v)`` (see
:func:`vacancy_gnn.data.synthetic.make_brute_force_reference`). This module answers
the two questions the project exists to answer:

1. Does the learned per-arrangement energy predict well (MAE/RMSE, parity)?
2. Does Boltzmann-averaging those predictions recover the brute-force ``G(v)``
   with far fewer oracle calls than a single SQS draw, and how does that compare
   to the naive lowest-energy-arrangement estimate (Section 2.1)?

Kept framework-free (numpy only): plotting lives in ``notebooks/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from vacancy_gnn.data.featurize import build_graph
from vacancy_gnn.data.schema import Dataset
from vacancy_gnn.metrics import free_energy_convergence, mae, rmse
from vacancy_gnn.models.base import EnergyModel
from vacancy_gnn.physics.boltzmann import free_energy, free_energy_sweep


@dataclass(frozen=True)
class CompositionLevelGroup:
    """All reference arrangements for one ``(composition, v)`` pair."""

    composition: str
    v: int
    indices: list[int]


def group_by_composition_and_v(dataset: Dataset) -> list[CompositionLevelGroup]:
    """Partition a reference dataset into ``(composition, v)`` groups.

    Args:
        dataset: A brute-force reference dataset, typically from
            :func:`vacancy_gnn.data.synthetic.make_brute_force_reference`.

    Returns:
        Groups in first-seen order, each holding the arrangement indices sharing
        one composition and vacancy count.
    """
    groups: dict[tuple[str, int], list[int]] = {}
    for i, a in enumerate(dataset.arrangements):
        groups.setdefault((a.composition, a.v), []).append(i)
    return [
        CompositionLevelGroup(composition=comp, v=v, indices=idx)
        for (comp, v), idx in groups.items()
    ]


@dataclass(frozen=True)
class ParityResult:
    """Per-arrangement energy prediction quality against the reference."""

    y_true: NDArray[np.float64]
    y_pred: NDArray[np.float64]
    mae: float
    rmse: float


def per_arrangement_parity(
    model: EnergyModel, reference: Dataset, *, cutoff: float = 5.0
) -> ParityResult:
    """Score ``model`` on every arrangement in the brute-force reference.

    Args:
        model: A fitted model implementing :class:`EnergyModel`.
        reference: The brute-force reference dataset.
        cutoff: Edge distance cutoff for graph construction; must match training.

    Returns:
        A :class:`ParityResult` with true/predicted energies and MAE/RMSE.
    """
    graphs = [build_graph(a, cutoff=cutoff) for a in reference.arrangements]
    y_true = np.array([a.energy_ev for a in reference.arrangements], dtype=np.float64)
    y_pred = model.predict(graphs)
    return ParityResult(
        y_true=y_true, y_pred=y_pred, mae=mae(y_true, y_pred), rmse=rmse(y_true, y_pred)
    )


@dataclass(frozen=True)
class FreeEnergyEstimate:
    """Estimated vs. true ``G(v)`` for one ``(composition, v)`` group."""

    composition: str
    v: int
    truth: float
    model_estimate: float
    single_draw_estimate: float
    error: float


def free_energy_accuracy(
    model: EnergyModel,
    reference: Dataset,
    *,
    temperature: float,
    cutoff: float = 5.0,
    seed: int = 0,
) -> list[FreeEnergyEstimate]:
    """Compare brute-force truth, the model's full-group estimate, and a single draw.

    For every ``(composition, v)`` group, ``truth`` is the Boltzmann average of all
    reference labels; ``model_estimate`` is the Boltzmann average of the model's
    predictions over the same arrangements (the "cheap and low-variance" claim);
    ``single_draw_estimate`` is the free energy from one randomly drawn reference
    label, standing in for a single-SQS run.

    Args:
        model: A fitted model implementing :class:`EnergyModel`.
        reference: The brute-force reference dataset.
        temperature: Temperature in kelvin for the free-energy average.
        cutoff: Edge distance cutoff for graph construction; must match training.
        seed: RNG seed for the single-draw baseline.

    Returns:
        One :class:`FreeEnergyEstimate` per ``(composition, v)`` group.
    """
    rng = np.random.default_rng(seed)
    results: list[FreeEnergyEstimate] = []
    for group in group_by_composition_and_v(reference):
        arrangements = [reference.arrangements[i] for i in group.indices]
        true_energies = np.array([a.energy_ev for a in arrangements], dtype=np.float64)
        graphs = [build_graph(a, cutoff=cutoff) for a in arrangements]
        pred_energies = model.predict(graphs)

        truth = free_energy(true_energies, temperature)
        model_estimate = free_energy(pred_energies, temperature)
        draw = true_energies[rng.integers(0, true_energies.size)]
        single_draw_estimate = free_energy([draw], temperature)

        results.append(
            FreeEnergyEstimate(
                composition=group.composition,
                v=group.v,
                truth=truth,
                model_estimate=model_estimate,
                single_draw_estimate=single_draw_estimate,
                error=model_estimate - truth,
            )
        )
    return results


@dataclass(frozen=True)
class ConvergenceCurve:
    """``G(v)`` estimate vs. number of scored arrangements, for one group."""

    composition: str
    v: int
    truth: float
    sample_sizes: NDArray[np.int64]
    model_curve: NDArray[np.float64]
    random_curve: NDArray[np.float64]


def oracle_efficiency_curves(
    model: EnergyModel,
    reference: Dataset,
    *,
    temperature: float,
    cutoff: float = 5.0,
    seed: int = 0,
) -> list[ConvergenceCurve]:
    """``G(v)`` convergence vs. number of arrangements scored, model vs. random order.

    Both curves use the same reference labels; the "model" curve orders
    arrangements by how the model's predicted energies would be produced (i.e. what
    a user gets by scoring model predictions incrementally), while "random" reflects
    scoring randomly drawn reference arrangements directly (no model). This isolates
    how many oracle (CHGNet) calls the model saves for a target accuracy: with a
    trained model, an equally accurate ``G(v)`` typically needs far fewer of the
    expensive reference labels, since the Boltzmann sum can lean on cheap model
    predictions instead (PLAN.md Section 7, "oracle-efficiency").

    Args:
        model: A fitted model implementing :class:`EnergyModel`.
        reference: The brute-force reference dataset.
        temperature: Temperature in kelvin.
        cutoff: Edge distance cutoff for graph construction; must match training.
        seed: RNG seed controlling the random-order curve.

    Returns:
        One :class:`ConvergenceCurve` per ``(composition, v)`` group.
    """
    curves: list[ConvergenceCurve] = []
    for i, group in enumerate(group_by_composition_and_v(reference)):
        arrangements = [reference.arrangements[i] for i in group.indices]
        true_energies = np.array([a.energy_ev for a in arrangements], dtype=np.float64)
        graphs = [build_graph(a, cutoff=cutoff) for a in arrangements]
        pred_energies = model.predict(graphs)

        truth = free_energy(true_energies, temperature)
        sizes, model_curve = free_energy_convergence(
            pred_energies, temperature, seed=seed + i
        )
        _, random_curve = free_energy_convergence(
            true_energies, temperature, seed=seed + i
        )

        curves.append(
            ConvergenceCurve(
                composition=group.composition,
                v=group.v,
                truth=truth,
                sample_sizes=sizes,
                model_curve=model_curve,
                random_curve=random_curve,
            )
        )
    return curves


@dataclass(frozen=True)
class MinVsAverageDivergence:
    """Per-composition divergence between the T->0 min and the reactor-T average.

    See PLAN.md Section 2.1: a large ``divergence`` flags an entropy-dominated
    composition where taking the lowest arrangement would badly misrank the
    result; a small one flags a min-dominated composition where the two nearly
    coincide.
    """

    composition: str
    v: int
    g_zero_t: float
    g_reactor_t: float
    divergence: float
    entropy_dominated: bool


def min_vs_average_divergence(
    reference: Dataset,
    *,
    reactor_temperature: float,
    entropy_dominated_threshold_ev: float = 0.05,
) -> list[MinVsAverageDivergence]:
    """Quantify, per ``(composition, v)`` group, how far the min departs from G(T).

    Args:
        reference: The brute-force reference dataset.
        reactor_temperature: The reactor temperature in kelvin (e.g.
            :data:`vacancy_gnn.physics.constants.T_AR` or ``T_FR``).
        entropy_dominated_threshold_ev: Divergence above this magnitude (eV) is
            flagged as entropy-dominated; below it, the composition is considered
            safely approximated by the T->0 minimum.

    Returns:
        One :class:`MinVsAverageDivergence` per ``(composition, v)`` group.
    """
    results: list[MinVsAverageDivergence] = []
    for group in group_by_composition_and_v(reference):
        energies = np.array(
            [reference.arrangements[i].energy_ev for i in group.indices],
            dtype=np.float64,
        )
        g_zero = free_energy(energies, 0.0)
        g_reactor = free_energy(energies, reactor_temperature)
        divergence = g_reactor - g_zero
        results.append(
            MinVsAverageDivergence(
                composition=group.composition,
                v=group.v,
                g_zero_t=g_zero,
                g_reactor_t=g_reactor,
                divergence=divergence,
                entropy_dominated=abs(divergence) > entropy_dominated_threshold_ev,
            )
        )
    return results


@dataclass(frozen=True)
class TemperatureSweep:
    """``G(v; T)`` across a temperature grid for one ``(composition, v)`` group."""

    composition: str
    v: int
    temperatures: NDArray[np.float64]
    free_energies: NDArray[np.float64]


def temperature_sweeps(
    reference: Dataset, *, temperatures: NDArray[np.float64] | list[float]
) -> list[TemperatureSweep]:
    """The T-sweep validation from PLAN.md Section 2.1, for every reference group.

    Args:
        reference: The brute-force reference dataset.
        temperatures: Temperature grid in kelvin, e.g.
            ``np.linspace(0.0, 2000.0, 50)``.

    Returns:
        One :class:`TemperatureSweep` per ``(composition, v)`` group.
    """
    temps = np.asarray(temperatures, dtype=np.float64).ravel()
    sweeps: list[TemperatureSweep] = []
    for group in group_by_composition_and_v(reference):
        energies = [reference.arrangements[i].energy_ev for i in group.indices]
        sweeps.append(
            TemperatureSweep(
                composition=group.composition,
                v=group.v,
                temperatures=temps,
                free_energies=free_energy_sweep(energies, temps),
            )
        )
    return sweeps


@dataclass(frozen=True)
class EvaluationReport:
    """Full evaluation harness output (PLAN.md Section 7)."""

    parity: ParityResult
    free_energy_accuracy: list[FreeEnergyEstimate]
    oracle_efficiency: list[ConvergenceCurve]
    min_vs_average: list[MinVsAverageDivergence]
    temperature_sweeps: list[TemperatureSweep] = field(repr=False)


def evaluate(
    model: EnergyModel,
    reference: Dataset,
    *,
    reactor_temperature: float,
    cutoff: float = 5.0,
    sweep_temperatures: NDArray[np.float64] | list[float] | None = None,
    seed: int = 0,
) -> EvaluationReport:
    """Run the full evaluation harness against a brute-force reference dataset.

    Args:
        model: A fitted model implementing :class:`EnergyModel`.
        reference: The brute-force reference dataset (many arrangements per
            ``(composition, v)``).
        reactor_temperature: Reactor temperature in kelvin for G(v) comparisons.
        cutoff: Edge distance cutoff for graph construction; must match training.
        sweep_temperatures: Temperature grid for the T-sweep; defaults to
            ``0`` through ``2 * reactor_temperature``.
        seed: RNG seed for the single-draw and random-order baselines.

    Returns:
        An :class:`EvaluationReport` bundling every metric in PLAN.md Section 7.
    """
    if sweep_temperatures is None:
        sweep_temperatures = np.linspace(0.0, 2.0 * reactor_temperature, 50)

    return EvaluationReport(
        parity=per_arrangement_parity(model, reference, cutoff=cutoff),
        free_energy_accuracy=free_energy_accuracy(
            model, reference, temperature=reactor_temperature, cutoff=cutoff, seed=seed
        ),
        oracle_efficiency=oracle_efficiency_curves(
            model, reference, temperature=reactor_temperature, cutoff=cutoff, seed=seed
        ),
        min_vs_average=min_vs_average_divergence(
            reference, reactor_temperature=reactor_temperature
        ),
        temperature_sweeps=temperature_sweeps(
            reference, temperatures=sweep_temperatures
        ),
    )
