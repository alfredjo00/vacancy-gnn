"""Regression metrics for per-arrangement energy prediction and G(v) convergence.

Kept numpy-only (no matplotlib) so it stays on the core import path; plotting code
that consumes these arrays lives in ``notebooks/`` and the evaluation harness.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vacancy_gnn.physics.boltzmann import free_energy


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error (eV)."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    _check(yt, yp)
    return float(np.abs(yt - yp).mean())


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error (eV)."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    _check(yt, yp)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def free_energy_convergence(
    energies: ArrayLike,
    temperature: float,
    *,
    sample_sizes: ArrayLike | None = None,
    seed: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """``G(v)`` estimated from growing random subsets of scored arrangements.

    This is the convergence curve from PLAN.md Section 7: as more model-scored
    arrangements are folded into the Boltzmann sum, the estimate should approach
    the brute-force truth (computed by calling this with the full reference set).

    Args:
        energies: 1-D array of arrangement energies (eV), e.g. all brute-force
            labels or all model predictions for one ``(composition, v)`` group.
        temperature: Temperature in kelvin passed to :func:`free_energy`.
        sample_sizes: Subset sizes to evaluate at; defaults to every integer from
            1 to ``len(energies)``.
        seed: RNG seed controlling which arrangements are drawn into each subset.

    Returns:
        A tuple ``(sample_sizes, g_estimates)`` of equal length.

    Raises:
        ValueError: If ``energies`` is empty or a requested sample size exceeds
            the number of available energies.
    """
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size == 0:
        raise ValueError("energies must be non-empty")

    sizes = (
        np.arange(1, e.size + 1, dtype=np.int64)
        if sample_sizes is None
        else np.asarray(sample_sizes, dtype=np.int64).ravel()
    )
    if sizes.size and int(sizes.max()) > e.size:
        raise ValueError(
            f"requested sample size {int(sizes.max())} exceeds available "
            f"energies ({e.size})"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(e.size)
    shuffled = e[order]

    estimates = np.array(
        [free_energy(shuffled[:n], temperature) for n in sizes.tolist()],
        dtype=np.float64,
    )
    return sizes, estimates


def _check(yt: NDArray[np.float64], yp: NDArray[np.float64]) -> None:
    if yt.size == 0:
        raise ValueError("metrics require at least one sample")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
