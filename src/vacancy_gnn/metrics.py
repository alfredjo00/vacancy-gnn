"""Regression metrics for per-arrangement energy prediction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


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


def _check(yt: np.ndarray, yp: np.ndarray) -> None:
    if yt.size == 0:
        raise ValueError("metrics require at least one sample")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
