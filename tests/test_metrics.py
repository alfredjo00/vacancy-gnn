"""Tests for regression metrics."""

from __future__ import annotations

import numpy as np
import pytest

from vacancy_gnn.metrics import mae, rmse


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
