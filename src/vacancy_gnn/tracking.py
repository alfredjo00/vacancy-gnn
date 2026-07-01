"""Experiment-tracking abstraction.

A thin protocol over the tracking backend so the training loop does not depend on
any specific tool. The default :class:`NullTracker` writes nothing (used in tests
and CI); W&B / MLflow backends are added behind the same interface when the
tracking decision is made (PLAN.md Section 8/9).
"""

from __future__ import annotations

from typing import Protocol


class Tracker(Protocol):
    """Minimal experiment-tracking interface."""

    def log_params(self, params: dict[str, object]) -> None:
        """Record run configuration."""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Record scalar metrics at a training step."""
        ...

    def finish(self) -> None:
        """Close the run."""
        ...


class NullTracker:
    """A tracker that records nothing. Default for tests and CI."""

    def __init__(self) -> None:
        self.params: dict[str, object] = {}
        self.history: list[tuple[int, dict[str, float]]] = []

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.history.append((step, dict(metrics)))

    def finish(self) -> None:
        return None


def get_tracker(backend: str) -> Tracker:
    """Return a tracker for the named backend.

    Args:
        backend: One of ``"none"``, ``"wandb"``, ``"mlflow"``.

    Returns:
        A :class:`Tracker`.

    Raises:
        ValueError: For an unknown backend.
        NotImplementedError: For a known-but-not-yet-wired backend.
    """
    if backend == "none":
        return NullTracker()
    if backend in {"wandb", "mlflow"}:
        raise NotImplementedError(
            f"tracking backend '{backend}' is not wired yet; see PLAN.md Section 9"
        )
    raise ValueError(f"unknown tracking backend: {backend!r}")
