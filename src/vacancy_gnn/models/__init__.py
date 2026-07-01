"""Models: cluster-expansion baseline and equivariant GNN behind one interface.

The baseline is framework-free and always importable. The equivariant GNN requires
the optional ``[ml]`` extra (torch), so it is imported lazily: accessing
``vacancy_gnn.models.EquivariantGNN`` triggers the import only on demand, keeping
the core installable and CI-green without torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vacancy_gnn.models.base import EnergyModel
from vacancy_gnn.models.baseline import LinearBaseline

if TYPE_CHECKING:
    from vacancy_gnn.models.egnn import EquivariantGNN

__all__ = ["EnergyModel", "EquivariantGNN", "LinearBaseline"]


def __getattr__(name: str) -> Any:
    if name == "EquivariantGNN":
        try:
            from vacancy_gnn.models.egnn import EquivariantGNN
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "EquivariantGNN requires the optional 'ml' extra: "
                "pip install -e '.[ml]'"
            ) from exc
        return EquivariantGNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
