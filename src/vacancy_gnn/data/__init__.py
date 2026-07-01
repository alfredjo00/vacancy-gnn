"""Dataset schema, featurization, and composition-aware splits."""

from __future__ import annotations

from vacancy_gnn.data.featurize import Graph, build_graph
from vacancy_gnn.data.schema import Arrangement, Dataset
from vacancy_gnn.data.splits import Split, composition_split, compositions_of

__all__ = [
    "Arrangement",
    "Dataset",
    "Graph",
    "Split",
    "build_graph",
    "composition_split",
    "compositions_of",
]
