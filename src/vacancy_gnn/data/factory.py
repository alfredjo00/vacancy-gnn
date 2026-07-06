"""Loader for offline data-factory exports (PLAN.md step 7).

:mod:`scripts.generate_factory_data` writes a JSON file of arrangement records
that are almost the :class:`~vacancy_gnn.data.schema.Arrangement` schema, plus
one extra ``subset`` tag (``"train"`` or ``"reference"``) marking whether a
record belongs to the breadth pool used for training or the depth pool used as
the brute-force ``G(v)`` reference (PLAN.md Section 5). This module reads that
export and splits it into the two :class:`~vacancy_gnn.data.schema.Dataset`\\ s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import TypeAdapter

from vacancy_gnn.data.schema import Arrangement, Dataset

Subset = Literal["train", "reference"]

_RECORD_ADAPTER = TypeAdapter(dict[str, object])


class FactoryLoadError(ValueError):
    """Raised when a factory export is missing required structure."""


class FactoryExport(NamedTuple):
    """The two datasets and run metadata read from a factory export."""

    train: Dataset
    reference: Dataset
    #: Isolated-atom reference energies (eV) per element symbol, keyed by
    #: symbol (e.g. ``"O"``, ``"Fe"``), as computed by
    #: :func:`scripts.generate_factory_data.compute_e0s_ev`. ``None`` for
    #: exports written before this field existed (e.g. the original
    #: ``factory_v2.json``); consumers should fall back to an unanchored
    #: :meth:`~vacancy_gnn.models.reference.CompositionReference.fit`.
    e0s_ev: dict[str, float] | None


def load_factory_export(path: str | Path) -> FactoryExport:
    """Load a factory JSON export and split it into (train, reference) datasets.

    Args:
        path: Path to a JSON file with the ``{"arrangements": [...]}`` shape
            written by :mod:`scripts.generate_factory_data`. Each record must
            carry every :class:`Arrangement` field plus a ``subset`` field set
            to ``"train"`` or ``"reference"``. May also carry a top-level
            ``e0s_ev`` mapping.

    Returns:
        A :class:`FactoryExport` with validated ``train``/``reference``
        :class:`Dataset`\\ s. The reference dataset holds the densely sampled
        compositions meant for the brute-force ``G(v)`` evaluation harness
        (PLAN.md Section 7); the train dataset holds the broad, shallow pool
        for fitting models.

    Raises:
        FactoryLoadError: If the file does not have the expected top-level
            ``arrangements`` key, or a record has an unrecognized ``subset``.
        pydantic.ValidationError: If a record fails :class:`Arrangement`
            validation.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or "arrangements" not in payload:
        raise FactoryLoadError(
            f"{path}: expected a JSON object with an 'arrangements' key"
        )

    train: list[Arrangement] = []
    reference: list[Arrangement] = []
    for i, raw in enumerate(payload["arrangements"]):
        record = _RECORD_ADAPTER.validate_python(raw)
        subset = record.pop("subset", None)
        if subset not in ("train", "reference"):
            raise FactoryLoadError(
                f"{path}: record {i} has subset={subset!r}, "
                "expected 'train' or 'reference'"
            )
        arrangement = Arrangement.model_validate(record)
        (train if subset == "train" else reference).append(arrangement)

    e0s_ev = payload.get("e0s_ev")
    return FactoryExport(
        train=Dataset(arrangements=train),
        reference=Dataset(arrangements=reference),
        e0s_ev=e0s_ev,
    )
