"""Tests for loading offline data-factory exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vacancy_gnn.data.factory import FactoryLoadError, load_factory_export

_CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
_OXYGEN = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def _record(composition: str, subset: str, v: int = 0) -> dict[str, object]:
    return {
        "composition": composition,
        "family": "FeMnAl",
        "subset": subset,
        "v": v,
        "cation_species": [26],
        "cation_positions": [[0.0, 0.0, 0.0]],
        "oxygen_positions": _OXYGEN,
        "vacancy_sites": list(range(v)),
        "cell": _CELL,
        "energy_ev": -1.0,
        "source_run": "test-factory",
    }


def _write(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "factory.json"
    path.write_text(json.dumps({"arrangements": records}))
    return path


def test_splits_into_train_and_reference(tmp_path: Path) -> None:
    records = [
        _record("A", "train"),
        _record("A", "train"),
        _record("B", "reference"),
    ]
    path = _write(tmp_path, records)

    export = load_factory_export(path)

    assert len(export.train) == 2
    assert len(export.reference) == 1
    assert export.train.compositions() == ["A"]
    assert export.reference.compositions() == ["B"]


def test_arrangement_fields_round_trip(tmp_path: Path) -> None:
    path = _write(tmp_path, [_record("A", "train", v=1)])

    export = load_factory_export(path)

    arrangement = export.train.arrangements[0]
    assert arrangement.composition == "A"
    assert arrangement.v == 1
    assert arrangement.vacancy_sites == [0]
    assert arrangement.source_run == "test-factory"


def test_missing_arrangements_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not_arrangements": []}))

    with pytest.raises(FactoryLoadError):
        load_factory_export(path)


def test_unrecognized_subset_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, [_record("A", "validation")])

    with pytest.raises(FactoryLoadError):
        load_factory_export(path)


def test_missing_subset_rejected(tmp_path: Path) -> None:
    record = _record("A", "train")
    del record["subset"]
    path = _write(tmp_path, [record])

    with pytest.raises(FactoryLoadError):
        load_factory_export(path)


def test_invalid_arrangement_raises_validation_error(tmp_path: Path) -> None:
    record = _record("A", "train", v=2)
    record["vacancy_sites"] = [0]  # only one site, but v=2
    path = _write(tmp_path, [record])

    with pytest.raises(Exception, match="vacancy_sites"):
        load_factory_export(path)


def test_committed_sample_loads() -> None:
    path = Path(__file__).parent.parent / "data" / "sample" / "factory_sample.json"

    export = load_factory_export(path)

    assert len(export.train) > 0
    assert len(export.reference) > 0
    assert export.train.compositions() != export.reference.compositions()


def test_missing_e0s_ev_is_none(tmp_path: Path) -> None:
    path = _write(tmp_path, [_record("A", "train")])

    export = load_factory_export(path)

    assert export.e0s_ev is None


def test_e0s_ev_is_surfaced(tmp_path: Path) -> None:
    path = tmp_path / "factory.json"
    path.write_text(
        json.dumps(
            {
                "arrangements": [_record("A", "train")],
                "e0s_ev": {"O": -1.0, "Fe": -3.0},
            }
        )
    )

    export = load_factory_export(path)

    assert export.e0s_ev == {"O": -1.0, "Fe": -3.0}
