"""Validation tests for the dataset schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vacancy_gnn.data.schema import Arrangement, Dataset

from .conftest import ArrangementFactory

# Valid geometry defaults so each test exercises the validator it targets rather
# than tripping on a missing required field.
_CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
_OXYGEN = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_valid_arrangement_round_trips(make_arrangement: ArrangementFactory) -> None:
    a = make_arrangement("Comp0", "FeMnAl", vacancy_sites=[0, 1])
    restored = Arrangement.model_validate(a.model_dump())
    assert restored == a


def test_vacancy_count_must_match_v() -> None:
    with pytest.raises(ValidationError):
        Arrangement(
            composition="X",
            family="F",
            v=3,
            cation_species=[26],
            cation_positions=[[0.0, 0.0, 0.0]],
            oxygen_positions=_OXYGEN,
            vacancy_sites=[0, 1],  # only 2, but v=3
            cell=_CELL,
            energy_ev=-1.0,
        )


def test_vacancy_sites_must_be_unique() -> None:
    with pytest.raises(ValidationError):
        Arrangement(
            composition="X",
            family="F",
            v=2,
            cation_species=[26],
            cation_positions=[[0.0, 0.0, 0.0]],
            oxygen_positions=_OXYGEN,
            vacancy_sites=[1, 1],
            cell=_CELL,
            energy_ev=-1.0,
        )


def test_vacancy_site_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Arrangement(
            composition="X",
            family="F",
            v=1,
            cation_species=[26],
            cation_positions=[[0.0, 0.0, 0.0]],
            oxygen_positions=_OXYGEN,  # 3 sites -> valid indices are 0..2
            vacancy_sites=[3],
            cell=_CELL,
            energy_ev=-1.0,
        )


def test_positions_must_match_species_length() -> None:
    with pytest.raises(ValidationError):
        Arrangement(
            composition="X",
            family="F",
            v=0,
            cation_species=[26, 25],
            cation_positions=[[0.0, 0.0, 0.0]],  # only one row for two species
            oxygen_positions=_OXYGEN,
            vacancy_sites=[],
            cell=_CELL,
            energy_ev=-1.0,
        )


def test_positions_need_three_coords() -> None:
    with pytest.raises(ValidationError):
        Arrangement(
            composition="X",
            family="F",
            v=0,
            cation_species=[26],
            cation_positions=[[0.0, 0.0]],
            oxygen_positions=_OXYGEN,
            vacancy_sites=[],
            cell=_CELL,
            energy_ev=-1.0,
        )


def test_cell_must_be_three_by_three() -> None:
    with pytest.raises(ValidationError):
        Arrangement(
            composition="X",
            family="F",
            v=0,
            cation_species=[26],
            cation_positions=[[0.0, 0.0, 0.0]],
            oxygen_positions=_OXYGEN,
            vacancy_sites=[],
            cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],  # only two rows
            energy_ev=-1.0,
        )


def test_dataset_composition_and_family_helpers(small_dataset: Dataset) -> None:
    assert len(small_dataset) == 12
    assert small_dataset.compositions() == [f"Comp{c}" for c in range(6)]
    assert small_dataset.families() == {"FeMnAl"}
