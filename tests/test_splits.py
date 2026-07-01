"""Tests for composition-aware splitting, especially the no-leakage guarantee."""

from __future__ import annotations

import pytest

from vacancy_gnn.data.schema import Dataset
from vacancy_gnn.data.splits import composition_split, compositions_of


def test_no_composition_leakage_across_splits(small_dataset: Dataset) -> None:
    split = composition_split(small_dataset, val_fraction=0.34, test_fraction=0.34)
    train_c = compositions_of(small_dataset, split.train)
    val_c = compositions_of(small_dataset, split.val)
    test_c = compositions_of(small_dataset, split.test)

    # The three composition sets are pairwise disjoint: the core guarantee.
    assert train_c & val_c == set()
    assert train_c & test_c == set()
    assert val_c & test_c == set()


def test_split_covers_every_arrangement_once(small_dataset: Dataset) -> None:
    split = composition_split(small_dataset, val_fraction=0.2, test_fraction=0.2)
    all_idx = sorted(split.train + split.val + split.test)
    assert all_idx == list(range(len(small_dataset)))


def test_split_is_deterministic(small_dataset: Dataset) -> None:
    a = composition_split(small_dataset, seed=7)
    b = composition_split(small_dataset, seed=7)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_different_seed_can_change_partition(small_dataset: Dataset) -> None:
    a = composition_split(small_dataset, val_fraction=0.34, test_fraction=0.34, seed=1)
    b = composition_split(small_dataset, val_fraction=0.34, test_fraction=0.34, seed=2)
    # Not guaranteed different for every pair, but for these seeds it is.
    assert (a.train, a.val, a.test) != (b.train, b.val, b.test)


def test_arrangements_of_same_composition_stay_together(small_dataset: Dataset) -> None:
    split = composition_split(small_dataset, val_fraction=0.34, test_fraction=0.34)
    part_of: dict[str, str] = {}
    for name, idxs in (("tr", split.train), ("va", split.val), ("te", split.test)):
        for i in idxs:
            comp = small_dataset.arrangements[i].composition
            assert part_of.setdefault(comp, name) == name


def test_rejects_infeasible_fractions(small_dataset: Dataset) -> None:
    with pytest.raises(ValueError):
        composition_split(small_dataset, val_fraction=0.6, test_fraction=0.6)


def test_rejects_too_few_compositions(small_dataset: Dataset) -> None:
    # 6 compositions with 0.5/0.5 leaves no room for a train set.
    with pytest.raises(ValueError):
        composition_split(small_dataset, val_fraction=0.5, test_fraction=0.5)
