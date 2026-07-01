"""Composition-aware train/val/test splits.

Splitting must never place two arrangements of the *same composition* into
different splits: arrangements of one composition are highly correlated, so a
random per-arrangement split would leak information and inflate reported
generalization (PLAN.md Section 4). We therefore split at the composition level and
assign all arrangements of a composition to the same partition.

The split is deterministic given ``seed`` so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vacancy_gnn.data.schema import Arrangement, Dataset


@dataclass(frozen=True)
class Split:
    """Index partitions into a dataset's ``arrangements`` list."""

    train: list[int]
    val: list[int]
    test: list[int]

    def sizes(self) -> tuple[int, int, int]:
        return len(self.train), len(self.val), len(self.test)


def composition_split(
    dataset: Dataset,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> Split:
    """Partition arrangements by composition into train/val/test.

    Args:
        dataset: The dataset to split.
        val_fraction: Fraction of *compositions* (not arrangements) held out for
            validation.
        test_fraction: Fraction of compositions held out for test.
        seed: RNG seed for the composition shuffle.

    Returns:
        A :class:`Split` of integer indices into ``dataset.arrangements``.

    Raises:
        ValueError: If the fractions are invalid, or there are too few compositions
            to populate every requested non-empty split.
    """
    if not (0.0 <= val_fraction < 1.0) or not (0.0 <= test_fraction < 1.0):
        raise ValueError("fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    comps = dataset.compositions()
    n_comp = len(comps)

    n_test = round(test_fraction * n_comp)
    n_val = round(val_fraction * n_comp)
    if n_val + n_test >= n_comp:
        raise ValueError(
            f"{n_comp} compositions cannot fill val={n_val}, test={n_test}, "
            "and a non-empty train set; provide more compositions or smaller "
            "fractions"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_comp)
    shuffled = [comps[i] for i in order]

    test_comps = set(shuffled[:n_test])
    val_comps = set(shuffled[n_test : n_test + n_val])

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for i, a in enumerate(dataset.arrangements):
        if a.composition in test_comps:
            test_idx.append(i)
        elif a.composition in val_comps:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return Split(train=train_idx, val=val_idx, test=test_idx)


def compositions_of(dataset: Dataset, indices: list[int]) -> set[str]:
    """Set of compositions covered by a list of arrangement indices."""
    arr: list[Arrangement] = dataset.arrangements
    return {arr[i].composition for i in indices}
