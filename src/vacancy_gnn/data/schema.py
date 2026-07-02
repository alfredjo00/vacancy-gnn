"""Dataset schema for labeled vacancy arrangements.

One :class:`Arrangement` is a single vacancy decoration of a composition at a
fixed vacancy level, together with its CHGNet energy label. The package trains on
these exported records only and never calls an interatomic potential at runtime
(PLAN.md Sections 1 and 5).

The pydantic models validate the exported data at load time so a malformed factory
export fails loudly and early rather than silently corrupting training.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Arrangement(BaseModel):
    """A single labeled vacancy arrangement.

    The record is self-contained: ``cell`` and ``oxygen_positions`` mean a
    consumer can reconstruct the full periodic geometry (cations plus the ideal
    oxygen sublattice, minus the vacant sites) without reaching back into the
    offline factory's prototype structure (PLAN.md Section 5).

    Attributes:
        composition: Reduced formula, e.g. ``"Fe12Mn8Al4O44"``.
        family: Cation family tag used for scoping and splits, e.g. ``"FeMnAl"``.
        v: Vacancy count (number of empty oxygen sites).
        cation_species: Atomic number per cation site, length = number of cation
            sites. Order is fixed by the reference lattice.
        cation_positions: Cartesian coordinates of the cation sites, shape
            ``(n_cations, 3)``.
        oxygen_positions: Cartesian coordinates of every ideal oxygen sublattice
            site, shape ``(n_oxygen_sites, 3)``. ``vacancy_sites`` indexes into
            this array; the occupied oxygen sites are the complement.
        vacancy_sites: Indices (into ``oxygen_positions``) of the empty oxygen
            sites. Length must equal ``v`` and entries must be unique.
        cell: Row-major ``3x3`` lattice vectors of the periodic supercell, in the
            same units as the positions. Used for minimum-image graph
            construction.
        energy_ev: MLIP relaxed energy label (eV).
        source_run: Provenance string back to the factory run.
    """

    model_config = ConfigDict(frozen=True)

    composition: str
    family: str
    v: int = Field(ge=0)
    cation_species: list[int]
    cation_positions: list[list[float]]
    oxygen_positions: list[list[float]]
    vacancy_sites: list[int]
    cell: list[list[float]]
    energy_ev: float
    source_run: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> Arrangement:
        if len(self.vacancy_sites) != self.v:
            raise ValueError(
                f"vacancy_sites has {len(self.vacancy_sites)} entries but v={self.v}"
            )
        if len(set(self.vacancy_sites)) != len(self.vacancy_sites):
            raise ValueError("vacancy_sites must be unique")
        if any(s < 0 for s in self.vacancy_sites):
            raise ValueError("vacancy_sites must be non-negative")
        n_oxygen = len(self.oxygen_positions)
        if any(s >= n_oxygen for s in self.vacancy_sites):
            raise ValueError(
                "vacancy_sites index out of range for oxygen_positions "
                f"(have {n_oxygen} sites)"
            )
        if any(len(row) != 3 for row in self.oxygen_positions):
            raise ValueError("each oxygen position must have 3 coordinates")
        n = len(self.cation_species)
        if len(self.cation_positions) != n:
            raise ValueError(
                f"cation_positions has {len(self.cation_positions)} rows "
                f"but cation_species has {n}"
            )
        if any(len(row) != 3 for row in self.cation_positions):
            raise ValueError("each cation position must have 3 coordinates")
        if len(self.cell) != 3 or any(len(row) != 3 for row in self.cell):
            raise ValueError("cell must be a 3x3 matrix")
        return self

    def positions_array(self) -> NDArray[np.float64]:
        """Cation positions as a ``(n_cations, 3)`` float array."""
        return np.asarray(self.cation_positions, dtype=np.float64)

    def species_array(self) -> NDArray[np.int64]:
        """Cation atomic numbers as a ``(n_cations,)`` int array."""
        return np.asarray(self.cation_species, dtype=np.int64)

    def oxygen_positions_array(self) -> NDArray[np.float64]:
        """Oxygen sublattice positions as a ``(n_oxygen_sites, 3)`` float array."""
        return np.asarray(self.oxygen_positions, dtype=np.float64)

    def cell_array(self) -> NDArray[np.float64]:
        """Lattice vectors as a row-major ``(3, 3)`` float array."""
        return np.asarray(self.cell, dtype=np.float64)


class Dataset(BaseModel):
    """A validated collection of arrangements from one or more families."""

    model_config = ConfigDict(frozen=True)

    arrangements: list[Arrangement]

    def __len__(self) -> int:
        return len(self.arrangements)

    def compositions(self) -> list[str]:
        """Unique compositions present, in first-seen order."""
        seen: dict[str, None] = {}
        for a in self.arrangements:
            seen.setdefault(a.composition, None)
        return list(seen)

    def families(self) -> set[str]:
        """Set of distinct family tags present."""
        return {a.family for a in self.arrangements}
