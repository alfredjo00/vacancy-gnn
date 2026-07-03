# Data

This directory holds the labeled vacancy-arrangement dataset. The package trains
on **exported labels only** and has no MLIP dependency at runtime; the labels are
produced offline by `scripts/generate_factory_data.py` (MACE-MPA-0 relaxations of
holed spinel supercells), then loaded through
`vacancy_gnn.data.factory.load_factory_export`. See PLAN.md Sections 1 and 5.

## Layout

- `sample/factory_sample.json` — a small subset (3 train compositions, 1
  reference composition, reduced depth) committed to the repo for tests and the
  quickstart. Regenerate with `scripts/make_data_sample.py` if `full/` changes.
- `full/` — the full dataset (gitignored; regenerate with
  `scripts/generate_factory_data.py`, which needs ase/pymatgen/mace-torch in a
  throwaway env; see that script's docstring).

## Export format

A JSON file shaped `{"arrangements": [...]}`. Each record is one
`vacancy_gnn.data.schema.Arrangement` plus a `subset` tag of `"train"` or
`"reference"`, consumed by `load_factory_export`, which validates every record
and splits them into a `(train, reference)` pair of `Dataset`s. `reference`
holds a few compositions sampled deeply across all vacancy levels (the
brute-force `G(v)` reference from PLAN.md Section 7); `train` holds many
compositions sampled shallowly.

| field              | type               | description                                                        |
|--------------------|--------------------|--------------------------------------------------------------------|
| `composition`      | str                | composition tag, e.g. `Al2Ca3Co5Cr7Fe6Mg5Ti1V5Zn2O48-factory-000`  |
| `family`           | str                | cation family / factory-run tag                                    |
| `subset`           | str                | `"train"` or `"reference"` (stripped on load)                      |
| `v`                | int                | vacancy count (0..7)                                                |
| `cation_species`   | list[int]          | atomic number per cation site                                       |
| `cation_positions` | list[list[float]]  | Cartesian coordinates of cation sites                               |
| `oxygen_positions` | list[list[float]]  | ideal oxygen sublattice coordinates                                 |
| `vacancy_sites`    | list[int]          | indices into `oxygen_positions` that are empty                     |
| `cell`             | list[list[float]]  | row-major 3x3 lattice vectors                                       |
| `energy_ev`        | float              | MACE-MPA-0 relaxed energy (label)                                   |
| `source_run`       | str                | provenance string back to the factory run                          |
