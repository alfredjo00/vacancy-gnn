# Data

This directory holds the labeled vacancy-arrangement dataset. The package trains
on **exported labels only** and has no CHGNet / MLIP dependency at runtime; the
labels are produced offline by the "data factory" (the existing CHGNet/Phonopy/SQS
pipeline), then exported to the schema below. See PLAN.md Sections 1 and 5.

## Layout

- `sample/` — a small subset committed to the repo for tests and the quickstart.
- `full/` — the full dataset (gitignored; downloadable release asset or
  regenerable via `scripts/export_from_pipeline.py`).

## Schema (one row per labeled arrangement)

| column           | type          | description                                        |
|------------------|---------------|----------------------------------------------------|
| `composition`    | str           | reduced formula, e.g. `Fe12Mn8Al4O44`              |
| `family`         | str           | cation family, e.g. `FeMnAl`                       |
| `v`              | int           | vacancy count (0..7)                               |
| `site_occupancy` | list[int]     | cation species per sublattice site                 |
| `vacancy_sites`  | list[int]     | indices of empty oxygen sites                       |
| `structure`      | bytes / xyz   | geometry (ideal-holed or relaxed; see PLAN.md 5)   |
| `energy_eV`      | float         | CHGNet relaxed energy (label)                       |
| `source_run`     | str           | provenance back to the factory run                  |
