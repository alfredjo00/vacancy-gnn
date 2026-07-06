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
- `full/` — the full dataset (gitignored; download from the GitHub release, or
  regenerate with `scripts/generate_factory_data.py`, which needs
  ase/pymatgen/mace-torch in a throwaway env; see that script's docstring).

## Getting the full dataset (release assets)

`factory_v3.json` and the trained GNN checkpoint are attached to the
[v0.1.0 release](https://github.com/alfredjo00/vacancy-gnn/releases/tag/v0.1.0),
with a `SHA256SUMS` file to verify against:

```bash
mkdir -p data/full
curl -L https://github.com/alfredjo00/vacancy-gnn/releases/download/v0.1.0/factory_v3.json.gz \
  | gunzip > data/full/factory_v3.json
curl -L https://github.com/alfredjo00/vacancy-gnn/releases/download/v0.1.0/egnn_v3_checkpoint.tar.gz \
  | tar xz -C data/full

vacancy-gnn evaluate --data data/full/factory_v3.json
```

### factory_v3.json

2160 relaxed arrangements of a 3x2x1 spinel supercell (36 cation sites, 48
oxygen sites) over 32 compositions from a 17-element cation pool:

- `subset="train"`: 30 compositions x 8 vacancy levels (v = 0..7) x 5
  arrangements (1200 rows), the breadth pool for fitting.
- `subset="reference"`: 2 held-out compositions x 8 levels x 60 arrangements
  (960 rows), the dense brute-force reference for the evaluation harness.

Labels: MACE-MPA-0 (`medium-mpa-0`, float64), BFGS relaxation of cell and ions
to fmax = 0.05 eV/A from the ideal holed lattice. Stored positions are the
**ideal** geometry (the locked ideal-geometry -> relaxed-energy target). The
`...-extend` suffix in `source_run` marks the 14 training compositions
appended to the original v2 run with `generate_factory_data.py --extend`.

### egnn_v3_checkpoint.tar.gz

`egnn_v3.json` + `egnn_v3.pt`: the `EquivariantGNN` behind the README results
table and the money-figure notebook (hidden 64, 3 layers, 600 epochs on the 30
training compositions; load with `EquivariantGNN.load`). Retraining from the
dataset takes about 3 minutes on a consumer GPU.

## Export format

A JSON file shaped `{"arrangements": [...], "e0s_ev": {...}}`. `e0s_ev` maps
element symbols to isolated-atom reference energies (eV) from the same
calculator, used to build the optional composition-reference prior
(`vacancy_gnn.models.reference.prior_from_e0s`). Each record is one
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
