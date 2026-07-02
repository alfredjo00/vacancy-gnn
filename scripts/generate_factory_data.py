#!/usr/bin/env python3
"""Offline data factory: build vacancy arrangements and label them with MACE-MPA-0.

This is the "factory" from PLAN.md Section 5: the one-time, offline step that
turns a spinel prototype structure into a labeled dataset of
(composition, vacancy arrangement) -> relaxed energy samples, exported in
vacancy_gnn's schema. It runs once and its output feeds the package's data/
directory; nothing in vacancy_gnn itself imports MACE or ase at runtime.

The labeler is the MACE-MPA-0 universal MLIP foundation model (Matbench
Discovery F1 ~0.85, vs. CHGNet's ~0.61), loaded via mace-torch's ASE calculator
interface. No account/license gating is needed to download it (unlike some
higher-ranked models such as eSEN-30M-OAM, which require accepting a Hugging
Face license). Relaxation is BFGS on cell + ions to fmax=0.05 eV/A (see
RELAX_FMAX).

Scope: a 3x2x1 supercell of the MgAl2O4 spinel prototype
(prototypes.db, from Materials Project
mp-3536), 84 atoms (12 A-site, 24 B-site, 48 O). Each composition fixes a random
stoichiometry of 4-9 unique cation elements (drawn from the A-site/B-site element pools
defined below); each
arrangement within that composition then independently shuffles the site
occupancy (SQS-style) and draws a random vacancy set, so arrangements vary in
both cation ordering and vacancy placement.

The run is bimodal (IMPROVEMENTS.md P1): many compositions with a few
arrangements each (breadth, for held-out-composition generalization) plus a
couple of compositions sampled densely (depth, so G(v) has a real brute-force
reference per PLAN.md Section 7). The reference compositions come last and are
distinct, so they can be held out of training cleanly.

Records are self-contained: each carries ``cell`` and ``oxygen_positions`` (the
ideal sublattice ``vacancy_sites`` index into), so vacancy_gnn can rebuild the
full periodic geometry without this script's prototype db. Because the script
builds the ideal lattice itself, vacancy_sites are exact indices with no
position-matching needed.

Not part of the installed package: needs ase, pymatgen, and mace-torch, none of
which are runtime or `dev`/`ml` extras. Install them in a throwaway environment
to run this script, e.g.:

    pip install ase pymatgen mace-torch

Usage:
    python scripts/generate_factory_data.py --out data/full/factory_v2.json

Writes partial results to ``out_path`` every 10 relaxations, so a crash or
interrupt loses at most that much GPU work; rerun with --resume to continue.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROTOTYPE_DB = Path("prototypes.db")
SUPERCELL_DIMENSIONS: tuple[int, int, int] = (3, 2, 1)
VACANCY_LEVELS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
#: Number of unique cation elements (summed over both sites) per composition,
#: inclusive. Bounds a realistic "high entropy" arity; without this cap,
#: independently drawing each site from its full element pool nearly always
#: yields 14-16 unique elements, which is not a plausible HEO composition.
MIN_UNIQUE_ELEMENTS = 4
MAX_UNIQUE_ELEMENTS = 9

# Real species only: this script assigns species directly
# minus proxy elements (this script assigns real species directly and does not
# use icet/SQS, so no proxy juggling is needed).
A_SITE_ELEMENTS: tuple[str, ...] = (
    "Mg",
    "Zn",
    "Cu",
    "Fe",
    "Mn",
    "Co",
    "Li",
    "Ga",
    "Ni",
    "In",
)
B_SITE_ELEMENTS: tuple[str, ...] = (
    "Cr",
    "Ti",
    "Al",
    "Zr",
    "V",
    "Sn",
    "Ca",
    "In",
    "Cu",
    "Fe",
    "Mn",
    "Mg",
    "Ni",
    "Co",
    "Li",
    "Ga",
)


@dataclass(frozen=True)
class CompositionSpec:
    """A fixed stoichiometry on the supercell's cation sites.

    ``cation_counts`` is the multiset of cation species (fixed counts, e.g.
    six Fe, six Mg, ...); ``cation_positions``/``a_site_mask`` are the
    lattice's cation site positions and which of them are A-site (the rest
    are B-site). Each arrangement independently shuffles ``cation_counts``
    onto the sites (subject to the A/B site split), so different
    arrangements of the same composition have identical stoichiometry but
    different site occupancy (SQS-style), matching real configurational
    sampling instead of freezing one decoration per composition.
    """

    tag: str
    cation_counts: list[int]
    cation_positions: list[list[float]]
    a_site_mask: list[bool]
    oxygen_positions: list[list[float]]
    cell: list[list[float]]


def _composition_tag(cation_species: list[int], n_oxygen: int, index: int) -> str:
    """Human-readable composition tag, e.g. ``Al12Fe6Mn3O48-factory-000``."""
    from ase.data import chemical_symbols

    counts: dict[str, int] = {}
    for z in cation_species:
        sym = chemical_symbols[z]
        counts[sym] = counts.get(sym, 0) + 1
    formula = "".join(f"{sym}{n}" for sym, n in sorted(counts.items()))
    return f"{formula}O{n_oxygen}-factory-{index:03d}"


def _pick_element_subset(rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Pick a restricted (A-site pool, B-site pool) pair for one composition.

    Draws a random total arity in [MIN_UNIQUE_ELEMENTS, MAX_UNIQUE_ELEMENTS]
    (inclusive) from the union of the two site pools, then splits that subset
    back into its A-site- and B-site-eligible members. Elements valid on both
    sites (e.g. Fe, Cu) count once toward the arity but remain usable on
    either site. Retries until both sites have at least one eligible element.
    """
    all_elements = sorted(set(A_SITE_ELEMENTS) | set(B_SITE_ELEMENTS))
    while True:
        arity = int(rng.integers(MIN_UNIQUE_ELEMENTS, MAX_UNIQUE_ELEMENTS + 1))
        subset = set(rng.choice(all_elements, size=arity, replace=False).tolist())
        a_pool = [e for e in A_SITE_ELEMENTS if e in subset]
        b_pool = [e for e in B_SITE_ELEMENTS if e in subset]
        if a_pool and b_pool:
            return a_pool, b_pool


def build_composition(rng: np.random.Generator, index: int) -> CompositionSpec:
    """Fix one random stoichiometry on the 3x2x1 spinel supercell's cation sites.

    Restricted to MIN_UNIQUE_ELEMENTS-MAX_UNIQUE_ELEMENTS unique cation
    elements total, matching realistic high-entropy-oxide arity (see
    _pick_element_subset). Returns fixed per-element counts; the site
    occupancy itself is shuffled per arrangement by
    :func:`shuffle_cation_arrangement`.
    """
    import ase.db
    from ase.data import atomic_numbers

    db = ase.db.connect(str(PROTOTYPE_DB))
    row = next(iter(db.select(limit=1)))
    prototype = row.toatoms()
    supercell = prototype.repeat(SUPERCELL_DIMENSIONS)

    numbers = supercell.numbers
    positions = supercell.positions

    # In the mp-3536 primitive cell, Mg (Z=12) sits on the A-site and Al (Z=13)
    # on the B-site; this holds after repeat() since species are per-atom.
    a_mask = numbers == 12
    b_mask = numbers == 13
    o_mask = numbers == 8
    cation_mask = a_mask | b_mask

    a_pool, b_pool = _pick_element_subset(rng)
    a_choices = rng.choice(a_pool, size=int(a_mask.sum()))
    b_choices = rng.choice(b_pool, size=int(b_mask.sum()))

    new_numbers = numbers.copy()
    new_numbers[a_mask] = [atomic_numbers[e] for e in a_choices]
    new_numbers[b_mask] = [atomic_numbers[e] for e in b_choices]

    cation_counts = new_numbers[cation_mask].tolist()
    cation_positions = positions[cation_mask].tolist()
    a_site_mask = a_mask[cation_mask].tolist()
    oxygen_positions = positions[o_mask].tolist()

    tag = _composition_tag(cation_counts, len(oxygen_positions), index)
    return CompositionSpec(
        tag=tag,
        cation_counts=cation_counts,
        cation_positions=cation_positions,
        a_site_mask=a_site_mask,
        oxygen_positions=oxygen_positions,
        cell=supercell.cell.tolist(),
    )


def shuffle_cation_arrangement(
    rng: np.random.Generator, spec: CompositionSpec
) -> list[int]:
    """Randomly permute ``spec.cation_counts`` onto sites, respecting the A/B split.

    Shuffles A-site elements only among A-site positions and B-site elements
    only among B-site positions (an element valid on both sites can appear in
    both sub-shuffles), so every arrangement has the same stoichiometry as
    the composition but a different site occupancy.
    """
    counts = np.array(spec.cation_counts)
    a_mask = np.array(spec.a_site_mask)

    arranged = np.empty_like(counts)
    a_values = counts[a_mask].copy()
    b_values = counts[~a_mask].copy()
    rng.shuffle(a_values)
    rng.shuffle(b_values)
    arranged[a_mask] = a_values
    arranged[~a_mask] = b_values
    return arranged.tolist()


def build_arrangement_atoms(
    spec: CompositionSpec, cation_species: list[int], vacancy_sites: list[int]
) -> object:
    """Build an ase.Atoms for one (composition, cation shuffle, vacancy) triple."""
    import ase

    vac = set(vacancy_sites)
    o_positions = [p for i, p in enumerate(spec.oxygen_positions) if i not in vac]
    numbers = cation_species + [8] * len(o_positions)
    positions = spec.cation_positions + o_positions
    return ase.Atoms(numbers=numbers, positions=positions, cell=spec.cell, pbc=True)


#: Force-convergence threshold (eV/A) for the geometry relaxation. MACE-MPA-0's
#: own force error vs DFT is tens of meV/A and Matbench Discovery relaxes at 0.05,
#: so converging tighter buys precision below the label-noise floor at several
#: times the walltime. 0.05 keeps the energies well within noise while roughly
#: tripling throughput (IMPROVEMENTS.md P2).
RELAX_FMAX = 0.05
SOURCE_RUN = "generate_factory_data.py-v2-mace-mpa-0-fmax0.05"


def relax_energy_ev(atoms: object, calc: object) -> float:
    """Relax ``atoms`` with MACE-MPA-0 (BFGS, cell + ions) and return final energy."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    atoms.calc = calc  # type: ignore[attr-defined]
    ecf = FrechetCellFilter(atoms)
    opt = BFGS(ecf, logfile=None)
    opt.run(fmax=RELAX_FMAX, steps=1000)
    return float(atoms.get_potential_energy())  # type: ignore[attr-defined]


def _write_checkpoint(records: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps({"arrangements": records}))
    tmp_path.replace(out_path)


def _composition_plan(
    n_train: int, train_per_level: int, n_reference: int, reference_per_level: int
) -> list[tuple[int, int]]:
    """Bimodal per-composition arrangement counts (IMPROVEMENTS.md P1).

    Returns an ordered list of ``(composition_index, arrangements_per_level)``:
    ``n_train`` compositions with ``train_per_level`` arrangements each (breadth,
    for held-out-composition generalization), then ``n_reference`` compositions
    with ``reference_per_level`` each (depth, so ``G(v)`` has a real brute-force
    reference per PLAN.md Section 7). The reference compositions come last and are
    distinct, so they can be held out of training cleanly.
    """
    plan = [(c, train_per_level) for c in range(n_train)]
    plan += [(n_train + c, reference_per_level) for c in range(n_reference)]
    return plan


def generate(
    n_train: int,
    train_per_level: int,
    n_reference: int,
    reference_per_level: int,
    seed: int,
    out_path: Path,
    resume: bool,
) -> None:
    from mace.calculators import mace_mp

    rng = np.random.default_rng(seed)

    records: list[dict[str, object]] = []
    n_prior_done = 0
    if resume and out_path.exists():
        records = json.loads(out_path.read_text())["arrangements"]
        n_prior_done = len(records)
        print(f"resuming: {n_prior_done} arrangements already in {out_path}")

    calc = mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float64")

    plan = _composition_plan(n_train, train_per_level, n_reference, reference_per_level)
    n_oxygen_sites = None
    t_start = time.time()
    n_total = sum(count for _, count in plan) * len(VACANCY_LEVELS)
    n_done = 0

    for c, per_level in plan:
        spec = build_composition(rng, c)
        if n_oxygen_sites is None:
            n_oxygen_sites = len(spec.oxygen_positions)
        is_reference = c >= n_train

        for v in VACANCY_LEVELS:
            for _ in range(per_level):
                # Always draw, to keep the RNG stream identical to a fresh run;
                # only skip the (expensive) relaxation itself when resuming
                # past a prefix that a prior run already checkpointed.
                cation_species = shuffle_cation_arrangement(rng, spec)
                vacancy_sites = sorted(
                    rng.choice(n_oxygen_sites, size=v, replace=False).tolist()
                )
                n_done += 1
                if n_done <= n_prior_done:
                    continue

                atoms = build_arrangement_atoms(spec, cation_species, vacancy_sites)
                energy = relax_energy_ev(atoms, calc)

                records.append(
                    {
                        "composition": spec.tag,
                        "family": "HEO-spinel-factory-v2",
                        "subset": "reference" if is_reference else "train",
                        "v": v,
                        "cation_species": cation_species,
                        "cation_positions": spec.cation_positions,
                        "oxygen_positions": spec.oxygen_positions,
                        "vacancy_sites": vacancy_sites,
                        "cell": spec.cell,
                        "energy_ev": energy,
                        "source_run": SOURCE_RUN,
                    }
                )
                if n_done % 10 == 0:
                    _write_checkpoint(records, out_path)
                if n_done % 5 == 0:
                    elapsed = time.time() - t_start
                    rate = (n_done - n_prior_done) / elapsed
                    eta_min = (n_total - n_done) / rate / 60.0
                    print(
                        f"[{n_done}/{n_total}] {spec.tag} v={v} "
                        f"E={energy:.4f} eV  ({rate:.2f} relax/s, "
                        f"ETA {eta_min:.1f} min)",
                        flush=True,
                    )

    _write_checkpoint(records, out_path)
    print(f"wrote {len(records)} arrangements -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-train",
        type=int,
        default=16,
        help="Compositions in the breadth (training) pool.",
    )
    parser.add_argument(
        "--train-per-level",
        type=int,
        default=5,
        help="Arrangements per vacancy level for each training composition.",
    )
    parser.add_argument(
        "--n-reference",
        type=int,
        default=2,
        help="Compositions in the depth (brute-force reference) pool.",
    )
    parser.add_argument(
        "--reference-per-level",
        type=int,
        default=60,
        help="Arrangements per vacancy level for each reference composition.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/full/factory_v2.json"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip relaxations already present in --out, replaying the same RNG "
        "stream so the remaining draws match a from-scratch run.",
    )
    args = parser.parse_args()

    generate(
        n_train=args.n_train,
        train_per_level=args.train_per_level,
        n_reference=args.n_reference,
        reference_per_level=args.reference_per_level,
        seed=args.seed,
        out_path=args.out,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
