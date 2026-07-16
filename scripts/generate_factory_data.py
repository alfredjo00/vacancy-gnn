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

Scope: a 3x2x1 supercell of the MgAl2O4 spinel prototype (Materials Project
mp-3536, supplied as an ASE database via --prototype-db), 84 atoms (12 A-site,
24 B-site, 48 O). Each composition fixes a random stoichiometry of 4-9 unique
cation elements (drawn from the A-site/B-site element pools below); each
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

To grow an existing dataset instead of regenerating it, use --extend: it keeps
every arrangement, E0, and the reference split from the input export and only
relaxes new training compositions, so no prior GPU work is repeated. Append 14
more training compositions (v2 -> v3), or double the training pool:

    python scripts/generate_factory_data.py \\
        --extend data/full/factory_v2.json --n-add 14 --out data/full/factory_v3.json
    python scripts/generate_factory_data.py \\
        --extend data/full/factory_v2.json --out data/full/factory_v3.json  # doubles

Writes partial results to ``out_path`` every 10 relaxations, so a crash or
interrupt loses at most that much GPU work; rerun with --resume to continue
(for --extend, --resume continues a partially finished extend in --out).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: ASE database holding the MgAl2O4 spinel prototype (Materials Project
#: mp-3536). Not shipped with this repo; pass one with --prototype-db.
DEFAULT_PROTOTYPE_DB = Path("prototypes.db")
SUPERCELL_DIMENSIONS: tuple[int, int, int] = (3, 2, 1)
VACANCY_LEVELS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
#: Number of unique cation elements (summed over both sites) per composition,
#: inclusive. Bounds a realistic "high entropy" arity; without this cap,
#: independently drawing each site from its full element pool nearly always
#: yields 14-16 unique elements, which is not a plausible HEO composition.
MIN_UNIQUE_ELEMENTS = 4
MAX_UNIQUE_ELEMENTS = 9

# Real species only: this script assigns species directly and does not use
# icet/SQS, so no proxy elements are needed.
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


def build_composition(
    rng: np.random.Generator, index: int, prototype_db: Path
) -> CompositionSpec:
    """Fix one random stoichiometry on the 3x2x1 spinel supercell's cation sites.

    Restricted to MIN_UNIQUE_ELEMENTS-MAX_UNIQUE_ELEMENTS unique cation
    elements total, matching realistic high-entropy-oxide arity (see
    _pick_element_subset). Returns fixed per-element counts; the site
    occupancy itself is shuffled per arrangement by
    :func:`shuffle_cation_arrangement`.
    """
    import ase.db
    from ase.data import atomic_numbers

    db = ase.db.connect(str(prototype_db))
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

#: XORed into the seed by :func:`extend` so appended compositions come from a
#: different RNG stream than the base run, and can never accidentally reproduce
#: a composition the base run already drew.
_EXTEND_SEED_SALT = 0x5EEDE


def relax_energy_ev(atoms: object, calc: object) -> float:
    """Relax ``atoms`` with MACE-MPA-0 (BFGS, cell + ions) and return final energy."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    atoms.calc = calc  # type: ignore[attr-defined]
    ecf = FrechetCellFilter(atoms)
    opt = BFGS(ecf, logfile=None)
    opt.run(fmax=RELAX_FMAX, steps=1000)
    return float(atoms.get_potential_energy())  # type: ignore[attr-defined]


#: Side length (A) of the empty box an isolated atom is placed in for
#: :func:`compute_e0s_ev`. Large enough that periodic images don't interact.
ISOLATED_ATOM_BOX_A = 20.0


def compute_e0s_ev(elements: list[str], calc: object) -> dict[str, float]:
    """Single-atom reference energies (eV) for ``elements`` plus oxygen.

    These are the trivial per-element offsets that
    :func:`vacancy_gnn.models.reference.prior_from_e0s` anchors the
    composition reference to (IMPROVEMENTS.md P8): a physically motivated
    center for directions the training compositions don't constrain, instead
    of an arbitrary minimum-norm value. One single-point energy per element in
    a large empty box; seconds of compute, no relaxation, no GPU needed.

    Args:
        elements: Element symbols to compute (oxygen is always included; pass
            the union of the run's A-site/B-site pools).
        calc: An ASE calculator (e.g. from ``mace_mp``).

    Returns:
        Mapping from element symbol to isolated-atom energy (eV).
    """
    from ase import Atoms

    symbols = sorted({*elements, "O"})
    e0s: dict[str, float] = {}
    for symbol in symbols:
        atoms = Atoms(
            symbol, positions=[[0.0, 0.0, 0.0]], cell=[ISOLATED_ATOM_BOX_A] * 3
        )
        atoms.calc = calc  # type: ignore[attr-defined]
        e0s[symbol] = float(atoms.get_potential_energy())  # type: ignore[attr-defined]
    return e0s


def _write_checkpoint(
    records: list[dict[str, object]], e0s_ev: dict[str, float], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps({"arrangements": records, "e0s_ev": e0s_ev}))
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


#: Element pool for the composition-reference coverage check, ordered like
#: vacancy_gnn.data.descriptors.DESCRIPTOR_SPECIES (cations only; the vacancy
#: marker isn't a composition-plan concern since v varies within every
#: composition and is always well constrained by data).
_PREFLIGHT_ELEMENTS: tuple[str, ...] = (
    "Li",
    "Mg",
    "Al",
    "Ca",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Zr",
    "In",
    "Sn",
)


def _cation_count_vector(cation_species: list[int]) -> np.ndarray:
    """Per-element counts over :data:`_PREFLIGHT_ELEMENTS`, from atomic numbers."""
    from ase.data import atomic_numbers

    order = [atomic_numbers[e] for e in _PREFLIGHT_ELEMENTS]
    idx = {z: i for i, z in enumerate(order)}
    counts = np.zeros(len(_PREFLIGHT_ELEMENTS), dtype=np.float64)
    for z in cation_species:
        counts[idx[z]] += 1.0
    return counts


def _out_of_span_norm(train_counts: np.ndarray, counts: np.ndarray) -> float:
    """Norm of ``counts`` orthogonal to the row space spanned by ``train_counts``.

    Same operation as vacancy_gnn.models.reference.out_of_span_norm, reimplemented
    here directly on count vectors (no Graph needed) so this pre-flight check has
    no dependency on the installed package or a relaxed structure.
    """
    _, s, vt = np.linalg.svd(train_counts, full_matrices=False)
    basis = vt[s > 1e-8 * s[0]] if s.size else vt[:0]
    projection = basis.T @ (basis @ counts)
    return float(np.linalg.norm(counts - projection))


def _replay_plan_specs(
    n_train: int,
    train_per_level: int,
    n_reference: int,
    reference_per_level: int,
    seed: int,
    prototype_db: Path,
) -> list[CompositionSpec]:
    """The exact composition specs a :func:`generate` run with these args builds.

    :func:`generate` interleaves per-arrangement RNG draws (cation shuffles,
    vacancy sets) between composition builds, so the spec for composition ``c``
    depends on how many arrangements every earlier composition consumed. This
    replays that stream faithfully (burning the arrangement draws without
    relaxing anything), so pre-flight sees the same compositions the real run
    would.
    """
    rng = np.random.default_rng(seed)
    plan = _composition_plan(n_train, train_per_level, n_reference, reference_per_level)
    specs: list[CompositionSpec] = []
    for c, per_level in plan:
        spec = build_composition(rng, c, prototype_db)
        specs.append(spec)
        n_oxygen_sites = len(spec.oxygen_positions)
        for v in VACANCY_LEVELS:
            for _ in range(per_level):
                shuffle_cation_arrangement(rng, spec)
                rng.choice(n_oxygen_sites, size=v, replace=False)
    return specs


def preflight(
    n_train: int,
    train_per_level: int,
    n_reference: int,
    reference_per_level: int,
    seed: int,
    prototype_db: Path,
) -> None:
    """Check composition-reference coverage before spending any GPU time.

    Builds only the composition plan (no relaxation, no GPU) for the exact run
    these arguments describe (see :func:`_replay_plan_specs`), then reports:

    - The rank of the training cation-count design matrix (must be
      ``len(_PREFLIGHT_ELEMENTS)`` for the composition reference to even be
      identifiable; see IMPROVEMENTS.md P8).
    - Each element's count range across the training compositions.
    - Each reference composition's out-of-span norm against the training
      compositions (see vacancy_gnn.models.reference.out_of_span_norm) and its
      statistical leverage ``x^T (X^T X)^+ x``. The norm catches the
      rank-deficient failure (nonzero = the reference composition needs a
      direction training never constrains); once the design is full rank the
      norm is identically zero and leverage is the sharper flag (>~1 means the
      fit must extrapolate toward a hull corner, the -168 eV swap-test regime
      from IMPROVEMENTS.md P8's addendum).

    Both flags are leading indicators of the exact P8 failure, cheap enough to
    check before burning GPU-hours on the full factory.
    """
    specs = _replay_plan_specs(
        n_train, train_per_level, n_reference, reference_per_level, seed, prototype_db
    )
    train_counts = np.stack(
        [_cation_count_vector(s.cation_counts) for s in specs[:n_train]]
    )
    rank = np.linalg.matrix_rank(train_counts)
    n_species = len(_PREFLIGHT_ELEMENTS)

    print(f"{n_train} training compositions, {n_species} cation species")
    print(f"design matrix rank: {rank} of {n_species}", end="")
    print(" (full rank)" if rank == n_species else " (RANK-DEFICIENT)")

    print("\nper-element count range across training compositions:")
    for i, sym in enumerate(_PREFLIGHT_ELEMENTS):
        col = train_counts[:, i]
        present = int((col > 0).sum())
        print(
            f"  {sym:3s} range=[{col.min():.0f},{col.max():.0f}]  "
            f"in {present}/{n_train} compositions"
        )

    xtx_pinv = np.linalg.pinv(train_counts.T @ train_counts)
    print("\nreference compositions vs training hull:")
    any_flagged = False
    for spec in specs[n_train:]:
        counts = _cation_count_vector(spec.cation_counts)
        norm = _out_of_span_norm(train_counts, counts)
        leverage = float(counts @ xtx_pinv @ counts)
        flag = ""
        if norm > 1.0 or leverage > 1.5:
            flag = "  <-- likely large composition-reference error (IMPROVEMENTS.md P8)"
            any_flagged = True
        print(f"  {spec.tag:50s} oos-norm={norm:.3f}  leverage={leverage:.2f}{flag}")

    if rank < n_species or any_flagged:
        print(
            "\nPRE-FLIGHT WARNING: composition-reference coverage looks weak for "
            "this plan; consider more/broader training compositions before "
            "running the full (GPU) factory."
        )
    else:
        print("\npre-flight OK: full rank, no reference composition flagged.")


def _relax_plan(
    plan: list[tuple[int, int, str]],
    rng: np.random.Generator,
    calc: object,
    e0s_ev: dict[str, float],
    records: list[dict[str, object]],
    out_path: Path,
    *,
    n_prior_done: int,
    source_run: str,
    prototype_db: Path,
) -> None:
    """Relax every arrangement in ``plan``, appending records and checkpointing.

    ``plan`` is a list of ``(composition_index, per_level, subset)`` tuples;
    each entry builds one composition and relaxes ``per_level`` arrangements at
    every vacancy level. Arrangement RNG draws always happen (so the stream is
    identical to a fresh run), but the first ``n_prior_done`` arrangements *that
    this plan produces* are skipped, which is what lets ``--resume`` continue an
    interrupted run.

    ``n_prior_done`` is plan-relative: it counts only arrangements this plan
    generates, not any records already present in ``records`` from another
    source. ``extend`` seeds ``records`` with a preserved base and passes a
    ``n_prior_done`` of 0 (fresh) or the count of new arrangements already
    finished (resume), so the base is appended-around, never skipped-over.

    Mutates ``records`` in place and writes ``out_path`` every 10 relaxations.
    """
    n_oxygen_sites = None
    t_start = time.time()
    n_total = sum(per_level for _, per_level, _ in plan) * len(VACANCY_LEVELS)
    n_done = 0

    for c, per_level, subset in plan:
        spec = build_composition(rng, c, prototype_db)
        if n_oxygen_sites is None:
            n_oxygen_sites = len(spec.oxygen_positions)

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
                        "subset": subset,
                        "v": v,
                        "cation_species": cation_species,
                        "cation_positions": spec.cation_positions,
                        "oxygen_positions": spec.oxygen_positions,
                        "vacancy_sites": vacancy_sites,
                        "cell": spec.cell,
                        "energy_ev": energy,
                        "source_run": source_run,
                    }
                )
                if n_done % 10 == 0:
                    _write_checkpoint(records, e0s_ev, out_path)
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

    _write_checkpoint(records, e0s_ev, out_path)


def _load_calc_and_e0s(e0s_ev: dict[str, float]) -> tuple[object, dict[str, float]]:
    """Load the MACE calculator and fill in E0s if not already known."""
    from mace.calculators import mace_mp

    calc = mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float64")
    if not e0s_ev:
        elements = sorted(set(A_SITE_ELEMENTS) | set(B_SITE_ELEMENTS))
        e0s_ev = compute_e0s_ev(elements, calc)
        print(f"computed E0s for {len(e0s_ev)} elements (incl. O)")
    return calc, e0s_ev


def generate(
    n_train: int,
    train_per_level: int,
    n_reference: int,
    reference_per_level: int,
    seed: int,
    out_path: Path,
    resume: bool,
    prototype_db: Path,
) -> None:
    rng = np.random.default_rng(seed)

    records: list[dict[str, object]] = []
    e0s_ev: dict[str, float] = {}
    n_prior_done = 0
    if resume and out_path.exists():
        prior = json.loads(out_path.read_text())
        records = prior["arrangements"]
        e0s_ev = prior.get("e0s_ev", {})
        n_prior_done = len(records)
        print(f"resuming: {n_prior_done} arrangements already in {out_path}")

    calc, e0s_ev = _load_calc_and_e0s(e0s_ev)

    plan = _composition_plan(n_train, train_per_level, n_reference, reference_per_level)
    subset_plan = [
        (c, per_level, "reference" if c >= n_train else "train")
        for c, per_level in plan
    ]
    _relax_plan(
        subset_plan,
        rng,
        calc,
        e0s_ev,
        records,
        out_path,
        n_prior_done=n_prior_done,
        source_run=SOURCE_RUN,
        prototype_db=prototype_db,
    )
    print(f"wrote {len(records)} arrangements -> {out_path}")


def _existing_composition_indices(records: list[dict[str, object]]) -> set[int]:
    """Composition indices already present, parsed from ``-factory-NNN`` tags."""
    indices: set[int] = set()
    for r in records:
        tag = str(r["composition"])
        suffix = tag.rsplit("-factory-", 1)
        if len(suffix) == 2 and suffix[1].isdigit():
            indices.add(int(suffix[1]))
    return indices


def extend(
    in_path: Path,
    out_path: Path,
    n_add: int,
    train_per_level: int,
    seed: int,
    resume: bool,
    prototype_db: Path,
) -> None:
    """Append ``n_add`` new training compositions to an existing export.

    Everything already in ``in_path`` (arrangements, E0s, the reference split)
    is kept verbatim; only new ``subset="train"`` compositions are relaxed and
    appended, so no prior GPU work is repeated. New compositions get fresh
    ``-factory-NNN`` indices continuing past the highest one already present, so
    their tags never collide with existing records.

    A distinct RNG seed (``seed`` XOR a fixed salt) draws the new compositions,
    so ``extend`` does not have to replay the original run's stream and cannot
    accidentally reproduce an existing composition.

    With ``--resume``, ``out_path`` is treated as a checkpoint of a partially
    finished extend run and continued; otherwise ``in_path`` is the base and
    ``out_path`` is written fresh (they may be the same file).
    """
    base = json.loads(in_path.read_text())
    base_records: list[dict[str, object]] = base["arrangements"]
    e0s_ev: dict[str, float] = base.get("e0s_ev", {})
    base_count = len(base_records)

    # ``records`` accumulates base + newly relaxed arrangements. On a fresh
    # extend it starts as the base (which is preserved by appending, never
    # touched by the relax loop). On --resume it starts from the checkpoint,
    # which already contains base + whatever new work finished before the
    # interrupt.
    records = list(base_records)
    # n_plan_done counts only NEW arrangements the plan has produced (the
    # skip offset _relax_plan applies), NOT the preserved base records.
    n_plan_done = 0
    if resume and out_path.exists() and out_path != in_path:
        prior = json.loads(out_path.read_text())
        records = prior["arrangements"]
        e0s_ev = prior.get("e0s_ev", e0s_ev)
        n_plan_done = len(records) - base_count
        print(
            f"resuming extend: {len(records)} arrangements in {out_path} "
            f"({n_plan_done} new already done)"
        )
    else:
        print(f"extending {in_path} ({base_count} arrangements) -> {out_path}")

    # Number new compositions past the highest index in the BASE, not in
    # ``records``: on --resume ``records`` already holds partially-added new
    # comps, so deriving the start from it would renumber them and desync the
    # plan from the RNG stream. The base fixes the numbering the same way for
    # a fresh run and every resume of it.
    start_index = max(_existing_composition_indices(base_records), default=-1) + 1
    print(f"adding {n_add} new training compositions from index {start_index}")

    # A distinct stream from the base run: salt the seed so new compositions are
    # genuinely new draws, never a replay of the original stream.
    rng = np.random.default_rng(seed ^ _EXTEND_SEED_SALT)
    calc, e0s_ev = _load_calc_and_e0s(e0s_ev)

    subset_plan = [(start_index + i, train_per_level, "train") for i in range(n_add)]
    _relax_plan(
        subset_plan,
        rng,
        calc,
        e0s_ev,
        records,
        out_path,
        n_prior_done=n_plan_done,
        source_run=f"{SOURCE_RUN}-extend",
        prototype_db=prototype_db,
    )
    added = len(records) - base_count
    print(f"wrote {len(records)} arrangements ({added} new) -> {out_path}")


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
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check composition-reference coverage (rank, per-element ranges, "
        "out-of-span norms) for this plan and exit, without running any "
        "relaxations or needing a GPU (IMPROVEMENTS.md P8/P1).",
    )
    parser.add_argument(
        "--extend",
        type=Path,
        default=None,
        metavar="EXPORT",
        help="Append new training compositions to an existing export instead of "
        "generating from scratch. Reuses all of EXPORT's arrangements, E0s, and "
        "reference split; only --n-add new training compositions are relaxed. "
        "Writes to --out (may equal EXPORT). See --n-add.",
    )
    parser.add_argument(
        "--n-add",
        type=int,
        default=None,
        help="With --extend: number of new training compositions to append. "
        "Defaults to the number of training compositions already in the export "
        "(i.e. doubling the training pool).",
    )
    parser.add_argument(
        "--prototype-db",
        type=Path,
        default=DEFAULT_PROTOTYPE_DB,
        help="ASE database holding the MgAl2O4 spinel prototype (Materials "
        "Project mp-3536). Not shipped with this repo.",
    )
    args = parser.parse_args()

    if not args.prototype_db.exists():
        parser.error(
            f"prototype database not found: {args.prototype_db}\n"
            "Pass one with --prototype-db (Materials Project mp-3536, as an "
            "ASE .db)."
        )

    if args.preflight:
        preflight(
            n_train=args.n_train,
            train_per_level=args.train_per_level,
            n_reference=args.n_reference,
            reference_per_level=args.reference_per_level,
            seed=args.seed,
            prototype_db=args.prototype_db,
        )
        return

    if args.extend is not None:
        n_add = args.n_add
        if n_add is None:
            base = json.loads(args.extend.read_text())
            n_train_existing = len(
                {
                    r["composition"]
                    for r in base["arrangements"]
                    if r["subset"] == "train"
                }
            )
            n_add = n_train_existing
            print(f"--n-add not given; doubling the {n_train_existing} training comps")
        extend(
            in_path=args.extend,
            out_path=args.out,
            n_add=n_add,
            train_per_level=args.train_per_level,
            seed=args.seed,
            resume=args.resume,
            prototype_db=args.prototype_db,
        )
        return

    generate(
        n_train=args.n_train,
        train_per_level=args.train_per_level,
        n_reference=args.n_reference,
        reference_per_level=args.reference_per_level,
        seed=args.seed,
        out_path=args.out,
        resume=args.resume,
        prototype_db=args.prototype_db,
    )


if __name__ == "__main__":
    main()
