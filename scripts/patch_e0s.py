#!/usr/bin/env python3
"""Patch isolated-atom reference energies (E0s) into an existing factory export.

``generate_factory_data.py`` now computes and writes ``e0s_ev`` alongside every
run, but exports written before that (e.g. the original ``factory_v2.json``)
don't have it. This script computes E0s with the same MACE-MPA-0 calculator and
inserts the key into an existing export in place, so
:func:`vacancy_gnn.models.reference.prior_from_e0s` has something to anchor to
(IMPROVEMENTS.md P8) without redoing any relaxations.

Isolated single-atom energies, no relaxation: seconds of CPU compute, no GPU
needed, unlike the factory run itself.

Not part of the installed package: needs ase and mace-torch, same as
generate_factory_data.py. Install them in a throwaway environment to run this.

Usage:
    python scripts/patch_e0s.py data/full/factory_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_factory_data import A_SITE_ELEMENTS, B_SITE_ELEMENTS, compute_e0s_ev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="Factory export JSON to patch.")
    parser.add_argument(
        "--device", default="cpu", help="ASE calculator device (cpu is fine here)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite e0s_ev even if the export already has it.",
    )
    args = parser.parse_args()

    payload = json.loads(args.export.read_text())
    if payload.get("e0s_ev") and not args.force:
        print(f"{args.export} already has e0s_ev; pass --force to recompute")
        return

    from mace.calculators import mace_mp

    calc = mace_mp(model="medium-mpa-0", device=args.device, default_dtype="float64")
    elements = sorted(set(A_SITE_ELEMENTS) | set(B_SITE_ELEMENTS))
    e0s_ev = compute_e0s_ev(elements, calc)

    payload["e0s_ev"] = e0s_ev
    args.export.write_text(json.dumps(payload))
    print(f"wrote e0s_ev for {len(e0s_ev)} elements -> {args.export}")


if __name__ == "__main__":
    main()
