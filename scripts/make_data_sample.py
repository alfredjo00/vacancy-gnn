#!/usr/bin/env python3
"""Carve a small committed sample from a full factory export.

Takes the gitignored ``data/full/factory_v2.json`` (or any factory export in the
same shape) and writes a much smaller ``data/sample/factory_sample.json`` with a
couple of train compositions and one reference composition, all at reduced depth.
The sample is small enough to commit and fast enough for tests, CI, and the
quickstart (PLAN.md Section 5), while still round-tripping through
:func:`vacancy_gnn.data.factory.load_factory_export` exactly like the full file.

This only needs vacancy_gnn itself (already an editable install); no MACE/ase.

Usage:
    python scripts/make_data_sample.py \\
        --source data/full/factory_v2.json --out data/sample/factory_sample.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vacancy_gnn.data.factory import load_factory_export


def make_sample(
    source: Path,
    out_path: Path,
    n_train_compositions: int,
    train_per_level: int,
    reference_per_level: int,
) -> None:
    train, reference = load_factory_export(source)

    train_comps = train.compositions()[:n_train_compositions]
    records: list[dict[str, object]] = []
    for comp in train_comps:
        by_v: dict[int, int] = {}
        for a in train.arrangements:
            if a.composition != comp:
                continue
            if by_v.get(a.v, 0) >= train_per_level:
                continue
            by_v[a.v] = by_v.get(a.v, 0) + 1
            records.append({**a.model_dump(), "subset": "train"})

    ref_comp = reference.compositions()[0]
    by_v = {}
    for a in reference.arrangements:
        if a.composition != ref_comp:
            continue
        if by_v.get(a.v, 0) >= reference_per_level:
            continue
        by_v[a.v] = by_v.get(a.v, 0) + 1
        records.append({**a.model_dump(), "subset": "reference"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"arrangements": records}, indent=None))
    print(f"wrote {len(records)} arrangements -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/full/factory_v2.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/sample/factory_sample.json")
    )
    parser.add_argument("--n-train-compositions", type=int, default=3)
    parser.add_argument("--train-per-level", type=int, default=2)
    parser.add_argument("--reference-per-level", type=int, default=5)
    args = parser.parse_args()

    make_sample(
        source=args.source,
        out_path=args.out,
        n_train_compositions=args.n_train_compositions,
        train_per_level=args.train_per_level,
        reference_per_level=args.reference_per_level,
    )


if __name__ == "__main__":
    main()
