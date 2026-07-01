# vacancy-gnn

[![CI](https://github.com/alfredjo00/vacancy-gnn/actions/workflows/ci.yml/badge.svg)](https://github.com/alfredjo00/vacancy-gnn/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Learned oxygen-vacancy configurational energies for **low-noise Gibbs free
energies** in high-entropy oxides.

## The problem

To predict the Gibbs free energy `G(v)` of a high-entropy oxide at oxygen-vacancy
level `v`, the standard approach relaxes one special quasi-random structure (SQS)
per level with a machine-learning interatomic potential. But at nonzero `v` there
are `C(48, v)` distinct places to put the vacancies, and their energies differ by
a lot: a vacancy next to a reducible cation (Cu, Mn, Fe) costs very differently
than one buried in an Al/Ti pocket. A single SQS samples once from that wide
distribution, so `G(v)` inherits **0.2-0.8 eV/level of noise** — worst exactly
where it matters (high `v`, reducing conditions).

The physically correct quantity is not one draw, nor the single lowest arrangement,
but the **Boltzmann-weighted configurational average**:

```
G(v; T) = -k_B T ln  sum_i exp(-E_i / k_B T)
```

This package learns per-arrangement vacancy energies with an equivariant GNN and
computes that average cheaply, turning a noisy single-SQS estimate into a smooth,
converged `G(v)`.

> Why the average and not just the lowest-energy arrangement? The minimum is the
> `T -> 0` limit of the expression above; at reactor temperatures (1223-1323 K) the
> configurational entropy is not negligible and taking the minimum biases rankings.
> See [`PLAN.md`](PLAN.md) Section 2.1.

![Noisy single-SQS G(v) vs. the learned Boltzmann average](notebooks/money_figure.png)

Single-SQS draws (red) scatter widely around the brute-force truth (dashed); the
learned Boltzmann average (blue) tightens as more arrangements are scored. See
[`notebooks/01_money_figure.ipynb`](notebooks/01_money_figure.ipynb) for the full,
reproducible walkthrough, including why the gap to truth reflects the current
synthetic-data baseline's fit quality, not the averaging method.

## Status

In progress. Implemented and tested: the pure thermodynamics core
(`vacancy_gnn.physics`), the data layer (schema, featurization, composition-aware
splits), the cluster-expansion linear baseline, a model-agnostic training loop with
experiment tracking and checkpointing, a dependency-light E(3)-equivariant GNN (the
centerpiece), and the evaluation harness (brute-force reference, parity/convergence
metrics, oracle-efficiency, min-vs-average divergence, T-sweep, and the money
figure). The offline CHGNet factory export and full-scale retraining land next per
the build order in [`PLAN.md`](PLAN.md).

The GNN needs the optional `ml` extra (`pip install -e ".[ml]"`, which pulls in
torch); the rest of the package installs and runs without it, so CI stays green
both with and without torch. Reproducing the money figure notebook needs the
`viz` extra (`pip install -e ".[viz]"`, matplotlib) plus Jupyter.

## Quickstart

```bash
pip install -e ".[dev]"
pytest

# Boltzmann-averaged free energy over a few arrangement energies (eV):
vacancy-gnn gibbs -e -4.0,-2.0,1.0 --temperature 1323
# At T -> 0 it reduces to the lowest arrangement:
vacancy-gnn gibbs -e -4.0,-2.0,1.0 --temperature 0

# Train the linear baseline end-to-end (synthetic data until the factory export):
vacancy-gnn train --checkpoint-dir checkpoints

# Evaluate against the brute-force reference (parity, min-vs-average divergence):
vacancy-gnn evaluate

# Predict the averaged G(v) for a composition and vacancy count:
vacancy-gnn predict --composition FeMnAl-ref-000 --vacancies 2
```

## Development

```bash
make check   # ruff + mypy + pytest
```

## License

MIT
