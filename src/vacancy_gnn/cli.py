"""Command-line interface.

``train``, ``evaluate``, and ``predict`` load a factory export (PLAN.md step 7)
through :func:`vacancy_gnn.data.factory.load_factory_export`, defaulting to the
small committed ``data/sample/factory_sample.json``; pass ``--data`` to point at
``data/full/factory_v2.json`` or another export for a real run. ``gibbs`` only
depends on the pure physics core and needs no dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from numpy.typing import NDArray

from vacancy_gnn import __version__
from vacancy_gnn.data.factory import FactoryExport, load_factory_export
from vacancy_gnn.evaluate import evaluate as run_evaluation
from vacancy_gnn.models import LinearBaseline
from vacancy_gnn.models.reference import prior_from_e0s
from vacancy_gnn.physics import free_energy
from vacancy_gnn.physics.constants import T_FR
from vacancy_gnn.predict import predict_free_energy
from vacancy_gnn.train import train as run_training

app = typer.Typer(
    name="vacancy-gnn",
    help="Learned oxygen-vacancy configurational energies for HEO Gibbs energies.",
    no_args_is_help=True,
    add_completion=False,
)

_DEFAULT_DATA = Path("data/sample/factory_sample.json")

#: Default shrinkage toward the E0 prior: OFF. The measurement gate for E0
#: anchoring failed (IMPROVEMENTS.md P8 addendum: raw isolated-atom energies
#: miss the element-specific cohesive term by -7 to -18 eV/atom, so prior-only
#: predictions are far worse than the plain fit, and no shrinkage value
#: recovers it). The machinery stays available for a future, better prior;
#: with shrinkage 0.0 the prior is ignored and the fit is the plain lstsq.
_DEFAULT_REFERENCE_SHRINKAGE = 0.0


def _load(data: Path) -> FactoryExport:
    """Load a factory export, exiting with a clear message if it is missing."""
    if not data.exists():
        typer.echo(
            f"no factory export at {data}; generate one with "
            "scripts/generate_factory_data.py or point --data at an existing file",
            err=True,
        )
        raise typer.Exit(1)
    return load_factory_export(data)


def _reference_prior(export: FactoryExport) -> NDArray[np.float64] | None:
    """Build the E0 prior for ``export``, or ``None`` if it has no E0s.

    Only consulted when ``_DEFAULT_REFERENCE_SHRINKAGE`` is nonzero; with
    anchoring off (the current default, see that constant's docstring) the
    prior is passed through but ignored by ``CompositionReference.fit``.

    Reads ``n_cations``/``n_oxygen_sites`` off the first training arrangement;
    every arrangement in a factory export shares one fixed cell (see
    :mod:`scripts.generate_factory_data`).
    """
    if not export.e0s_ev:
        return None
    arrangements = export.train.arrangements or export.reference.arrangements
    if not arrangements:
        return None
    sample = arrangements[0]
    return prior_from_e0s(
        export.e0s_ev,
        n_cations=len(sample.cation_species),
        n_oxygen_sites=len(sample.oxygen_positions),
    )


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def gibbs(
    energies: str = typer.Option(
        ...,
        "--energies",
        "-e",
        help="Comma-separated arrangement energies in eV, e.g. '-4.0,-2.0,1.0'.",
    ),
    temperature: float = typer.Option(
        1323.0, "--temperature", "-t", help="Temperature in kelvin."
    ),
) -> None:
    """Boltzmann-averaged configurational free energy G(T) for given energies."""
    values = [float(x) for x in energies.split(",") if x.strip()]
    g = free_energy(values, temperature)
    typer.echo(f"{g:.6f}")


@app.command()
def train(
    data: Path = typer.Option(
        _DEFAULT_DATA, "--data", help="Factory export to train on ('train' subset)."
    ),
    regularization: float = typer.Option(
        1e-3, "--reg", help="Ridge penalty for the linear baseline."
    ),
    cutoff: float = typer.Option(5.0, "--cutoff", help="Edge distance cutoff."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    checkpoint_dir: Path | None = typer.Option(
        None, "--checkpoint-dir", help="Directory to save the fitted model."
    ),
) -> None:
    """Train the linear baseline on a factory export's training split."""
    export = _load(data)
    prior = _reference_prior(export)
    model = LinearBaseline(
        regularization=regularization,
        reference_prior=prior,
        reference_shrinkage=_DEFAULT_REFERENCE_SHRINKAGE if prior is not None else 0.0,
    )
    result = run_training(
        model,
        export.train,
        cutoff=cutoff,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    typer.echo(
        f"val MAE={result.val_mae:.4f} eV  val RMSE={result.val_rmse:.4f} eV  "
        f"(train={result.n_train}, val={result.n_val})"
    )
    if result.checkpoint is not None:
        typer.echo(f"saved checkpoint -> {result.checkpoint}")


@app.command()
def evaluate(
    data: Path = typer.Option(
        _DEFAULT_DATA,
        "--data",
        help="Factory export to train on and evaluate against.",
    ),
    regularization: float = typer.Option(
        1e-3, "--reg", help="Ridge penalty for the linear baseline."
    ),
    cutoff: float = typer.Option(5.0, "--cutoff", help="Edge distance cutoff."),
    temperature: float = typer.Option(
        T_FR, "--temperature", "-t", help="Reactor temperature in kelvin."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
) -> None:
    """Evaluate a freshly trained baseline against the brute-force reference split.

    Trains on the export's ``train`` subset and scores against its ``reference``
    subset (PLAN.md Section 7): parity MAE/RMSE and the per-composition
    min-vs-average divergence.
    """
    export = _load(data)
    prior = _reference_prior(export)
    model = LinearBaseline(
        regularization=regularization,
        reference_prior=prior,
        reference_shrinkage=_DEFAULT_REFERENCE_SHRINKAGE if prior is not None else 0.0,
    )
    run_training(model, export.train, cutoff=cutoff, seed=seed)

    report = run_evaluation(
        model,
        export.reference,
        reactor_temperature=temperature,
        cutoff=cutoff,
        seed=seed,
        train=export.train,
    )

    typer.echo(
        f"parity: MAE={report.parity.mae:.4f} eV  RMSE={report.parity.rmse:.4f} eV"
    )
    for warning in report.out_of_hull_warnings:
        typer.echo(f"warning: {warning}", err=True)
    for est in report.free_energy_accuracy:
        typer.echo(
            f"{est.composition} v={est.v}: truth={est.truth:.4f} eV  "
            f"model={est.model_estimate:.4f} eV  error={est.error:+.4f} eV"
        )
    for div in report.min_vs_average:
        label = "entropy-dominated" if div.entropy_dominated else "min-dominated"
        typer.echo(
            f"{div.composition} v={div.v}: G(T->0)={div.g_zero_t:.4f} eV  "
            f"G(T_reactor)={div.g_reactor_t:.4f} eV  ({label})"
        )


@app.command()
def predict(
    composition: str = typer.Option(..., "--composition", help="Composition tag."),
    vacancies: int = typer.Option(..., "--vacancies", "-v", help="Vacancy count."),
    data: Path = typer.Option(
        _DEFAULT_DATA,
        "--data",
        help="Factory export to train on; candidates come from its 'reference' subset.",
    ),
    n_candidates: int = typer.Option(
        50, "--n-candidates", help="Max candidate arrangements to score."
    ),
    regularization: float = typer.Option(
        1e-3, "--reg", help="Ridge penalty for the linear baseline."
    ),
    cutoff: float = typer.Option(5.0, "--cutoff", help="Edge distance cutoff."),
    temperature: float = typer.Option(
        T_FR, "--temperature", "-t", help="Reactor temperature in kelvin."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
) -> None:
    """Predict the Boltzmann-averaged G(v) for a composition and vacancy count.

    Trains on the export's ``train`` subset, then scores up to ``n_candidates``
    arrangements at the requested composition and vacancy count drawn from its
    ``reference`` subset, and Boltzmann-averages the predictions (PLAN.md
    Section 6).
    """
    export = _load(data)
    prior = _reference_prior(export)
    model = LinearBaseline(
        regularization=regularization,
        reference_prior=prior,
        reference_shrinkage=_DEFAULT_REFERENCE_SHRINKAGE if prior is not None else 0.0,
    )
    run_training(model, export.train, cutoff=cutoff, seed=seed)

    reference = export.reference
    candidates = [
        a
        for a in reference.arrangements
        if a.composition == composition and a.v == vacancies
    ][:n_candidates]
    if not candidates:
        available = sorted(reference.compositions())
        typer.echo(
            f"no candidates for composition={composition!r}, v={vacancies}; "
            f"available compositions: {available}",
            err=True,
        )
        raise typer.Exit(1)

    g = predict_free_energy(model, candidates, temperature=temperature, cutoff=cutoff)
    typer.echo(f"{g:.6f}")


if __name__ == "__main__":
    app()
