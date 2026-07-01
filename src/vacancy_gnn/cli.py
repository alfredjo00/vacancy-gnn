"""Command-line interface.

``train``, ``evaluate``, and ``predict`` all run end-to-end against synthetic
data until the offline factory export lands (PLAN.md step 7); ``gibbs`` only
depends on the pure physics core and needs no dataset.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vacancy_gnn import __version__
from vacancy_gnn.data.synthetic import (
    make_brute_force_reference,
    make_synthetic_dataset,
)
from vacancy_gnn.evaluate import evaluate as run_evaluation
from vacancy_gnn.models import LinearBaseline
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
    regularization: float = typer.Option(
        1e-3, "--reg", help="Ridge penalty for the linear baseline."
    ),
    cutoff: float = typer.Option(5.0, "--cutoff", help="Edge distance cutoff."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    checkpoint_dir: Path | None = typer.Option(
        None, "--checkpoint-dir", help="Directory to save the fitted model."
    ),
) -> None:
    """Train the linear baseline on a synthetic dataset (real end-to-end run).

    Uses synthetic data until the factory export lands (PLAN.md step 4); the GNN
    plugs into the same loop later.
    """
    dataset = make_synthetic_dataset(seed=seed)
    model = LinearBaseline(regularization=regularization)
    result = run_training(
        model,
        dataset,
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
    regularization: float = typer.Option(
        1e-3, "--reg", help="Ridge penalty for the linear baseline."
    ),
    cutoff: float = typer.Option(5.0, "--cutoff", help="Edge distance cutoff."),
    temperature: float = typer.Option(
        T_FR, "--temperature", "-t", help="Reactor temperature in kelvin."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
) -> None:
    """Evaluate a freshly trained baseline against a brute-force reference.

    Trains on synthetic data and scores against
    :func:`vacancy_gnn.data.synthetic.make_brute_force_reference` until the
    factory export lands (PLAN.md step 6/7); reports the harness in PLAN.md
    Section 7 (parity MAE/RMSE and the per-composition min-vs-average
    divergence).
    """
    dataset = make_synthetic_dataset(seed=seed)
    model = LinearBaseline(regularization=regularization)
    run_training(model, dataset, cutoff=cutoff, seed=seed)

    reference = make_brute_force_reference(seed=seed + 1000)
    report = run_evaluation(
        model, reference, reactor_temperature=temperature, cutoff=cutoff, seed=seed
    )

    typer.echo(
        f"parity: MAE={report.parity.mae:.4f} eV  RMSE={report.parity.rmse:.4f} eV"
    )
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
    n_candidates: int = typer.Option(
        50, "--n-candidates", help="Number of candidate arrangements to score."
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

    Trains on synthetic data, then scores ``n_candidates`` freshly sampled
    arrangements at the requested vacancy count and averages the predictions
    (PLAN.md step 6). ``composition`` selects among the brute-force reference
    compositions generated for this run.
    """
    dataset = make_synthetic_dataset(seed=seed)
    model = LinearBaseline(regularization=regularization)
    run_training(model, dataset, cutoff=cutoff, seed=seed)

    reference = make_brute_force_reference(
        vacancy_levels=(vacancies,),
        arrangements_per_level=n_candidates,
        seed=seed + 1000,
    )
    candidates = [
        a
        for a in reference.arrangements
        if a.composition == composition and a.v == vacancies
    ]
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
