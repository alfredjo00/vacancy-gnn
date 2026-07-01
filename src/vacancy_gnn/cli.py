"""Command-line interface.

The heavy subcommands (``train``, ``evaluate``, ``predict``) are stubbed here so
the CLI surface exists and is smoke-tested from day one; they are filled in as the
corresponding modules land (PLAN.md Section 10). ``gibbs`` is already functional
because it only depends on the pure physics core.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vacancy_gnn import __version__
from vacancy_gnn.data.synthetic import make_synthetic_dataset
from vacancy_gnn.models import LinearBaseline
from vacancy_gnn.physics import free_energy
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
def evaluate() -> None:
    """Evaluate against the brute-force reference (stub; PLAN.md step 6)."""
    raise typer.Exit(_not_yet("evaluate"))


@app.command()
def predict() -> None:
    """Predict averaged G(v) for a composition (stub; PLAN.md step 6)."""
    raise typer.Exit(_not_yet("predict"))


def _not_yet(name: str) -> int:
    typer.echo(f"'{name}' is not implemented yet; see PLAN.md build order.", err=True)
    return 1


if __name__ == "__main__":
    app()
