"""Command-line interface.

The heavy subcommands (``train``, ``evaluate``, ``predict``) are stubbed here so
the CLI surface exists and is smoke-tested from day one; they are filled in as the
corresponding modules land (PLAN.md Section 10). ``gibbs`` is already functional
because it only depends on the pure physics core.
"""

from __future__ import annotations

import typer

from vacancy_gnn import __version__
from vacancy_gnn.physics import free_energy

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
def train() -> None:
    """Train a model (stub; implemented in PLAN.md step 4/5)."""
    raise typer.Exit(_not_yet("train"))


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
