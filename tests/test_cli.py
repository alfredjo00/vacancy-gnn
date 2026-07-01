"""Smoke tests for the CLI surface."""

from __future__ import annotations

from typer.testing import CliRunner

from vacancy_gnn import __version__
from vacancy_gnn.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_gibbs_zero_temperature_reports_minimum() -> None:
    result = runner.invoke(app, ["gibbs", "-e", "-2.0,-5.0,-1.0", "-t", "0"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "-5.000000"


def test_unimplemented_command_exits_nonzero() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
