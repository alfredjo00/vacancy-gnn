"""Pure thermodynamics: Boltzmann-averaged configurational free energy."""

from __future__ import annotations

from vacancy_gnn.physics.boltzmann import (
    boltzmann_weights,
    free_energy,
    free_energy_sweep,
)

__all__ = ["boltzmann_weights", "free_energy", "free_energy_sweep"]
