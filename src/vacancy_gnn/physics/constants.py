"""Physical constants and reactor conditions.

Units are eV and kelvin throughout, matching the CHGNet energy labels.
"""

from __future__ import annotations

from typing import Final

#: Boltzmann constant in eV/K (CODATA 2018).
K_B_EV: Final[float] = 8.617333262e-5

#: Air reactor temperature (K).
T_AR: Final[float] = 1223.0

#: Fuel reactor temperature (K).
T_FR: Final[float] = 1323.0

#: Air reactor oxygen partial pressure (atm).
P_O2_AR: Final[float] = 0.2

#: Fuel reactor oxygen partial pressure (atm).
P_O2_FR: Final[float] = 1e-14
