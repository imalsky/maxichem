"""What the checks and the benches share: the gases they score against VULCAN, the miss score, and the start gas.

The miss rule is in notes.md ("Miss rule"): a scored point's floored error against VULCAN above MISS is a miss.
"""

from types import ModuleType
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import numpy as np
    import torch

Fractions = TypeVar("Fractions", "np.ndarray", "torch.Tensor")  # importing the package imports neither

GASES = ["H2O", "CO", "CO2", "CH4", "NH3", "HCN", "C2H2", "H", "OH", "O", "H2", "N2"]  # the 12 scored gases
FLOORS = [1e-10] * 7 + [1e-8] * 3 + [1e-6] * 2  # mole fraction: below this a gas is not scored, per GASES
SULFUR_GASES = ["H2S", "SO2", "COS", "S2"]  # also scored on a network with sulfur
SULFUR_FLOOR = 1e-10  # mole fraction
MISS = 0.1  # dex: a scored point whose floored error against VULCAN is above this is a miss
LARGE_MISS = 1.0  # dex: a large miss
TINY = 1e-300  # mole fraction: keeps log10 finite, far below every floor
SOLAR_START = {"H2": 0.84, "He": 0.16, "H2O": 9e-4, "CO": 5e-4, "CH4": 1e-5, "N2": 6e-5, "NH3": 1e-6}  # mole fractions
H2S_PER_H2_SOLAR = 1.41e-5  # the sulfur of a solar gas, for starts made here
# The rate rule the checks' truth sets and stored references must be made with (the collision limit counts M as a
# collision partner, docs/method.md); the truth generator and the torch reference generator write the same string
RATE_RULE = "collision-limit-2026-11-10: M counts as a collision partner"


def scored_gases(elements: list[str]) -> tuple[list[str], list[float]]:
    """Return the gases scored on a network with these elements, and their floors in mole fraction."""
    if "S" in elements:
        return GASES + SULFUR_GASES, FLOORS + [SULFUR_FLOOR] * len(SULFUR_GASES)
    return list(GASES), list(FLOORS)


def floored_log10(x: Fractions, log_floors: Fractions, xp: ModuleType) -> Fractions:
    """Return log10 of mole fractions, raised to each gas's floor; NaN where x is negative or not finite.

    The floored error of a scored point is the absolute difference of two of these. A NaN, from a negative or
    non-finite value on either side, is a failure, never a zero error.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor [..., gases]
        Mole fractions of the scored gases.
    log_floors : numpy.ndarray or torch.Tensor [gases]
        log10 of their floors, of x's kind (and device).
    xp : module
        numpy or torch, the module of x.

    Returns
    -------
    numpy.ndarray or torch.Tensor [..., gases]
        Floored log10, dex.
    """
    floored = xp.maximum(xp.log10(xp.where(x > TINY, x, TINY)), log_floors)
    result: Fractions = xp.where(xp.isfinite(x) & (x >= 0), floored, xp.nan)
    return result
