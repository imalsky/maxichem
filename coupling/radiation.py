"""Radiative heating of a column from two-stream fluxes, its budget check, the level temperatures, the schedule.

A two-stream solver gives fluxes per unit area of a plane-parallel column, so the heating is their divergence along
the column, -d(F_up - F_down)/dz, also on a sphere: weighting the fluxes by face areas adds a false 2 F / r per
volume, which grows as 1 / rho per mass and dominates near the lid. docs/architecture.md places this module.
"""

import math
from fractions import Fraction

import torch

__all__ = [
    "RadiationBudgetError",
    "RadiationClock",
    "check_budget",
    "column_heating",
    "column_residual",
    "level_temperatures",
]


class RadiationBudgetError(ArithmeticError):
    """A column's radiative fluxes or heating are not finite, or its heating does not close its energy budget."""


def column_heating(net_up: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    """Return the radiative heating of each layer from the net upward flux on its faces.

    Parameters
    ----------
    net_up : torch.Tensor [ncol, nl + 1]
        Net upward flux F_up - F_down on the layer faces, bottom face first, W m^-2.
    dz : torch.Tensor [ncol, nl] or [nl]
        Layer thicknesses, m.

    Returns
    -------
    torch.Tensor [ncol, nl]
        Heating per volume, W m^-3.
    """
    if net_up.dim() != 2 or net_up.shape[1] < 2:
        raise ValueError(f"net_up must be [ncol, nl + 1] with nl >= 1, got shape {tuple(net_up.shape)}")
    if dz.shape[-1] != net_up.shape[1] - 1:
        raise ValueError(f"dz has {dz.shape[-1]} layers, net_up has faces for {net_up.shape[1] - 1}")
    if not bool((dz > 0).all()):
        raise ValueError("dz must be positive everywhere")
    return -(net_up[:, 1:] - net_up[:, :-1]) / dz


def column_residual(heat: torch.Tensor, dz: torch.Tensor, net_up: torch.Tensor) -> torch.Tensor:
    """Return each column's energy budget error: the heating summed over the column minus the net flux it takes in.

    With ``column_heating`` it is round-off; a heating that is not the flux divergence shows up here.

    Parameters
    ----------
    heat : torch.Tensor [ncol, nl]
        Heating per volume, W m^-3.
    dz : torch.Tensor [ncol, nl] or [nl]
        Layer thicknesses, m.
    net_up : torch.Tensor [ncol, nl + 1]
        Net upward flux on the layer faces, bottom face first, W m^-2.

    Returns
    -------
    torch.Tensor [ncol]
        sum(heat * dz) - (net_up[bottom] - net_up[top]), W m^-2.
    """
    return (heat * dz).sum(-1) - (net_up[:, 0] - net_up[:, -1])


def check_budget(heat: torch.Tensor, dz: torch.Tensor, net_up: torch.Tensor) -> float:
    """Refuse non-finite fluxes or heating, and a column budget beyond round-off; return the worst residual.

    The residual telescopes: each layer's heat * dz is its flux difference with a few roundings, and the sum over nl
    layers adds at most nl more, each of at most 2 max|net_up| (twice the largest flux). So round-off stays below
    2 nl (nl + 3) eps max|net_up|, which is the limit here; the measured residual is about 2e-16 of max|net_up| (100
    layers, float64). A larger one means the heating is not the flux divergence. The bound holds for float64
    arithmetic throughout, without underflow, so other precisions are refused rather than mixed.

    Parameters
    ----------
    heat : torch.Tensor [ncol, nl]
        Heating per volume, W m^-3.
    dz : torch.Tensor [ncol, nl] or [nl]
        Layer thicknesses, m.
    net_up : torch.Tensor [ncol, nl + 1]
        Net upward flux on the layer faces, bottom face first, W m^-2.

    Returns
    -------
    float
        The largest column residual relative to that column's largest |net_up|; 0 for a column with no flux.
    """
    kinds = {name: a.dtype for name, a in (("heat", heat), ("dz", dz), ("net_up", net_up))}
    if any(d != torch.float64 for d in kinds.values()):
        raise ValueError(f"check_budget's round-off bound is for float64; got {kinds}: pass float64 arrays")
    if heat.dim() != 2 or net_up.shape != (heat.shape[0], heat.shape[1] + 1) or dz.shape[-1] != heat.shape[1]:
        shapes = [tuple(a.shape) for a in (heat, dz, net_up)]
        raise ValueError(f"heat [ncol, nl], dz [ncol, nl] or [nl] and net_up [ncol, nl + 1] expected, got {shapes}")
    if not bool((torch.isfinite(dz) & (dz > 0)).all()):
        raise ValueError(
            f"dz must be finite and positive; its least value is {float(dz.min())!r} m, largest {float(dz.max())!r} m"
        )
    fine = torch.isfinite(net_up).all(-1) & torch.isfinite(heat).all(-1)
    if not bool(fine.all()):
        bad = torch.nonzero(~fine).flatten()
        raise RadiationBudgetError(
            f"non-finite radiative flux or heating in {bad.numel()} column(s), the first column {int(bad[0])}: "
            "check that column's temperature, density and pressure, which the radiation took as inputs"
        )
    nl = heat.shape[-1]
    scale = net_up.abs().amax(-1)
    rel = column_residual(heat, dz, net_up).abs() / torch.where(scale > 0, scale, 1.0)
    limit = 2.0 * nl * (nl + 3) * torch.finfo(heat.dtype).eps
    if not bool(torch.isfinite(rel).all()):
        raise RadiationBudgetError(
            f"column {int(torch.nonzero(~torch.isfinite(rel))[0])} has a non-finite budget residual from finite "
            "inputs (an overflow): check that column's fluxes"
        )
    worst = float(rel.max())
    if worst > limit:
        raise RadiationBudgetError(
            f"column {int(rel.argmax())} breaks its energy budget by {worst:.3e} of its largest net flux, above the "
            f"round-off limit {limit:.3e}: the heating is not the flux divergence; build it with column_heating"
        )
    return worst


def level_temperatures(T: torch.Tensor) -> torch.Tensor:
    """Return the temperatures on the layer faces of a uniform grid, for the longwave.

    Inside faces take the mean of their two layers; the end faces, half a layer beyond the end centres, take the line
    through the two end layers: 1.5 T0 - 0.5 T1. A T linear in height gives exact face values.

    Parameters
    ----------
    T : torch.Tensor [ncol, nl]
        Layer temperatures on a uniform grid, bottom first, K. nl >= 2.

    Returns
    -------
    torch.Tensor [ncol, nl + 1]
        Face temperatures, bottom face first, K.
    """
    if T.dim() != 2 or T.shape[1] < 2:
        raise ValueError(f"T must be [ncol, nl] with nl >= 2, got shape {tuple(T.shape)}")
    out = torch.empty((T.shape[0], T.shape[1] + 1), dtype=T.dtype, device=T.device)
    out[:, 0] = 1.5 * T[:, 0] - 0.5 * T[:, 1]
    out[:, 1:-1] = 0.5 * (T[:, :-1] + T[:, 1:])
    out[:, -1] = 1.5 * T[:, -1] - 0.5 * T[:, -2]
    return out


class RadiationClock:
    """When to recompute the radiation: at the first step time at or after each multiple of the period.

    The schedule is fixed, so the mean cadence is the period whatever the step: 70 s steps and a 300 s period give
    0, 350, 630, 910, 1260 s, ... A step longer than the period fires once and skips the multiples it passed.

    The clock keeps its own time as the exact sum of the steps it was given (a fraction, so a long run accumulates no
    rounding). A step or period written in decimal (70.1 s) is a float up to half an ulp off; the clock allows exactly
    that much per step and per multiple of the period, so a multiple the decimal steps reach exactly fires on time.

    Parameters
    ----------
    period_s : float
        The period, s; finite and positive.
    """

    def __init__(self, period_s: float) -> None:
        if not (math.isfinite(period_s) and period_s > 0):
            raise ValueError(f"the radiation period must be finite and positive, got {period_s!r} s")
        self.period_s = period_s
        # the least decimal period this float stands for, s
        self._period_low = Fraction(period_s) - Fraction(math.ulp(period_s)) / 2
        self._time = Fraction(0)  # the exact sum of the steps taken, s
        self._slack = Fraction(0)  # half an ulp of each step taken, s
        self._next = 0  # the next multiple of the period that fires

    def due(self) -> bool:
        """Return whether the radiation is due at the clock's time, and if so advance the schedule past it."""
        reached = self._time + self._slack
        if reached < self._next * self._period_low:
            return False
        self._next = math.floor(reached / self._period_low) + 1
        return True

    def advance(self, step_s: float) -> None:
        """Move the clock on by one step of ``step_s`` (s; finite, not negative)."""
        if not (math.isfinite(step_s) and step_s >= 0):
            raise ValueError(f"the step must be finite and not negative, got {step_s!r} s")
        self._time += Fraction(step_s)
        self._slack += Fraction(math.ulp(step_s)) / 2
