"""A stand-in for photolysis rates: a table's J at the layer nearest in log pressure on the day side, zero at night.

No cosine factor and no slant attenuation. A cell outside the table's pressure range takes the nearest edge row; the
lookup counts those cells so the driver can report them. docs/architecture.md places this module.
"""

from dataclasses import dataclass

import torch

__all__ = ["JLookup", "nearest_layer_rates"]


@dataclass(frozen=True)
class JLookup:
    """Photolysis rates of the cells, and the day-side cells that took an edge row of the table.

    Parameters
    ----------
    J : torch.Tensor [cells, branches]
        Photolysis rates, s^-1.
    above : int
        Day-side cells at a lower pressure than the table's top row.
    below : int
        Day-side cells at a higher pressure than the table's bottom row.
    """

    J: torch.Tensor
    above: int
    below: int


def nearest_layer_rates(
    pressure: torch.Tensor, table_pressure: torch.Tensor, table_rates: torch.Tensor, day: torch.Tensor
) -> JLookup:
    """Return each day-side cell the J of the table's layer nearest in log pressure, and zero on the night side.

    Parameters
    ----------
    pressure : torch.Tensor [cells]
        Cell pressures, Pa.
    table_pressure : torch.Tensor [layers]
        The table's layer pressures, Pa.
    table_rates : torch.Tensor [layers, branches]
        The table's photolysis rates, s^-1.
    day : torch.Tensor [cells]
        True where the star is above the horizon.

    Returns
    -------
    JLookup
        The rates and the counts of day-side cells outside the table's range.
    """
    if pressure.dim() != 1 or day.shape != pressure.shape:
        shapes = (tuple(pressure.shape), tuple(day.shape))
        raise ValueError(f"pressure and day must both be [cells], got shapes {shapes}")
    if table_rates.dim() != 2 or table_pressure.shape != table_rates.shape[:1]:
        shapes = (tuple(table_pressure.shape), tuple(table_rates.shape))
        raise ValueError(f"table_pressure [layers] and table_rates [layers, branches] disagree: {shapes}")
    if day.dtype != torch.bool:
        raise ValueError(f"day must be a boolean mask, got dtype {day.dtype}: pass cos z > 0")
    if table_pressure.numel() == 0:
        raise ValueError("the table has no layers: check the --jref file")
    for name, p in (("pressure", pressure), ("table_pressure", table_pressure)):
        if not bool((torch.isfinite(p) & (p > 0)).all()):
            raise ValueError(f"{name} must be finite and positive; its least value is {float(p.min())!r} Pa")
    if not bool((torch.isfinite(table_rates) & (table_rates >= 0)).all()):
        raise ValueError(
            f"table_rates must be finite and not negative; its least value is {float(table_rates.min())!r}"
        )
    i = (torch.log(pressure)[:, None] - torch.log(table_pressure)[None]).abs().argmin(1)
    top, bottom = table_pressure.min(), table_pressure.max()
    above = int((day & (pressure < top)).sum())
    below = int((day & (pressure > bottom)).sum())
    return JLookup(J=torch.where(day[:, None], table_rates[i], 0.0), above=above, below=below)
