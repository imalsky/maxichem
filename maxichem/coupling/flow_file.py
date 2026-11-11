"""Saved states that record the gas and grid they were made for: spun-up flows and radiative-equilibrium start columns.

A state saved under one gas and loaded under another keeps p and rho, so T = p / (rho R) changes silently (a pure-H2
flow in an H2 and He deck reads about 16 % hotter). Each file holds the deck's dry gas and grid beside the state, and
loading refuses a file whose record differs from the running deck, or that has none. docs/architecture.md places
this module.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["DryGas", "Flow", "FlowFileError", "load_flow", "load_start_column", "save_flow", "save_start_column"]

# Two sums of the same atomic weights (the model's and maxichem.coupling.composition's) agree to round-off; another gas
# differs by percents.
_MOLAR_MASS_RTOL = 1.0e-9


class FlowFileError(ValueError):
    """A saved state was made for another gas or grid, or does not record which."""


@dataclass(frozen=True)
class DryGas:
    """The dynamics' dry gas, as the deck states it and the model reports it.

    Parameters
    ----------
    composition : dict of str to float
        Atoms of each element per molecule, the deck's ``composition``.
    cv_R : float
        Heat capacity at constant volume over the gas constant, the deck's ``cv_R``.
    molar_mass : float
        Molar mass, kg/mol, as the model reports it.
    """

    composition: dict[str, float]
    cv_R: float  # noqa: N815 (the deck's own key)
    molar_mass: float

    def __post_init__(self) -> None:
        """Refuse a molar mass or heat capacity that is not finite and positive."""
        for name in ("cv_R", "molar_mass"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"DryGas.{name} must be finite and positive, got {value!r}")


@dataclass(frozen=True)
class Flow:
    """A loaded flow.

    Parameters
    ----------
    hydro_w : list of torch.Tensor
        Each block's primitive variables, ghosts included.
    tsim_s : float
        Model time at the save, s.
    """

    hydro_w: list[torch.Tensor]
    tsim_s: float


def _record(gas: DryGas, grid: dict[str, Any]) -> dict[str, Any]:
    return {"gas": asdict(gas), "grid": grid}


def _require_same(path: Path, saved: dict[str, Any] | None, gas: DryGas, grid: dict[str, Any], remake: str) -> None:
    """Refuse a missing record or one that differs from the deck's; ``remake`` says how to make a new file."""
    if saved is None:
        raise FlowFileError(
            f"{path} does not record the gas and grid it was made for (an older driver saved it), so it cannot be "
            f"checked against this deck ({gas}); loading it would give another T = p / (rho R). {remake}"
        )
    try:
        s_gas = {name: saved["gas"][name] for name in ("composition", "cv_R", "molar_mass")}
        s_grid = saved["grid"]
    except (KeyError, TypeError):
        raise FlowFileError(
            f"{path} holds a malformed record (it needs gas, with composition, cv_R and molar_mass, and grid), got "
            f"{saved!r}. {remake}"
        ) from None
    differ = [
        f"{name}: file {s_gas[name]!r}, deck {getattr(gas, name)!r}"
        for name in ("composition", "cv_R")
        if s_gas[name] != getattr(gas, name)
    ]
    if not math.isclose(s_gas["molar_mass"], gas.molar_mass, rel_tol=_MOLAR_MASS_RTOL):
        differ.append(f"molar_mass: file {s_gas['molar_mass']!r}, deck {gas.molar_mass!r} kg/mol")
    if s_grid != grid:
        differ.append(f"grid: file {s_grid!r}, deck {grid!r}")
    if differ:
        raise FlowFileError(f"{path} was made for another deck ({'; '.join(differ)}). {remake}")


def save_flow(path: str | Path, hydro_w: list[torch.Tensor], tsim_s: float, gas: DryGas, grid: dict[str, Any]) -> None:
    """Save a flow with the gas and grid it was made for.

    Parameters
    ----------
    path : str or Path
        The torch file to write.
    hydro_w : list of torch.Tensor
        Each block's primitive variables, ghosts included.
    tsim_s : float
        Model time, s.
    gas : DryGas
        The deck's dry gas.
    grid : dict
        The deck's ``geometry`` section.
    """
    torch.save({"hydro_w": hydro_w, "tsim": tsim_s, "record": _record(gas, grid)}, path)


def load_flow(
    path: str | Path, gas: DryGas, grid: dict[str, Any], shapes: list[tuple[int, ...]], device: torch.device
) -> Flow:
    """Load a flow saved by ``save_flow``, refusing one made for another gas or grid, or with no record.

    Parameters
    ----------
    path : str or Path
        The torch file.
    gas : DryGas
        The running deck's dry gas.
    grid : dict
        The running deck's ``geometry`` section.
    shapes : list of tuple of int
        The running deck's ``hydro_w`` shape of each block.
    device : torch.device
        Where the state goes.

    Returns
    -------
    Flow
        The state and its model time.
    """
    path = Path(path)
    saved = torch.load(path, map_location=device, weights_only=True)
    remake = "Re-spin the flow under this deck (--spin N --save FILE) and load that."
    _require_same(path, saved.get("record"), gas, grid, remake)
    got = [tuple(w.shape) for w in saved["hydro_w"]]
    if got != [tuple(s) for s in shapes]:
        raise FlowFileError(f"{path} holds blocks of shapes {got}, the deck has {shapes}. {remake}")
    return Flow(hydro_w=saved["hydro_w"], tsim_s=float(saved["tsim"]))


def save_start_column(path: str | Path, levels: np.ndarray, gas: DryGas, grid: dict[str, Any]) -> None:
    """Save a start column's level temperatures with the gas and grid they were made for.

    Parameters
    ----------
    path : str or Path
        The ``.npz`` file to write.
    levels : np.ndarray [nz + 1]
        Temperatures on the layer faces, bottom first, K.
    gas : DryGas
        The deck's dry gas.
    grid : dict
        The column's grid with its layer count ``nz``, for example ``{"nz": 140, "depth_m": 8.4e6}``.
    """
    _check_levels(Path(path), levels, grid)
    np.savez(path, L=levels, record=np.array(json.dumps(_record(gas, grid))))


def _check_levels(path: Path, levels: np.ndarray, grid: dict[str, Any]) -> None:
    """Refuse level temperatures that are not [nz + 1], finite and positive, or a grid without its layer count."""
    nz = grid.get("nz")
    if not isinstance(nz, int) or nz < 1:
        raise FlowFileError(f"{path}: the column's grid must give its layer count as an integer nz >= 1, got {grid!r}")
    if levels.shape != (nz + 1,):
        raise FlowFileError(f"{path}: {nz} layers need {nz + 1} level temperatures, got shape {levels.shape}")
    if not bool(np.all(np.isfinite(levels) & (levels > 0))):
        raise FlowFileError(f"{path}: level temperatures must be finite and positive; the least is {levels.min()!r} K")


def load_start_column(path: str | Path, gas: DryGas, grid: dict[str, Any]) -> np.ndarray:
    """Load a start column saved by ``save_start_column``, refusing one made for another gas or grid, or with no record.

    Parameters
    ----------
    path : str or Path
        The ``.npz`` file.
    gas : DryGas
        The running deck's dry gas.
    grid : dict
        The running deck's column grid, as it was saved.

    Returns
    -------
    np.ndarray [nz + 1]
        Level temperatures, bottom first, K.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        saved = json.loads(str(f["record"])) if "record" in f.files else None
        remake = "Make the column again with re_start.py under this deck's gas and grid, and start from that."
        # the record went through JSON, which reads a tuple back as a list: compare with the deck's grid read alike
        _require_same(path, saved, gas, json.loads(json.dumps(grid)), remake)
        levels = np.asarray(f["L"])
    _check_levels(path, levels, grid)
    return levels
