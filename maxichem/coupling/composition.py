"""The tracers' composition: molar masses, mass mixing ratios and mole fractions, element totals, the start mixture.

The GCM carries each species as a mass mixing ratio q (kg per kg of gas); the chemistry takes mole fractions. Each cell
keeps its tracer mass through a chemistry call. docs/architecture.md places this module.
"""

from collections.abc import Sequence

import numpy as np
import torch

__all__ = [
    "ATOMIC_MASS",
    "GAS_CONSTANT",
    "START_PER_H",
    "element_moles",
    "mass_to_mole",
    "molar_masses",
    "mole_to_mass",
    "signed_mole_fractions",
    "start_mole_fractions",
    "tracer_totals",
]

# J mol^-1 K^-1: k_B N_A, exact in the SI. kintera, which snapy's EOS uses, has 8.31446 (3e-7 lower).
GAS_CONSTANT = 8.31446261815324

# kg/mol: pyharp's atomic weights (harp/element.cpp), so the tracers and the dynamics' gas weigh atoms alike
ATOMIC_MASS = {"H": 1.008e-3, "He": 4.002602e-3, "C": 12.011e-3, "N": 14.007e-3, "O": 15.999e-3, "S": 32.06e-3}

# Elements per H nucleus at the start. O, C, N and S are the rows of VULCAN-JAX's thermo/solar_element_abundances.dat
# (dex, H = 12): O 8.7826, C 8.4434, N 7.9130, S 7.1492, as 10^(dex - 12) to five digits. He/H is the owner's choice,
# not the file's 0.0838: near the dynamics' gas {H 1.68, He 0.16} (He/H 0.0952); helium is inert, and the 1.2 %
# difference in the gas constant between the tracers and the dynamics was accepted.
START_PER_H = {
    "O": 6.0618e-4,
    "C": 2.7759e-4,
    "N": 8.1846e-5,
    "S": 1.4099e-5,
    "He": 0.09692,
}

# The molecule that holds each element at the start: all C in CO, the rest of O in H2O, N in N2, S in H2S.
_CARRIER = {"H": "H2", "He": "He", "C": "CO", "O": "H2O", "N": "N2", "S": "H2S"}
_OPTIONAL = ("N", "S")  # a network without these elements starts without them


def _check_molar_masses(mw: torch.Tensor) -> None:
    if mw.dim() != 1:
        raise ValueError(f"mw must be [S], got shape {tuple(mw.shape)}")
    if not bool((torch.isfinite(mw) & (mw > 0)).all()):
        raise ValueError(f"mw must be finite and positive, its least value is {float(mw.min())!r} kg/mol")


def _check_cells(name: str, a: torch.Tensor, mw: torch.Tensor) -> None:
    """Refuse a [cells, S] array of the wrong shape or with non-finite entries, and a bad molar-mass vector."""
    _check_molar_masses(mw)
    if a.dim() != 2 or a.shape[1] != mw.shape[0]:
        raise ValueError(f"{name} must be [cells, {mw.shape[0]}], got shape {tuple(a.shape)}")
    if not bool(torch.isfinite(a).all()):
        raise ValueError(f"{name} holds {int((~torch.isfinite(a)).sum())} non-finite entries")


def _check_per_cell(name: str, a: torch.Tensor, cells: int) -> None:
    """Refuse a per-cell array that is not exactly [cells], or is negative or not finite."""
    if a.shape != (cells,):
        raise ValueError(f"{name} must be [{cells}], got shape {tuple(a.shape)}")
    if not bool((torch.isfinite(a) & (a >= 0)).all()):
        raise ValueError(f"{name} must be finite and not negative, its least value is {float(a.min())!r}")


def molar_masses(atoms: np.ndarray | torch.Tensor, elements: Sequence[str]) -> torch.Tensor:
    """Return each species' molar mass from its atoms.

    Parameters
    ----------
    atoms : array [S, E]
        Atoms of each element per molecule, as ``VulcanNetwork.atoms``.
    elements : sequence of str [E]
        The elements of the columns, as ``VulcanNetwork.elements``.

    Returns
    -------
    torch.Tensor [S]
        Molar masses, kg/mol, float64 on the CPU.
    """
    missing = [e for e in elements if e not in ATOMIC_MASS]
    if missing:
        raise ValueError(f"no atomic mass for {missing}: add them to maxichem.coupling.composition.ATOMIC_MASS")
    a = torch.as_tensor(atoms, dtype=torch.float64).cpu()
    if a.dim() != 2 or a.shape[1] != len(elements):
        raise ValueError(f"atoms must be [species, {len(elements)}], got shape {tuple(a.shape)}")
    if not bool((torch.isfinite(a) & (a >= 0)).all()):
        raise ValueError("atoms must be finite and not negative: check the network's atoms table")
    mw = a @ torch.tensor([ATOMIC_MASS[e] for e in elements], dtype=torch.float64)
    empty = torch.nonzero(mw <= 0).flatten().tolist()
    if empty:
        raise ValueError(f"species rows {empty} have no atoms, so no molar mass: check the network's atoms table")
    return mw


def mass_to_mole(q: torch.Tensor, mw: torch.Tensor) -> torch.Tensor:
    """Return the mole fractions of the tracers in each cell.

    Parameters
    ----------
    q : torch.Tensor [cells, S]
        Mass mixing ratios, kg per kg of gas; finite and not negative (clamp the transport's negatives first).
    mw : torch.Tensor [S]
        Molar masses, kg/mol.

    Returns
    -------
    torch.Tensor [cells, S]
        Mole fractions, summing to 1 in each cell.
    """
    _check_cells("q", q, mw)
    if q.dtype != mw.dtype:
        raise ValueError(f"q and mw must share one precision, got {q.dtype}, {mw.dtype}")
    if bool((q < 0).any()):
        raise ValueError(f"q holds negatives, the least {float(q.min()):.3g}: clamp them before the conversion")
    n = q / mw
    total = n.sum(1, keepdim=True)
    if bool((total <= 0).any()):
        cell = int(torch.nonzero(total.flatten() <= 0)[0])
        raise ValueError(f"cell {cell} holds no tracer mass, so it has no mole fractions: check its tracers")
    return n / total


def signed_mole_fractions(q: torch.Tensor, mw: torch.Tensor) -> torch.Tensor:
    """Return mole fractions for output, keeping the transport's negatives; never feed these to the chemistry.

    ``mass_to_mole`` refuses negative q; a state written for figures keeps them, so they stay visible.

    Parameters
    ----------
    q : torch.Tensor [cells, S]
        Mass mixing ratios, kg per kg of gas, signed.
    mw : torch.Tensor [S]
        Molar masses, kg/mol.

    Returns
    -------
    torch.Tensor [cells, S]
        n / sum(n) with n = q / mw, summing to 1 in each cell.
    """
    _check_cells("q", q, mw)
    n = q / mw
    return n / n.sum(1, keepdim=True)


def mole_to_mass(x: torch.Tensor, mw: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Return mass mixing ratios that hold each cell's given tracer mass.

    Parameters
    ----------
    x : torch.Tensor [cells, S]
        Mole fractions, finite and not negative.
    mw : torch.Tensor [S]
        Molar masses, kg/mol.
    mass : torch.Tensor [cells]
        Each cell's tracer mass, sum of q, kg per kg of gas; finite and not negative.

    Returns
    -------
    torch.Tensor [cells, S]
        Mass mixing ratios, kg per kg of gas, summing to ``mass`` in each cell.
    """
    _check_cells("x", x, mw)
    _check_per_cell("mass", mass, x.shape[0])
    if not (x.dtype == mw.dtype == mass.dtype):
        raise ValueError(f"x, mw and mass must share one precision, got {x.dtype}, {mw.dtype}, {mass.dtype}")
    if bool((x < 0).any()):
        raise ValueError(f"x holds negatives, the least {float(x.min()):.3g}: the call's result is not usable")
    xm = x * mw
    weight = xm.sum(1)
    if bool((weight <= 0).any()):
        cell = int(torch.nonzero(weight <= 0)[0])
        raise ValueError(f"cell {cell} has mole fractions of no mass (sum of x * mw <= 0): check the call's result")
    return xm * (mass / weight)[:, None]


def element_moles(q: torch.Tensor, mw: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Return each cell's moles of each element per kg of gas.

    Parameters
    ----------
    q : torch.Tensor [cells, S]
        Mass mixing ratios, kg per kg of gas, finite (negatives from the transport are counted as they are).
    mw : torch.Tensor [S]
        Molar masses, kg/mol.
    atoms : torch.Tensor [S, E]
        Atoms of each element per molecule.

    Returns
    -------
    torch.Tensor [cells, E]
        mol per kg of gas.
    """
    _check_cells("q", q, mw)
    if atoms.dim() != 2 or atoms.shape[0] != mw.shape[0]:
        raise ValueError(f"atoms must be [{mw.shape[0]}, elements], got shape {tuple(atoms.shape)}")
    if not bool(torch.isfinite(atoms).all()):
        raise ValueError("atoms holds non-finite entries: check the network's atoms table")
    if not (q.dtype == mw.dtype == atoms.dtype):
        raise ValueError(f"q, mw and atoms must share one precision, got {q.dtype}, {mw.dtype}, {atoms.dtype}")
    return (q / mw) @ atoms


def tracer_totals(q: torch.Tensor, cell_mass: torch.Tensor, mw: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Return the tracers' mass and each element's moles summed over the cells.

    Parameters
    ----------
    q : torch.Tensor [cells, S]
        Mass mixing ratios, kg per kg of gas, finite.
    cell_mass : torch.Tensor [cells]
        Gas mass of each cell (density times volume), kg; finite and not negative.
    mw : torch.Tensor [S]
        Molar masses, kg/mol.
    atoms : torch.Tensor [S, E]
        Atoms of each element per molecule.

    Returns
    -------
    torch.Tensor [1 + E]
        The tracer mass in kg, then the moles of each element in the atoms table's column order.
    """
    moles = element_moles(q, mw, atoms)
    _check_per_cell("cell_mass", cell_mass, q.shape[0])
    mass = (q * cell_mass[:, None]).sum()
    return torch.cat([mass.view(1), (moles * cell_mass[:, None]).sum(0)])


def start_mole_fractions(species: Sequence[str], elements: Sequence[str]) -> torch.Tensor:
    """Return the start gas: H2 and He with ``START_PER_H`` of each element, in its carrier molecule.

    All C is in CO, the rest of O in H2O, N in N2 and S in H2S; a network without N or S starts without them. A
    chemistry call relaxes this gas to the cell's state.

    Parameters
    ----------
    species : sequence of str [S]
        The network's species, in its order.
    elements : sequence of str [E]
        The network's elements.

    Returns
    -------
    torch.Tensor [S]
        Mole fractions, float64 on the CPU, summing to 1.
    """
    unknown = [e for e in elements if e not in _CARRIER]
    if unknown:
        raise ValueError(
            f"the network has elements {unknown} with no start carrier: add one to maxichem.coupling.composition"
        )
    need = [e for e in _CARRIER if e in elements or e not in _OPTIONAL]
    lacking = [f"{_CARRIER[e]} (for {e})" for e in need if _CARRIER[e] not in species]
    if lacking:
        raise ValueError(f"the network lacks the start species {lacking}: the start mixture needs them")
    r = {e: (START_PER_H[e] if e in need else 0.0) for e in START_PER_H}
    # H nuclei per H2 molecule: 2 in H2, and 2 more per H2O and per H2S, each counted per H nucleus
    nh = 2.0 / (1.0 - 2.0 * (r["O"] - r["C"]) - 2.0 * r["S"])
    moles = {"H2": 1.0, "He": r["He"] * nh, "CO": r["C"] * nh, "H2O": (r["O"] - r["C"]) * nh}
    if "N" in need:
        moles["N2"] = 0.5 * r["N"] * nh
    if "S" in need:
        moles["H2S"] = r["S"] * nh
    x = torch.zeros(len(species), dtype=torch.float64)
    for name, n in moles.items():
        x[list(species).index(name)] = n
    return x / x.sum()
