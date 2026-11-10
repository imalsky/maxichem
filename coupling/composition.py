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
    "start_mole_fractions",
    "tracer_totals",
]

GAS_CONSTANT = 8.314462618  # J mol^-1 K^-1, CODATA (exact); kintera, which snapy's EOS uses, has 8.31446 (3e-7 lower)

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


def _check_rows(name: str, a: torch.Tensor, columns: int) -> None:
    if a.dim() != 2 or a.shape[1] != columns:
        raise ValueError(f"{name} must be [cells, {columns}], got shape {tuple(a.shape)}")


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
        raise ValueError(f"no atomic mass for {missing}: add them to coupling.composition.ATOMIC_MASS")
    a = torch.as_tensor(atoms, dtype=torch.float64).cpu()
    if a.dim() != 2 or a.shape[1] != len(elements):
        raise ValueError(f"atoms must be [species, {len(elements)}], got shape {tuple(a.shape)}")
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
    _check_rows("q", q, mw.numel())
    if not bool(torch.isfinite(q).all()):
        raise ValueError(f"q holds {int((~torch.isfinite(q)).sum())} non-finite entries: the transport broke first")
    if bool((q < 0).any()):
        raise ValueError(f"q holds negatives, the least {float(q.min()):.3g}: clamp them before the conversion")
    n = q / mw
    total = n.sum(1, keepdim=True)
    if bool((total <= 0).any()):
        cell = int(torch.nonzero(total.flatten() <= 0)[0])
        raise ValueError(f"cell {cell} holds no tracer mass, so it has no mole fractions: check its tracers")
    return n / total


def mole_to_mass(x: torch.Tensor, mw: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Return mass mixing ratios that hold each cell's given tracer mass.

    Parameters
    ----------
    x : torch.Tensor [cells, S]
        Mole fractions, finite.
    mw : torch.Tensor [S]
        Molar masses, kg/mol.
    mass : torch.Tensor [cells]
        Each cell's tracer mass, sum of q, kg per kg of gas.

    Returns
    -------
    torch.Tensor [cells, S]
        Mass mixing ratios, kg per kg of gas, summing to ``mass`` in each cell.
    """
    _check_rows("x", x, mw.numel())
    if mass.shape != x.shape[:1]:
        raise ValueError(f"mass must be [{x.shape[0]}], got shape {tuple(mass.shape)}")
    if not (bool(torch.isfinite(x).all()) and bool(torch.isfinite(mass).all())):
        raise ValueError("x or mass holds non-finite entries: the chemistry call's result is not usable")
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
        Mass mixing ratios, kg per kg of gas.
    mw : torch.Tensor [S]
        Molar masses, kg/mol.
    atoms : torch.Tensor [S, E]
        Atoms of each element per molecule.

    Returns
    -------
    torch.Tensor [cells, E]
        mol per kg of gas.
    """
    _check_rows("q", q, mw.numel())
    if atoms.shape[0] != mw.numel():
        raise ValueError(f"atoms has {atoms.shape[0]} species rows, mw has {mw.numel()}")
    return (q / mw) @ atoms


def tracer_totals(q: torch.Tensor, cell_mass: torch.Tensor, mw: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Return the tracers' mass and each element's moles summed over the cells.

    Parameters
    ----------
    q : torch.Tensor [cells, S]
        Mass mixing ratios, kg per kg of gas.
    cell_mass : torch.Tensor [cells]
        Gas mass of each cell (density times volume), kg.
    mw : torch.Tensor [S]
        Molar masses, kg/mol.
    atoms : torch.Tensor [S, E]
        Atoms of each element per molecule.

    Returns
    -------
    torch.Tensor [1 + E]
        The tracer mass in kg, then the moles of each element in the atoms table's column order.
    """
    if cell_mass.shape != q.shape[:1]:
        raise ValueError(f"cell_mass must be [{q.shape[0]}], got shape {tuple(cell_mass.shape)}")
    mass = (q * cell_mass[:, None]).sum()
    moles = (element_moles(q, mw, atoms) * cell_mass[:, None]).sum(0)
    return torch.cat([mass.view(1), moles])


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
        raise ValueError(f"the network has elements {unknown} with no start carrier: add one to coupling.composition")
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
