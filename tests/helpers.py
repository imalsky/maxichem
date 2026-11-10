"""What the GPU tests share: start gases, scored gases, tensors and the check against the stored reference."""

from dataclasses import dataclass
from typing import Any

import libref
import numpy as np
import torch

from maxichem import _checks, kincuda, kinvulcan, paths

H2S_PER_H2_RECORDED = 2.7e-5  # the sulfur the stored references' recorded starts were made with; part of their digest

REFERENCES = {"NCHO_photo_network.txt": paths.libref, "SNCHO_photo_network_2025.txt": paths.libref_sulfur}
CHAIN_PATHS = 256
CHAIN_OFFSET = 1500  # forcing rows between the four offsets of each path

# The parity limits. The reference runs in torch, its batched sums in another order, so it is not bitwise the kernel.
TOL = 1e-6  # dex, on the scored gases: rounding of the two sums; measured worst 2.8e-8 (NCHO_photo), 7.7e-8 (sulfur)
CAP = 1e-5  # dex: no cell-call above this; a reference whose step sequence branches differently can reach it
OVER_TOL_SHARE = 1e-5  # the share of cell-calls allowed above TOL, at least one per case
STEP_MISMATCH_SHARE = 1e-4  # the share of cell-calls whose accepted steps may differ, for the same reason


def tensor(values: Any, device: Any) -> torch.Tensor:
    """Return an array as a float64 tensor."""
    return torch.tensor(np.asarray(values), dtype=torch.float64, device=device)


def full(cells: int, value: float, device: Any) -> torch.Tensor:
    """Return a float64 tensor of this many cells filled with value."""
    return torch.full((cells,), value, dtype=torch.float64, device=device)


def state(net: Any, cells: int, device: Any, **fractions: float) -> torch.Tensor:
    """Return mole fractions [cells, species] with the given species and the rest zero."""
    x = torch.zeros(cells, len(net.species), dtype=torch.float64, device=device)
    for name, value in fractions.items():
        x[:, net.species.index(name)] = value
    return x / x.sum(1, keepdim=True)


def decay_answer(t: Any) -> tuple[Any, Any]:
    """Return the mole fractions of A and B for A -> 2 B at constant pressure, from pure A, at J t."""
    survive = np.exp(-t)
    total = 2.0 - survive
    return survive / total, 2.0 * (1.0 - survive) / total


@dataclass
class Network:
    """A real network with what the reference checks need."""

    name: str
    net: Any
    species: list[str]
    gases: list[str]
    floors: list[float]
    gas_index: list[int]
    signature: str
    reference: Any
    device: Any

    def floored(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return per cell the worst |log10 max(a, S) - log10 max(b, S)| over the scored gases."""
        floor = torch.log10(torch.tensor(self.floors, dtype=torch.float64, device=self.device))
        la, lb = (_checks.floored_log10(q, floor, torch) for q in (a, b))
        return (la - lb).abs().max(1).values.nan_to_num(nan=np.inf)

    def stored_call(self, stored: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return a stored reference call as tensors: x on the scored gases, accepted steps, finished, finite."""
        return {
            "x": tensor(stored["x"], self.device),
            "steps": tensor(stored["steps"], self.device),
            "done": torch.tensor(stored["done"], device=self.device),
            "finite": torch.tensor(stored["finite"], device=self.device),
        }

    def entry(self, key: str, *inputs: Any) -> dict[str, torch.Tensor]:
        """Return a stored reference call, refusing inputs other than those it was written for."""
        return self.stored_call(libref.entry(key, self.signature, *inputs, path=self.reference))


def check_reference(reference: Any, species: list[str], gases: list[str]) -> None:
    """Refuse a reference made at another atol, species order or set of gases."""
    stored = libref.load(reference)
    assert float(stored["atol"]) == kincuda.ATOL, "the reference was made at another atol"
    assert [str(q) for q in stored["species"]] == species, "the reference's species order differs from the reader's"
    assert [str(q) for q in stored["gases"]] == gases, "the reference stores other gases"


class Tally:
    """Parity counts over the calls of one case."""

    def __init__(self, network: Network, frac: float = OVER_TOL_SHARE, cap: float = CAP) -> None:
        self.network = network
        self.worst, self.calls, self.mis, self.nf, self.unf, self.redo = 0.0, 0, 0, [0, 0], [0, 0], 0
        self.frac, self.cap, self.over = frac, cap, 0

    def add(self, ref: dict[str, torch.Tensor], mod: Any, solver: Any) -> None:
        """Add one call's comparison to the counts."""
        xb, _, db, nb = mod
        e = self.network.floored(ref["x"], xb[:, self.network.gas_index])
        self.worst = max(self.worst, float(e.max()))
        self.over += int((e > TOL).sum())
        self.calls += xb.shape[0]
        self.mis += int((ref["steps"] != nb).sum())
        for i, (fin, d) in enumerate(((ref["finite"], ref["done"]), (torch.isfinite(xb).all(1), db))):
            self.nf[i] += int(torch.logical_not(fin).sum())
            self.unf[i] += int(torch.logical_not(d).sum())
        self.redo += solver.fallback["recomputed"]

    def report(self, verdict: Any, name: str) -> None:
        """Report the case from the counts."""
        ok = (
            self.worst <= self.cap
            and self.over <= max(1.0, self.frac * self.calls)
            and self.mis <= STEP_MISMATCH_SHARE * self.calls
            and self.nf == [0, 0]
            and self.unf == [0, 0]
            and self.redo == 0
        )
        verdict(
            name,
            ok,
            f"worst {self.worst:.2e} dex ({self.over} cell-calls above {TOL:g}); step-count mismatches "
            f"{self.mis} of {self.calls} cell-calls; "
            f"non-finite ref/kernel {self.nf[0]}/{self.nf[1]}; unfinished {self.unf[0]}/{self.unf[1]}; "
            f"recomputed by the guard {self.redo}",
        )


class Parity:
    """A network with its reference, solvers and recorded starts."""

    def __init__(self, name: str, device: Any) -> None:
        thermo = paths.vulcan_thermo()
        reference = REFERENCES[name]()
        self.yard, self.panel = paths.yard() / "design", paths.designs() / "decision/P"
        paths.need(self.yard, self.panel, reference)
        net = kinvulcan.VulcanNetwork(thermo / name, thermo / "NASA9")
        gases, floors = _checks.scored_gases(net.elements)
        species = list(net.species)
        check_reference(reference, species, gases)
        signature = libref.netsig(thermo / name, thermo / "NASA9", species)
        gas_index = [species.index(g) for g in gases]
        self.network = Network(name, net, species, gases, floors, gas_index, signature, reference, device)
        self.device = device
        self.atoms = tensor(np.asarray(net.atoms, np.float64), device)
        self.units = np.load(self.yard / "units.npz")
        self.species53 = [str(s) for s in self.units["species53"]]
        self.rows = np.load(self.yard / "rows.npz")
        self.solver = kincuda.CudaSolver(net, device)
        self.lit = kincuda.CudaSolver(net, device, photo=True)
        self.J = tensor(libref.arrays("photo", path=reference)["J"], device)

    def close(self) -> None:
        """Unload the kernels."""
        self.solver.close()
        self.lit.close()

    def mapped(self, x53: np.ndarray) -> np.ndarray:
        """Map 53-species states onto the network's species by name, renormalized; H2S from H2 on a sulfur network."""
        species = self.network.species
        x = np.zeros((len(x53), len(species)))
        for j, q in enumerate(species):
            if q in self.species53:
                x[:, j] = x53[:, self.species53.index(q)]
        if "H2S" in species:
            x[:, species.index("H2S")] = H2S_PER_H2_RECORDED * x53[:, self.species53.index("H2")]
        return x / x.sum(1, keepdims=True)

    def starts(self, cells: int, seed: int, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return recorded starts (x, T, P) on the network, each species scaled by 10^(sigma N(0, 1))."""
        row_of_unit = {int(q): i for i, q in enumerate(self.rows["unit"])}
        rng = np.random.default_rng(seed)
        units = rng.choice(len(self.units["t0"]), cells, replace=True)
        rows = np.array([row_of_unit[int(q)] for q in units])
        x = self.mapped(self.units["x53"][units])
        if sigma:
            x = x * 10.0 ** (sigma * rng.standard_normal(x.shape))
            x = x / x.sum(1, keepdims=True)
        return (
            tensor(x, self.device),
            tensor(self.rows["T"][rows], self.device),
            tensor(self.rows["P_barye"][rows], self.device),
        )

    def full(self, cells: int, value: float) -> torch.Tensor:
        """Return a float64 tensor of this many cells filled with value."""
        return full(cells, value, self.device)

    def chain_inputs(self, cells: int, calls: int) -> tuple[Any, ...]:
        """Return panel P's starts, step sizes and forcing (T, P, dt per call) for a chain of calls."""
        rows, starts, forcing = (np.load(self.panel / f"{f}.npz") for f in ("rows", "starts", "forcing"))
        assert [str(q) for q in starts["species53"]] == self.species53
        path, offset = np.arange(cells) % CHAIN_PATHS, (np.arange(cells) // CHAIN_PATHS) * CHAIN_OFFSET
        t_chain, p_chain, dt_chain = (
            np.stack([np.asarray(q, np.float64)[path, offset + k] for k in range(calls)])
            for q in (forcing["T"], forcing["Pbar"] * 1e6, forcing["dt"])
        )
        x = tensor(self.mapped(starts["x53"][rows["unit"][path]]), self.device)
        return x, self.full(cells, 1e-10), t_chain, p_chain, dt_chain

    def shares(self, x: torch.Tensor) -> torch.Tensor:
        """Return each element's share of the atoms, per cell."""
        elements = x @ self.atoms
        return elements / elements.sum(1, keepdim=True)
