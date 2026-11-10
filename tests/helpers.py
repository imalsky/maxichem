"""What the GPU tests share: start gases, scored gases, tensors and the check against the stored reference."""

from dataclasses import dataclass
from typing import Any

import libref
import numpy as np
import torch

from maxichem import _checks, kincuda, kinvulcan, paths

H2S_PER_H2_RECORDED = 2.7e-5  # the sulfur the stored references' recorded starts were made with; part of their digest
# The known answers' constants, typed here so that a wrong constant in the solver cannot also move the answer
BOLTZMANN = 1.380649e-16  # erg K^-1 (the SI value, exact since 2019, in cgs)
STANDARD_PRESSURE = 1.0e6  # dyn cm^-2: 1 bar, the NASA-9 standard state

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


def warp_sum(row: list[float], perm: list[int]) -> float:
    """Return sum(x) as the call kernel's warp adds it, bit for bit (Python floats are float64).

    Lane l adds x[perm[i]] for i = l, l + 32, ..., then the 32 lanes meet in a butterfly.
    """
    lanes = [0.0] * 32
    for i, j in enumerate(perm):
        lanes[i % 32] += row[j]
    for step in (16, 8, 4, 2, 1):
        lanes = [lanes[lane] + lanes[lane ^ step] for lane in range(32)]
    return lanes[0]


def on_the_tolerance(rows: list[list[float]], perm: list[int], tolerance: float, warp_above: bool) -> list[float]:
    """Return one of rows, scaled so that |sum - 1| is above the tolerance in one order of the sum, below in the other.

    Above in the warp's order (warp_sum) if warp_above, else in the network's order; both within rounding of it.
    """
    for row in rows:
        total = warp_sum(row, perm)
        for k in range(-2000, 2000):
            scaled = [q / total * (1.0 + tolerance + 1e-18 * k) for q in row]
            warp, plain = abs(warp_sum(scaled, perm) - 1.0), abs(sum(scaled) - 1.0)
            if (warp > tolerance >= plain) if warp_above else (plain > tolerance >= warp):
                return scaled
    raise RuntimeError("no row scales to put the two orders of the sum on either side of the tolerance")


def number_density(T: Any, P: Any) -> Any:
    """Return n_tot = P / (k_B T) in cm^-3, from T in K and P in dyn cm^-2, with k_B typed here, not the solver's."""
    return np.asarray(P) / (BOLTZMANN * np.asarray(T))


def enthalpy_entropy(a: Any, T: float) -> tuple[float, float]:
    """Return H / RT and S / R from one NASA-9 range a1..a7, b1, b2 (McBride et al. 2002), written out here."""
    enthalpy = (
        -a[0] / T**2 + a[1] * np.log(T) / T + a[2] + a[3] * T / 2 + a[4] * T**2 / 3 + a[5] * T**3 / 4 + a[6] * T**4 / 5
    ) + a[7] / T
    entropy = (
        -a[0] / T**2 / 2 - a[1] / T + a[2] * np.log(T) + a[3] * T + a[4] * T**2 / 2 + a[5] * T**3 / 3 + a[6] * T**4 / 4
    ) + a[8]
    return float(enthalpy), float(entropy)


def gibbs_rt(a: Any, T: float) -> float:
    """Return G / RT from one NASA-9 range a1..a7, b1, b2."""
    enthalpy, entropy = enthalpy_entropy(a, T)
    return enthalpy - entropy


def constant_cp(enthalpy_r: float, cp_r: float = 3.5) -> list[float]:
    """Return one NASA-9 range of constant Cp / R whose H / R is enthalpy_r (K) plus Cp T / R, and S / R constant 0."""
    return [0.0, 0.0, cp_r, 0.0, 0.0, 0.0, 0.0, enthalpy_r, 0.0]


def write_network(folder: Any, text: str, thermo: dict[str, Any]) -> kinvulcan.VulcanNetwork:
    """Write an invented network and its NASA-9 files into folder and read them with the public reader.

    thermo maps each species to one range (both sides of the split) or a pair (below, above).
    """
    nasa = folder / "NASA9"
    nasa.mkdir()
    for name, ranges in thermo.items():
        low, high = ranges if isinstance(ranges, tuple) else (ranges, ranges)
        # a file holds a1..a7, an unused entry, b1, b2 for each range, five numbers per line
        numbers = [*low[:7], 0.0, *low[7:], *high[:7], 0.0, *high[7:]]
        rows = [" ".join(f"{q:.17g}" for q in numbers[i : i + 5]) for i in range(0, 20, 5)]
        (nasa / f"{name}.txt").write_text("\n".join(rows) + "\n")
    (folder / "network.txt").write_text(text)
    return kinvulcan.VulcanNetwork(folder / "network.txt", nasa)


RECOMBINATION = """# Invented test network: H + H + M -> H2 + M, the dissociation negligible

# 3-body reactions without high-pressure rates
1 [ H + H + M -> H2 + M ] 1.0E-32 0.0 0.0

# reverse stops
"""
RECOMBINATION_K0 = 1.0e-32  # cm6 s^-1
RECOMBINATION_T = 1000.0  # K
H2_BOUND = -2.0e5  # K: H / R of H2 below that of 2 H, so the dissociation is negligible


def recombination(folder: Any) -> tuple[kinvulcan.VulcanNetwork, dict[str, Any]]:
    """Return the network of H + H + M -> H2 + M, and its NASA-9 ranges by species."""
    atom = constant_cp(0.0, 2.5)
    thermo = {"H": atom, "H2": constant_cp(H2_BOUND), "He": atom}
    return write_network(folder, RECOMBINATION, thermo), thermo


def recombination_time(n: Any, P: float) -> Any:
    """Return t (s) at which N_H = n, from pure H at RECOMBINATION_T and P (dyn cm^-2).

    With N per starting molecule, N_H2 = (1 - N) / 2 and sum N = (1 + N) / 2, so dN/dt = -2 k0 n_tot^2 N^2 / sum N,
    and 1 / N - ln N = 1 + 4 k0 n_tot^2 t. The mole fraction of H is 2 N / (1 + N). The sum N / n_tot factor of the
    constant-pressure form is what makes it differ from the constant-volume answer 1 / N = 1 + 2 k0 n_tot^2 t.
    """
    nd = number_density(RECOMBINATION_T, P)
    return (1.0 / n - np.log(n) - 1.0) / (4.0 * RECOMBINATION_K0 * nd**2)


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
