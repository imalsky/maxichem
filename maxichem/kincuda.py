"""Batched GPU solver: one call per cell of a VULCAN network, in CUDA kernels.

Public names: ``CudaSolver``, ``check_inputs``, and the constants ``ATOL``, ``COUNTS``, ``DRIFT``, ``K_COLL``,
``REC_STEPS``, ``T_MIN``. The method is in docs/method.md, the kernels' layout in docs/architecture.md.
"""

import ctypes
import math
import operator
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

from ._constants import (
    ATOL,
    COUNT_SLOTS,
    COUNTS,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
    DRIFT,
    GUARD_BLOCK_THREADS,
    K_COLL,
    KERNEL_ARGUMENTS,
    MAX_NAMED_BARRIERS,
    MAX_THREADS_PER_BLOCK,
    PHOTOLYSIS_ARGUMENT,
    REC_STEPS,
    REGISTER_ALLOCATION_UNIT,
    T_MIN,
)
from ._cuda import (
    ATTR_LOCAL_BYTES,
    ATTR_REGISTERS,
    SOLVER_KERNELS,
    blocks_per_multiprocessor,
    check_driver,
    compile_cubin,
    device_of,
    driver,
    function_attribute,
    launch_kernel,
    load_module,
    on_device,
    unload_at_deletion,
    unload_modules,
)
from ._export import write_export
from ._kernel_header import call_header, kernel_sources
from ._tables import (
    pack_small_tables,
    rates_fit_in_block,
    shared_doubles_per_cell,
    shared_doubles_recover,
    tables_for_device,
)
from .kinvulcan import VulcanNetwork

__all__ = ["ATOL", "COUNTS", "DRIFT", "K_COLL", "REC_STEPS", "T_MIN", "CudaSolver", "check_inputs"]


def check_inputs(
    dev: torch.device,
    ns: int,
    x: torch.Tensor,
    T: torch.Tensor,
    P: torch.Tensor,
    dt: torch.Tensor,
    h: torch.Tensor,
    rtol: float,
    atol: float,
    max_steps: int,
) -> int:
    """Check the call's inputs before any launch and return the number of cells.

    Parameters
    ----------
    dev : torch.device
        The solver's device.
    ns : int
        Species in the network.
    x : torch.Tensor
        Mole fractions [B, ns], float64 on ``dev``.
    T, P, dt, h : torch.Tensor
        Temperature (K), pressure (dyn cm^-2), call length (s), starting step (s): [B], float64 on ``dev``.
    rtol, atol : float
        Tolerances; finite and positive. atol is in mole fraction.
    max_steps : int
        Tries per call, positive; ``REC_STEPS`` times it must fit an int32.

    Returns
    -------
    int
        The number of cells B.

    Raises
    ------
    ValueError
        If an input has the wrong shape, dtype, device or range.
    """
    if x.dim() != 2 or x.shape[1] != ns:
        raise ValueError(f"x [B, {ns}] expected, got {tuple(x.shape)}")
    B = x.shape[0]
    for name, a in (("x", x), ("T", T), ("P", P), ("dt", dt), ("h", h)):
        if a.device != dev or a.dtype != torch.float64:
            raise ValueError(f"{name}: float64 on {dev} expected, got {a.dtype} on {a.device}")
        if name != "x" and tuple(a.shape) != (B,):
            raise ValueError(f"{name} [{B}] expected, got {tuple(a.shape)}")
    if not (math.isfinite(rtol) and rtol > 0.0 and math.isfinite(atol) and atol > 0.0):
        raise ValueError(f"rtol and atol finite and positive expected, got {rtol}, {atol}")
    try:
        ms = operator.index(max_steps)
    except TypeError:
        raise ValueError(f"max_steps: an integer expected, got {max_steps!r}") from None
    if ms < 1 or REC_STEPS * ms > 2**31 - 1:
        raise ValueError(f"max_steps from 1 to {(2**31 - 1) // REC_STEPS} expected, got {ms}")
    return int(B)


class CudaSolver:
    """Batched chemistry solver on one GPU.

    Calling it advances each cell by dt, with no host sync. Calls run one at a time on torch's current stream: the
    queues, the counts and the scratch are shared. See docs/api.md for its members and docs/method.md for the method.

    Parameters
    ----------
    net : VulcanNetwork
        The network.
    device : torch.device or str
        The CUDA device.
    drift : float, optional
        The element-drift allowance, relative; the default is ``DRIFT``.
    photo : bool
        The network's photolysis branches enter the call, which then takes J.
    """

    def __init__(
        self, net: VulcanNetwork, device: torch.device | str, drift: float | None = None, photo: bool = False
    ) -> None:
        if photo and (not net.photo or net.photo_missing):
            raise ValueError(
                "photo=True needs a network with photolysis branches whose species are all in its species "
                f"list (branches {len(net.photo)}, missing species {net.photo_missing})"
            )
        self.dev = device_of(device)
        self.drift = float(DRIFT if drift is None else drift)
        self._modules: list[Any] = []  # every module loaded, also when the setup fails part of the way
        self._setup(net, photo)

    @on_device
    def _setup(self, net: VulcanNetwork, photo: bool) -> None:
        """Build the kernels and their tables on the solver's device."""
        torch.zeros(1, device=self.dev)  # torch's context exists from here on
        prop = torch.cuda.get_device_properties(self.dev)
        tab = tables_for_device(net, photo, prop.shared_memory_per_block_optin)
        self.ns, self.nr, self.w, self.nj = tab["ns"], tab["nr"], tab["warps"], tab["nj"]
        threads = 32 * self.w
        blob, _ = pack_small_tables(tab)
        # the recover kernel's W in shared memory if it fits
        wsh = 8 * shared_doubles_recover(tab, True) + len(blob) <= prop.shared_memory_per_block_optin
        arch = f"{prop.major}{prop.minor}"
        self._load_packed(tab, wsh, arch, prop)
        self._build = (tab, wsh, arch)  # what export() recompiles
        self._names = (list(net.species), list(net.elements))  # export() writes them
        self.wpb = self.w * self.cpb  # warps per block
        self.smem = 8 * self.cpb * shared_doubles_per_cell(tab) + len(blob)
        self.smem_rec = 8 * shared_doubles_recover(tab, wsh) + len(blob)
        blocks = blocks_per_multiprocessor(self._fn, threads * self.cpb, self.smem)
        rec_blocks = blocks_per_multiprocessor(self._rec, threads, self.smem_rec)
        self.max_blocks, self.rec_blocks = prop.multi_processor_count * blocks, prop.multi_processor_count * rec_blocks
        # global scratch (doubles): per cell slot of the call kernel the rates and the copy for a retry, per block of
        # the recover kernel the same and W when it is not in shared memory
        retry = 2 * self.ns + tab["nnz"]
        self.stride, self.stride_rec = 2 * self.nr + 2 + retry, 2 * self.nr + 2 + retry + (0 if wsh else self.ns**2)
        self.info = {
            "ns": self.ns,
            "nr": self.nr,
            "nj": self.nj,
            "nnz": tab["nnz"],
            "k0": tab["k0"],
            "nt": self.ns - tab["k0"],
            "w": self.w,
            "sround": tab["sst"].shape[0],
            "jround": tab["jround"],
            "lu_ops": len(tab["lu"]),
            "alias": rates_fit_in_block(tab),
            "regs": function_attribute(self._fn, ATTR_REGISTERS),
            "spill_bytes": function_attribute(self._fn, ATTR_LOCAL_BYTES),
            "cpb": self.cpb,
            "wpb": self.wpb,
            "smem": self.smem,
            "blob": len(blob),
            "cells_per_sm": self.cpb * blocks,
            "rec_smem": self.smem_rec,
            "rec_w_shared": wsh,
            "rec_blocks": self.rec_blocks,
            "drift": self.drift,
        }
        self._upload_tables(tab, blob)

    def _load_packed(self, tab: dict[str, Any], wsh: bool, arch: str, prop: Any) -> None:
        """Load the kernels with the most cells per block that cost no local memory.

        The start comes from the registers of a one-cell compile. It then grows while the shared memory and the named
        barriers allow, and the compiler fits the cells in the registers without more local memory.
        """
        threads = 32 * self.w
        fit = (prop.shared_memory_per_block_optin - len(pack_small_tables(tab)[0])) // (
            8 * shared_doubles_per_cell(tab)
        )
        top = min(MAX_THREADS_PER_BLOCK // threads, MAX_NAMED_BARRIERS, fit)

        def load(cells_per_block: int) -> tuple[Any, list[Any]]:
            sources = kernel_sources(call_header(tab, cells_per_block, wsh, self.drift))
            return load_module(compile_cubin(SOLVER_KERNELS, sources, arch), KERNEL_ARGUMENTS, self._modules)

        def drop(module: Any) -> None:
            self._modules.remove(module)
            check_driver(driver().cuModuleUnload(module), "unload")

        one_module, one = load(1)
        registers = function_attribute(one[0], ATTR_REGISTERS)
        registers = -(-registers // REGISTER_ALLOCATION_UNIT) * REGISTER_ALLOCATION_UNIT
        local = function_attribute(one[0], ATTR_LOCAL_BYTES)
        self.cpb = max(1, min(top, prop.regs_per_multiprocessor // (threads * registers)))
        chosen_module, chosen = load(self.cpb) if self.cpb > 1 else (one_module, one)
        while self.cpb < top:
            module, larger = load(self.cpb + 1)
            if function_attribute(larger[0], ATTR_LOCAL_BYTES) > local:
                drop(module)
                break
            if chosen is not one:
                drop(chosen_module)
            chosen_module, chosen, self.cpb = module, larger, self.cpb + 1
        if chosen is not one:
            drop(one_module)
        self._fn, self._guard, self._rec = chosen

    def _upload_tables(self, tab: dict[str, Any], blob: bytes) -> None:
        """Copy the kernels' tables to the device, and allocate the scratch and the counts."""

        def to_device(values: Any, view: Any = None) -> torch.Tensor:
            return torch.tensor(values if view is None else np.asarray(values).view(view), device=self.dev)

        self._tables = {
            "perm": to_device(tab["perm"]),
            "blob": to_device(np.frombuffer(blob, np.int32).copy()),
            "rc": to_device(tab["rc"]),
            "rf": to_device(tab["rf"]),
            "nasa": to_device(tab["nasa"]),
            "sst": to_device(tab["sst"].ravel(), np.int32),
            "jt": to_device(tab["jt"], np.int64),
            "lu": to_device(tab["lu"], np.int64),
            "optr": to_device(tab["optr"]),
            "didx": to_device(tab["didx"], np.int16),
            "atoms": to_device(tab["atoms"]),
        }
        self._rounds = {"sround": ctypes.c_int(tab["sst"].shape[0]), "jround": ctypes.c_int(tab["jround"])}
        self._scratch = torch.empty(self.max_blocks * self.cpb * self.stride, dtype=torch.float64, device=self.dev)
        self._scratch_rec = torch.empty(self.rec_blocks * self.stride_rec, dtype=torch.float64, device=self.dev)
        self.count = torch.zeros(len(COUNT_SLOTS), dtype=torch.int32, device=self.dev)
        self.total = torch.zeros(len(COUNT_SLOTS), dtype=torch.int64, device=self.dev)

    def close(self) -> None:
        """Unload the kernels, after the device's work is done. The solver cannot be called afterwards."""
        unload_modules(self.dev, self._modules)

    def __enter__(self) -> Self:
        """Return the solver; leaving the context closes it."""
        return self

    def __exit__(self, *exception: object) -> None:
        """Close the solver."""
        self.close()

    def __del__(self) -> None:
        """Unload the kernels; a backstop for an object that was not closed."""
        if hasattr(self, "_modules"):  # a constructor that failed early has none
            unload_at_deletion(self.dev, self._modules)

    @on_device
    def launch(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        dt: torch.Tensor,
        h: torch.Tensor,
        rtol: float = DEFAULT_RTOL,
        atol: float = ATOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        J: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Launch the call, the guard and the recovery, with no host sync.

        Parameters
        ----------
        x : torch.Tensor
            Mole fractions [B, S], float64, in the network's species order.
        T, P, dt, h : torch.Tensor
            Temperature (K), pressure (dyn cm^-2), call length (s), starting step (s): [B], float64.
        rtol, atol : float
            Tolerances; atol is in mole fraction.
        max_steps : int
            Tries per call; the recovery has ``REC_STEPS`` times as many.
        J : torch.Tensor, optional
            Photolysis rates [B, branches] in s^-1 (``photo=True`` only); None means 0.

        Returns
        -------
        tuple of torch.Tensor
            x, h, done, accepted steps, and a per-cell flag: 0 (the call's result passed the guard), 1 (recomputed,
            passes) or 2 (recomputed, still fails). The counts go to ``self.count``.
        """
        if not self._modules:
            raise RuntimeError("this solver was closed; build a new one")
        B = check_inputs(self.dev, self.ns, x, T, P, dt, h, rtol, atol, max_steps)
        x, T, P, dt, h = (a.contiguous() for a in (x, T, P, dt, h))
        jx = self._photolysis_rates(J, B)
        xo, ho = torch.empty_like(x), h.clone()
        done, redo, lst = (torch.empty(B, dtype=torch.int32, device=self.dev) for _ in range(3))
        nok = torch.empty(B, dtype=torch.float64, device=self.dev)
        self.count.zero_()
        stream = torch.cuda.current_stream(self.dev).cuda_stream
        values = {
            **self._tables,
            **self._rounds,
            "x0": x,
            "Ta": T,
            "Pa": P,
            "dta": dt,
            "hin": h,
            "hio": ho,
            "xout": xo,
            "doneo": done,
            "noko": nok,
            "redo": redo,
            "list": lst,
            "count": self.count,
            "B": ctypes.c_int(B),
            "rtol": ctypes.c_double(rtol),
            "atol": ctypes.c_double(atol),
            "max_steps": ctypes.c_int(max_steps),
            "scratch": self._scratch,
            PHOTOLYSIS_ARGUMENT: jx,
        }
        grid = max(1, min(-(-B // self.cpb), self.max_blocks))
        launch_kernel(self._fn, grid, 32 * self.w * self.cpb, self.smem, stream, self._arguments("call", values))
        guard_grid = -(-B // GUARD_BLOCK_THREADS) or 1
        launch_kernel(self._guard, guard_grid, GUARD_BLOCK_THREADS, 0, stream, self._arguments("guard", values))
        values |= {"max_steps": ctypes.c_int(REC_STEPS * max_steps), "scratch": self._scratch_rec}
        grid = max(1, min(B, self.rec_blocks))
        launch_kernel(self._rec, grid, 32 * self.w, self.smem_rec, stream, self._arguments("recover", values))
        return xo, ho, done.bool(), nok, redo

    def _photolysis_rates(self, J: torch.Tensor | None, cells: int) -> torch.Tensor | None:
        """Return the call's J, zeros when it is not given; refuse a J of the wrong form."""
        if not self.nj:
            if J is not None:
                raise ValueError("this solver has no photolysis (built with photo=False)")
            return None
        if J is None:
            return torch.zeros(cells, self.nj, dtype=torch.float64, device=self.dev)
        if J.device == self.dev and J.dtype == torch.float64 and tuple(J.shape) == (cells, self.nj):
            return J.contiguous()
        raise ValueError(f"J [B, {self.nj}]: a float64 tensor on the device of x")

    def _arguments(self, kernel: str, values: dict[str, Any]) -> list[Any]:
        """Return a kernel's arguments, in its signature's order, from the values by parameter name."""
        names = KERNEL_ARGUMENTS[kernel]
        return [values[name] for name in names] + ([values[PHOTOLYSIS_ARGUMENT]] if self.nj else [])

    def __call__(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        dt: torch.Tensor,
        h: torch.Tensor,
        rtol: float = DEFAULT_RTOL,
        atol: float = ATOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        J: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the call: ``launch`` without the flag, and the counts added to ``self.total`` on the stream."""
        xo, ho, done, nok, _ = self.launch(x, T, P, dt, h, rtol, atol, max_steps, J)
        self.total += self.count
        return xo, ho, done, nok

    def export(self, folder: str | Path) -> None:
        """Write what a caller without Python needs to run this solver's kernels (cpp/maxichem.hpp loads it)."""
        write_export(self, folder)

    @property
    def fallback(self) -> dict[str, int]:
        """Return the last call's counts (one small copy from the device)."""
        return dict(zip(COUNTS, self.count[1:-1].tolist(), strict=True))

    def check(self) -> dict[str, int]:
        """Return the counts since construction (one small copy); raise if any cell was below T_MIN."""
        n = dict(zip(COUNTS, self.total[1:-1].tolist(), strict=True))
        if n["cold"]:
            raise ValueError(f"{n['cold']} cells below {T_MIN:g} K: outside the chemistry solver's domain")
        return n
