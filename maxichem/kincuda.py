"""Batched GPU solver: one call per cell of a VULCAN network, in CUDA kernels.

The method is in docs/method.md, the kernels' layout in docs/architecture.md.
"""

import ctypes
from pathlib import Path
from typing import Any, NamedTuple

import torch

from ._constants import (
    ATOL,
    COUNTS,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
    DOUBLE_BYTES,
    DRIFT,
    GUARD_BLOCK_THREADS,
    PHOTOLYSIS_ARGUMENT,
    REC_STEPS,
    SOLVER_KERNEL_NAMES,
    T_MAX,
    T_MIN,
    WARP_LANES,
    X_SUM_TOL,
)
from ._cuda import (
    ATTR_LOCAL_BYTES,
    ATTR_REGISTERS,
    SOLVER_KERNELS,
    blocks_per_multiprocessor,
    check_driver,
    compile_cubin,
    driver,
    function_attribute,
    launch_kernel,
    load_module,
    on_device,
)
from ._export import write_export
from ._kernel_header import call_header, kernel_sources
from ._solver import Solver, check_inputs, check_photolysis, shared_tables
from ._tables import (
    call_layout,
    cells_per_block,
    pack_small_tables,
    rates_fit_in_block,
    recovery_layout,
    scratch_layout,
    tables_for_device,
)
from .kinvulcan import VulcanNetwork

__all__ = [
    "ATOL",
    "COUNTS",
    "DRIFT",
    "REC_STEPS",
    "T_MAX",
    "T_MIN",
    "X_SUM_TOL",
    "CallResult",
    "CudaSolver",
    "LaunchResult",
]
MAX_THREADS_PER_BLOCK = 1024  # CUDA's limit
MAX_NAMED_BARRIERS = 15  # per block: the hardware's named barriers, one per cell (bar.sync in common.cuh)
REGISTER_ALLOCATION_UNIT = 8  # registers per thread are allocated in this unit


class CallResult(NamedTuple):
    """What a call returns, per cell.

    Attributes
    ----------
    x : torch.Tensor [B, S]
        Mole fractions at the end of the call, float64, in the network's species order.
    h : torch.Tensor [B]
        The step size to start the cell's next call with, s, float64.
    done : torch.Tensor [B]
        bool: the call finished and its result passed the guard.
    steps : torch.Tensor [B]
        Accepted steps, float64.
    """

    x: torch.Tensor
    h: torch.Tensor
    done: torch.Tensor
    steps: torch.Tensor


class LaunchResult(NamedTuple):
    """What ``CudaSolver.launch`` returns, per cell: ``CallResult``'s fields and the flag.

    Attributes
    ----------
    x, h, done, steps : torch.Tensor
        As ``CallResult``.
    flag : torch.Tensor [B]
        int32: 0 the call's result passed the guard, 1 recomputed and passes, 2 recomputed and still fails.
    """

    x: torch.Tensor
    h: torch.Tensor
    done: torch.Tensor
    steps: torch.Tensor
    flag: torch.Tensor


class CudaSolver(Solver):
    """Batched chemistry solver on one GPU.

    Calling it advances each cell by dt; it reads the call's counts back and raises if it refused a cell. Calls run
    one at a time on torch's current stream: the
    queues, the counts and the scratch are shared. See docs/api.md for its members and docs/method.md for the method.

    Parameters
    ----------
    net : VulcanNetwork
        Its species order is the order of x.
    device : torch.device or str
        A CUDA device; a bare "cuda" means the current one.
    drift : float, optional
        The guard's element-drift allowance, relative; ``DRIFT`` if None. Keyword only, as photo.
    photo : bool
        The network's photolysis branches enter the call, which then takes J.
    """

    def __init__(
        self, net: VulcanNetwork, device: torch.device | str, *, drift: float | None = None, photo: bool = False
    ) -> None:
        super().__init__(net, device, drift, photo)
        self._setup(net, photo)

    @on_device
    def _setup(self, net: VulcanNetwork, photo: bool) -> None:
        """Build the kernels and their tables on the solver's device."""
        prop = torch.cuda.get_device_properties(self.dev)
        tab = tables_for_device(net, photo, prop.shared_memory_per_block_optin)
        self.ns, self.nr, self.w, self.nj = tab["ns"], tab["nr"], tab["warps"], tab["nj"]
        threads = WARP_LANES * self.w
        blob, _ = pack_small_tables(tab)
        # the recover kernel's W in shared memory if it fits
        wsh = DOUBLE_BYTES * recovery_layout(tab, True)[1] + len(blob) <= prop.shared_memory_per_block_optin
        arch = f"{prop.major}{prop.minor}"
        self._load_packed(tab, wsh, arch, prop)
        self._build = (tab, wsh, arch)  # what export() recompiles
        self._names = (list(net.species), list(net.elements))  # export() writes them
        self.wpb = self.w * self.cpb  # warps per block
        self.smem = DOUBLE_BYTES * self.cpb * call_layout(tab)[1] + len(blob)
        self.smem_rec = DOUBLE_BYTES * recovery_layout(tab, wsh)[1] + len(blob)
        blocks = blocks_per_multiprocessor(self._call_kernel, threads * self.cpb, self.smem)
        rec_blocks = blocks_per_multiprocessor(self._recover_kernel, threads, self.smem_rec)
        self.max_blocks, self.rec_blocks = prop.multi_processor_count * blocks, prop.multi_processor_count * rec_blocks
        self.stride = scratch_layout(tab, False)[1]  # the call kernel's cell slot and the recover kernel's block
        self.stride_rec = scratch_layout(tab, not wsh)[1]
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
            "regs": function_attribute(self._call_kernel, ATTR_REGISTERS),
            "spill_bytes": function_attribute(self._call_kernel, ATTR_LOCAL_BYTES),
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
        threads = WARP_LANES * self.w
        fit = cells_per_block(tab, prop.shared_memory_per_block_optin)
        top = min(MAX_THREADS_PER_BLOCK // threads, MAX_NAMED_BARRIERS, fit)

        def load(cells_per_block: int) -> tuple[ctypes.c_void_p, list[ctypes.c_void_p]]:
            sources = kernel_sources(call_header(tab, cells_per_block, wsh, self.drift))
            return load_module(compile_cubin(SOLVER_KERNELS, sources, arch), SOLVER_KERNEL_NAMES, self._modules)

        def drop(module: ctypes.c_void_p) -> None:
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
        self._call_kernel, self._guard_kernel, self._recover_kernel = chosen

    def _upload_tables(self, tab: dict[str, Any], blob: bytes) -> None:
        """Copy the kernels' tables to the device, and allocate the scratch."""
        self._tables = shared_tables(tab, blob, self.dev)
        self._rounds = {"sround": ctypes.c_int(tab["sst"].shape[0]), "jround": ctypes.c_int(tab["jround"])}
        self._scratch = {
            "call": torch.empty(self.max_blocks * self.cpb * self.stride, dtype=torch.float64, device=self.dev),
            "recover": torch.empty(self.rec_blocks * self.stride_rec, dtype=torch.float64, device=self.dev),
        }

    @on_device
    def launch(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        dt: torch.Tensor,
        h: torch.Tensor,
        *,
        rtol: float = DEFAULT_RTOL,
        atol: float = ATOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        J: torch.Tensor | None = None,
    ) -> LaunchResult:
        """Launch the call, the guard and the recovery; read the counts; refuse bad input and T out of range.

        The one host sync is the read of the counts, so the call can raise when it refused a cell.

        Parameters
        ----------
        x : torch.Tensor
            Mole fractions [B, S], float64, in the network's species order.
        T, P, dt, h : torch.Tensor
            Temperature (K), pressure (dyn cm^-2), call length (s), starting step (s): [B], float64.
        rtol, atol : float
            Tolerances, keyword only; atol is in mole fraction.
        max_steps : int
            Tries per call, keyword only; the recovery has ``REC_STEPS`` times as many.
        J : torch.Tensor, optional
            Photolysis rates [B, branches] in s^-1, keyword only: required with ``photo=True``, refused without it.
            Dark cells pass zeros.

        Returns
        -------
        LaunchResult
            x, h, done, accepted steps and the per-cell flag, which ``self.flag`` also holds. The counts go to
            ``self.count``, and are added to ``self.total``.

        Raises
        ------
        ValueError
            If a cell was bad input or outside [``T_MIN``, ``T_MAX``] (docs/api.md): the message gives the counts,
            the limits and the first such cell. The counts stay readable in ``fallback``; ``self.flag`` is None, as
            after any call that raises, since the guard gives a refused cell flag 0.
        """
        self.flag = None  # set again only when the call returns
        self._require_open()
        B = check_inputs(self.dev, self.ns, x, T, P, dt, h, rtol, atol, max_steps)
        x, T, P, dt, h = (a.contiguous() for a in (x, T, P, dt, h))
        jx = check_photolysis(J, B, self.nj, self.dev)
        xo, ho = torch.empty_like(x), h.clone()
        done, redo, lst, status = (torch.empty(B, dtype=torch.int32, device=self.dev) for _ in range(4))
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
            "status": status,
            "redo": redo,
            "list": lst,
            "count": self.count,
            "B": ctypes.c_int(B),
            "rtol": ctypes.c_double(rtol),
            "atol": ctypes.c_double(atol),
            "max_steps": ctypes.c_int(max_steps),
            "scratch": self._scratch["call"],
            PHOTOLYSIS_ARGUMENT: jx,
        }
        grid = max(1, min(-(-B // self.cpb), self.max_blocks))
        launch_kernel(
            self._call_kernel, grid, WARP_LANES * self.w * self.cpb, self.smem, stream, self._arguments("call", values)
        )
        guard_grid = -(-B // GUARD_BLOCK_THREADS) or 1
        launch_kernel(self._guard_kernel, guard_grid, GUARD_BLOCK_THREADS, 0, stream, self._arguments("guard", values))
        values |= {"max_steps": ctypes.c_int(REC_STEPS * max_steps), "scratch": self._scratch["recover"]}
        grid = max(1, min(B, self.rec_blocks))
        arguments = self._arguments("recover", values)
        launch_kernel(self._recover_kernel, grid, WARP_LANES * self.w, self.smem_rec, stream, arguments)
        self.total += self.count
        self._refuse(x, T, P, dt, jx, status)
        self.flag = redo
        return LaunchResult(xo, ho, done.bool(), nok, redo)

    def __call__(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        dt: torch.Tensor,
        h: torch.Tensor,
        *,
        rtol: float = DEFAULT_RTOL,
        atol: float = ATOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        J: torch.Tensor | None = None,
    ) -> CallResult:
        """Advance each cell by dt: ``launch`` without the flag in its result.

        Parameters
        ----------
        x : torch.Tensor
            Mole fractions [B, S], float64, in the network's species order.
        T, P, dt, h : torch.Tensor
            Temperature (K), pressure (dyn cm^-2), call length (s), starting step (s): [B], float64.
        rtol, atol : float
            Tolerances, keyword only; atol is in mole fraction.
        max_steps : int
            Tries per call, keyword only; the recovery has ``REC_STEPS`` times as many.
        J : torch.Tensor, optional
            Photolysis rates [B, branches] in s^-1, keyword only: required with ``photo=True``, refused without it.

        Returns
        -------
        CallResult
            x [B, S], h [B] (s), done [B] (bool) and accepted steps [B]; ``self.flag`` holds the per-cell flag.

        Raises
        ------
        ValueError
            As ``launch``, which leaves ``self.flag`` None.
        """
        r = self.launch(x, T, P, dt, h, rtol=rtol, atol=atol, max_steps=max_steps, J=J)
        return CallResult(r.x, r.h, r.done, r.steps)

    def export(self, folder: str | Path) -> None:
        """Write what a caller without Python needs to run this solver's kernels.

        Parameters
        ----------
        folder : str or Path
            Made if missing; it gets the kernels' source, their cubin for this device, their tables and ``meta.txt``,
            which ``cpp/maxichem.hpp`` loads. A cubin is for one GPU architecture: export on the machine that runs it.
        """
        self._require_open()
        write_export(self, folder)
