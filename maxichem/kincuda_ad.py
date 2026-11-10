"""Derivatives of one solver call in CUDA kernels: a tangent (JVP) and an adjoint (VJP).

The method and the two contracts, frozen and adaptive, are in docs/method.md.
"""

import ctypes
import operator
from typing import Any

import numpy as np
import torch

from ._ad_tables import ad_tables, layout, scratch_sizes, supported
from ._constants import (
    ARRH_BLOCK_THREADS,
    ATOL,
    CAP,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
    DMAX,
    GUARD_BLOCK_THREADS,
    PHOTOLYSIS_ARGUMENT,
    REC_STEPS,
)
from ._cuda import (
    blocks_per_multiprocessor,
    compile_cubin,
    launch_kernel,
    load_module,
    on_device,
    opt_in_shared_memory,
)
from ._kernel_header import derivative_header, kernel_sources
from ._solver import Solver, check_inputs, check_photolysis, check_tensor, shared_tables, to_device
from ._tables import pack_small_tables, tables_for_device
from .kinvulcan import VulcanNetwork

__all__ = ["ADSolver"]
MAX_WARPS_PER_BLOCK = 32  # the derivative kernels' queue and shared memory hold at most this many


def _pointer(a: torch.Tensor | None) -> Any:
    """Return a tensor argument, or a null pointer for an optional one left out."""
    return ctypes.c_void_p(0) if a is None else a


class ADSolver(Solver):
    """One solver call with its derivatives on one GPU.

    Each call returns the result of ``CudaSolver.launch``, bit for bit, and the derivative of that result. After each
    kernel the guard lists the cells the solver would recompute, and those are redone on the recovery's path with
    their derivatives. A cell that still fails has flag 2 and NaN derivatives. Calls run one at a time on torch's
    current stream. See docs/api.md.

    Parameters
    ----------
    net : VulcanNetwork
        At most 96 species and 2,047 reactions.
    device : torch.device or str
        A CUDA device; a bare "cuda" means the current one.
    cap : int
        Attempts stored per cell before the adjoint switches to checkpoints.
    drift : float, optional
        The guard's element-drift allowance, relative; ``DRIFT`` if None.
    photo : bool
        The network's photolysis branches enter the call, which then takes J.
    """

    def __init__(
        self,
        net: VulcanNetwork,
        device: torch.device | str,
        cap: int = CAP,
        drift: float | None = None,
        photo: bool = False,
    ) -> None:
        try:
            cap = operator.index(cap)
        except TypeError:
            raise ValueError(f"cap: an integer expected, got {cap!r}") from None
        if cap < 1:
            raise ValueError(f"cap: at least 1 stored attempt expected, got {cap}")
        super().__init__(net, device, drift, photo)
        self._setup(net, cap, photo)

    @on_device
    def _setup(self, net: VulcanNetwork, cap: int, photo: bool) -> None:
        """Build the kernels and their tables on the solver's device."""
        prop = torch.cuda.get_device_properties(self.dev)
        tab = tables_for_device(net, photo, prop.shared_memory_per_block_optin)  # the call kernel's warps per cell
        supported(tab)
        ext = ad_tables(net, tab)
        self.ns, self.nr, self.cap, self.nj, self.rcw = tab["ns"], tab["nr"], cap, tab["nj"], tab["rc"].shape[1]
        blob, _ = pack_small_tables(tab)
        self.wpb: dict[str, int] = {}
        self.smem: dict[str, int] = {}
        self.max_blocks: dict[str, int] = {}
        self._kernels: dict[str, ctypes.c_void_p] = {}
        self._sizes = scratch_sizes(tab, cap)
        for kind in ("jvp", "vjp"):
            _, perwa = layout(tab, ext, kind)
            wpb = min(MAX_WARPS_PER_BLOCK, (prop.shared_memory_per_block_optin - len(blob)) // (8 * perwa))
            if wpb < 1:
                raise RuntimeError(
                    f"the {kind} kernel needs {8 * perwa + len(blob)} B of shared memory per block; "
                    f"the device has {prop.shared_memory_per_block_optin}"
                )
            self.wpb[kind], self.smem[kind] = wpb, 8 * wpb * perwa + len(blob)
            sources = kernel_sources(derivative_header(tab, ext, kind, wpb, cap, self.drift))
            cubin = compile_cubin(f"{kind}.cu", sources, f"{prop.major}{prop.minor}")
            names = (kind, f"{kind}_rec", "arrh" if kind == "vjp" else "guard")
            _, functions = load_module(cubin, names, self._modules)
            self._kernels |= dict(zip(names, functions, strict=True))
            blocks = blocks_per_multiprocessor(self._kernels[kind], 32 * wpb, self.smem[kind])
            opt_in_shared_memory(self._kernels[f"{kind}_rec"], self.smem[kind])
            self.max_blocks[kind] = prop.multi_processor_count * blocks
        self._upload_tables(tab, ext, blob)

    def _upload_tables(self, tab: dict[str, Any], ext: dict[str, Any], blob: bytes) -> None:
        """Copy the kernels' tables to the device, and allocate the scratch."""

        def doubles(size: int) -> torch.Tensor:
            return torch.empty(size, dtype=torch.float64, device=self.dev)

        self._nw = {k: self.max_blocks[k] * self.wpb[k] for k in ("jvp", "vjp")}
        rates = doubles(self._nw["vjp"] * self._sizes["VJP_STRIDE"])
        self._scratch = {
            "vjp": rates,
            "jvp": rates,  # grows with the tangent directions
            "rec": doubles(self._nw["vjp"] * 2 * self.cap * self._sizes["REC"]),
            "rec_recovery": doubles(self._nw["vjp"] * 2 * self._sizes["CAPR"] * self._sizes["REC"]),
            "wscr": doubles(max(self._nw.values()) * self.ns**2),
        }
        self._tables = {
            **shared_tables(tab, blob, self.dev),
            "gt": to_device(ext["gt"].ravel(), self.dev, np.int64),
            "xg": to_device(ext["xg"], self.dev),
        }
        self._rounds = {
            "sround": ctypes.c_int(tab["sst"].shape[0]),
            "jround": ctypes.c_int(tab["jround"]),
            "ground": ctypes.c_int(ext["gt"].shape[0]),
        }
        self.flag: torch.Tensor | None = None

    @property
    def rc0(self) -> torch.Tensor:
        """Return the network's rate table [R, W], which a call uses unless given another."""
        self._require_open()
        return self._tables["rc"]

    def _prep(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        dt: torch.Tensor,
        h: torch.Tensor,
        lm: torch.Tensor | None,
        rc: torch.Tensor | None,
        J: torch.Tensor | None,
        max_steps: int,
        rtol: float,
        atol: float,
    ) -> tuple[int, list[torch.Tensor], Any, torch.Tensor, Any]:
        """Check a call's inputs and return B, the contiguous (x, T, P, dt, h), lm, rc and J."""
        self._require_open()
        B = check_inputs(self.dev, self.ns, x, T, P, dt, h, rtol, atol, max_steps)
        limit = self._sizes["CAPR"] ** 2 // REC_STEPS  # what the recovery's checkpoints reach, below the call's CAP^2
        if max_steps > limit:
            raise ValueError(f"max_steps: at most {limit} at cap {self.cap} expected, got {max_steps}")
        if lm is not None:
            check_tensor("lm", lm, (B, self.nr), self.dev)
            lm = lm.contiguous()
        rc = self.rc0 if rc is None else rc
        check_tensor("rc", rc, (self.nr, self.rcw), self.dev)
        J = check_photolysis(J, B, self.nj, self.dev)
        return B, [a.contiguous() for a in (x, T, P, dt, h)], lm, rc.contiguous(), J

    def _run_guard(self, values: dict[str, Any], stream: int) -> dict[str, torch.Tensor]:
        """Run the call kernel's guard on a kernel's result: the redo and list of the cells for the recovery's path."""
        cells = values["x0"].shape[0]
        flags = {name: torch.empty(cells, dtype=torch.int32, device=self.dev) for name in ("redo", "list")}
        arguments = self._arguments("guard", {**values, **flags})
        launch_kernel(
            self._kernels["guard"], -(-cells // GUARD_BLOCK_THREADS) or 1, GUARD_BLOCK_THREADS, 0, stream, arguments
        )
        return flags

    def _call_values(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        dt: torch.Tensor,
        h: torch.Tensor,
        lm: torch.Tensor | None,
        rc: torch.Tensor,
        J: torch.Tensor | None,
        rtol: float,
        atol: float,
        max_steps: int,
        adapt: bool,
    ) -> dict[str, Any]:
        """Return the arguments both derivative kernels take, by parameter name."""
        return {
            **self._tables,
            **self._rounds,
            "wscr": self._scratch["wscr"],
            "x0": x,
            "Ta": T,
            "Pa": P,
            "dta": dt,
            "hin": h,
            "lm": _pointer(lm),
            "rc": rc,
            "count": self.count,
            "B": ctypes.c_int(x.shape[0]),
            "rtol": ctypes.c_double(rtol),
            "atol": ctypes.c_double(atol),
            "max_steps": ctypes.c_int(max_steps),
            "adapt": ctypes.c_int(int(adapt)),
            PHOTOLYSIS_ARGUMENT: _pointer(J),
        }

    def _empty_cotangents(self, x: torch.Tensor, T: torch.Tensor) -> dict[str, torch.Tensor]:
        """Allocate the cotangents a vjp call fills."""
        cells = x.shape[0]
        cotangents = {
            "x": torch.empty_like(x),
            "T": torch.empty_like(T),
            "P": torch.empty_like(T),
            "dt": torch.empty_like(T),
            "h": torch.empty_like(T),
            "lm": torch.empty(cells, self.nr, dtype=torch.float64, device=self.dev),
            "rc": torch.empty(self.nr, 9, dtype=torch.float64, device=self.dev),
        }
        if self.nj:
            cotangents["J"] = torch.empty(cells, self.nj, dtype=torch.float64, device=self.dev)
        return cotangents

    def _grow_tangent_scratch(self, directions: int) -> None:
        """Make the tangent kernels' scratch large enough for this many directions."""
        need = self._nw["jvp"] * (self._sizes["JVP_STRIDE"] + directions * self._sizes["JVP_DIRECTION_STRIDE"])
        if self._scratch["jvp"].numel() < need:
            self._scratch["jvp"] = torch.empty(need, dtype=torch.float64, device=self.dev)

    def _check_tangents(self, tangents: dict[str, Any], cells: int) -> int:
        """Refuse a tangent of the wrong form, naming it, and return their directions (1 without a leading axis)."""
        dx = tangents["dx"]
        if dx.dim() not in (2, 3):
            raise ValueError(f"dx: [B, S] or [D, B, S] expected, got {list(dx.shape)}")
        one = dx.dim() == 2
        directions = 1 if one else dx.shape[0]
        if not 1 <= directions <= DMAX:
            raise ValueError(f"dx: from 1 to {DMAX} directions on the leading axis expected, got {list(dx.shape)}")
        if tangents["dJ"] is not None and not self.nj:
            raise ValueError("dJ given, but this solver has no photolysis (built with photo=False)")
        lead = () if one else (directions,)
        shapes = {
            "dx": (*lead, cells, self.ns),
            "dT": (*lead, cells),
            "dP": (*lead, cells),
            "ddt": (*lead, cells),
            "dh": (*lead, cells),
            "dlm": (*lead, cells, self.nr),
            "drc": (*lead, self.nr, 9),
            "dJ": (*lead, cells, self.nj),
        }
        for name, a in tangents.items():
            if a is not None:
                check_tensor(name, a, shapes[name], self.dev)
        return directions

    @on_device
    def jvp(  # noqa: PLR0913  launch's inputs, lm, rc, adapt and a tangent for each input
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        dt: torch.Tensor,
        h: torch.Tensor,
        dx: torch.Tensor,
        dT: torch.Tensor,
        dP: torch.Tensor,
        ddt: torch.Tensor,
        dh: torch.Tensor,
        lm: torch.Tensor | None = None,
        dlm: torch.Tensor | None = None,
        rc: torch.Tensor | None = None,
        drc: torch.Tensor | None = None,
        adapt: bool = True,
        rtol: float = DEFAULT_RTOL,
        atol: float = ATOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        J: torch.Tensor | None = None,
        dJ: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the call and its tangent.

        Parameters
        ----------
        x, T, P, dt, h : torch.Tensor
            As ``CudaSolver.launch``: mole fractions [B, S], temperature (K), pressure (dyn cm^-2), call length (s)
            and starting step (s) [B]; float64.
        dx, dT, dP, ddt, dh : torch.Tensor
            Their tangents, in the same units and shapes, or with a leading axis of D <= ``DMAX`` directions on
            every tangent, which then share one call.
        lm : torch.Tensor, optional
            Log rate multipliers [B, R]; none applied if None.
        dlm : torch.Tensor, optional
            Their tangent [B, R] (or [D, B, R]).
        rc : torch.Tensor, optional
            The rate table [R, W]; ``rc0`` if None.
        drc : torch.Tensor, optional
            The tangent of its nine Arrhenius columns [R, 9] (or [D, R, 9]).
        adapt : bool
            Differentiate the step control too (docs/method.md, "Derivatives"); False holds the accepted steps fixed.
        rtol, atol, max_steps
            As ``CudaSolver.launch``.
        J : torch.Tensor, optional
            Photolysis rates [B, branches], s^-1, as ``CudaSolver.launch``.
        dJ : torch.Tensor, optional
            Their tangent [B, branches] (or [D, B, branches]), s^-1.

        Returns
        -------
        tuple of torch.Tensor
            x, h, done, accepted steps, dx, dh; the last two have the leading axis if the inputs do.

        Raises
        ------
        ValueError
            As ``CudaSolver.launch``, if a cell was bad input or below ``T_MIN``.
        """
        B, (x, T, P, dt, h), lm, rc, J = self._prep(x, T, P, dt, h, lm, rc, J, max_steps, rtol, atol)
        tangents = {"dx": dx, "dT": dT, "dP": dP, "ddt": ddt, "dh": dh, "dlm": dlm, "drc": drc, "dJ": dJ}
        D = self._check_tangents(tangents, B)
        one = dx.dim() == 2
        tan = [a.contiguous() for a in (dx, dT, dP, ddt, dh)]
        dlm, drc, dJ = (None if a is None else a.contiguous() for a in (dlm, drc, dJ))
        xo, dxo = torch.empty_like(x), torch.empty((D, B, self.ns), dtype=torch.float64, device=self.dev)
        ho, nok = (torch.empty(B, dtype=torch.float64, device=self.dev) for _ in range(2))
        dho = torch.empty((D, B), dtype=torch.float64, device=self.dev)
        done = torch.empty(B, dtype=torch.int32, device=self.dev)
        self._grow_tangent_scratch(D)
        nb = max(1, min(-(-B // self.wpb["jvp"]), self.max_blocks["jvp"]))
        self.count.zero_()
        stream = torch.cuda.current_stream(self.dev).cuda_stream

        values = {
            **self._call_values(x, T, P, dt, h, lm, rc, J, rtol, atol, max_steps, adapt),
            **dict(zip(("dx0", "dTa", "dPa", "ddta", "dhin"), tan, strict=True)),
            "dlm": _pointer(dlm),
            "drc": _pointer(drc),
            "D": ctypes.c_int(D),
            "xout": xo,
            "hout": ho,
            "doneo": done,
            "noko": nok,
            "dxout": dxo,
            "dhout": dho,
            "scratch": self._scratch["jvp"],
            "dJin": _pointer(dJ),
        }
        block = 32 * self.wpb["jvp"]
        launch_kernel(self._kernels["jvp"], nb, block, self.smem["jvp"], stream, self._arguments("jvp", values))
        flags = self._run_guard(values, stream)
        values |= {**flags, "max_steps": ctypes.c_int(REC_STEPS * max_steps)}
        launch_kernel(self._kernels["jvp_rec"], nb, block, self.smem["jvp"], stream, self._arguments("jvp_rec", values))
        self.total += self.count
        self.flag = flags["redo"]
        self._refuse(x, T, P, dt, J, done, self.flag)
        return xo, ho, done.bool(), nok, dxo[0] if one else dxo, dho[0] if one else dho

    @on_device
    def vjp(  # noqa: PLR0913  launch's inputs, lm, rc, adapt and the outputs' cotangents, as jvp
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        dt: torch.Tensor,
        h: torch.Tensor,
        gx: torch.Tensor,
        gh: torch.Tensor | None = None,
        lm: torch.Tensor | None = None,
        rc: torch.Tensor | None = None,
        adapt: bool = True,
        rtol: float = DEFAULT_RTOL,
        atol: float = ATOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        J: torch.Tensor | None = None,
    ) -> tuple[Any, ...]:
        """Run the call and the cotangents of its inputs.

        Parameters
        ----------
        x, T, P, dt, h : torch.Tensor
            As ``CudaSolver.launch``: mole fractions [B, S], temperature (K), pressure (dyn cm^-2), call length (s)
            and starting step (s) [B]; float64.
        gx : torch.Tensor
            The cotangent of the output x [B, S].
        gh : torch.Tensor, optional
            The cotangent of the output h [B].
        lm : torch.Tensor, optional
            Log rate multipliers [B, R]; none applied if None.
        rc : torch.Tensor, optional
            The rate table [R, W]; ``rc0`` if None.
        adapt : bool
            Differentiate the step control too (docs/method.md, "Derivatives"); False holds the accepted steps fixed.
        rtol, atol, max_steps
            As ``CudaSolver.launch``.
        J : torch.Tensor, optional
            Photolysis rates [B, branches], s^-1, as ``CudaSolver.launch``.

        Returns
        -------
        tuple
            x, h, done, accepted steps, and a dict of the cotangents x, T, P, dt, h, lm [B, R], rc [R, 9] and, with
            ``photo=True``, J [B, branches]. A cell with flag 2 has NaN in its own cotangents and is left out of the
            sum that rc is; ``fallback["still_failed"]`` counts the cells left out.

        Raises
        ------
        ValueError
            As ``CudaSolver.launch``, if a cell was bad input or below ``T_MIN``.
        """
        B, (x, T, P, dt, h), lm, rc, J = self._prep(x, T, P, dt, h, lm, rc, J, max_steps, rtol, atol)
        check_tensor("gx", gx, (B, self.ns), self.dev)
        gx = gx.contiguous()
        if gh is not None:
            check_tensor("gh", gh, (B,), self.dev)
            gh = gh.contiguous()
        xo = torch.empty_like(x)
        ho, nok = (torch.empty(B, dtype=torch.float64, device=self.dev) for _ in range(2))
        done = torch.empty(B, dtype=torch.int32, device=self.dev)
        g = self._empty_cotangents(x, T)
        g_ln_a = torch.empty(B, self.nr, dtype=torch.float64, device=self.dev)
        nb = max(1, min(-(-B // self.wpb["vjp"]), self.max_blocks["vjp"]))
        self.count.zero_()
        stream = torch.cuda.current_stream(self.dev).cuda_stream

        values = {
            **self._call_values(x, T, P, dt, h, lm, rc, J, rtol, atol, max_steps, adapt),
            "gx": gx,
            "gh": _pointer(gh),
            "xout": xo,
            "hout": ho,
            "doneo": done,
            "noko": nok,
            "gx0": g["x"],
            "gT": g["T"],
            "gP": g["P"],
            "gdt": g["dt"],
            "ghin": g["h"],
            "glm": g["lm"],
            "grc": g["rc"],
            "scratch": self._scratch["vjp"],
            "rec": self._scratch["rec"],
            "gJ": _pointer(g.get("J")),
            "gA": g_ln_a,
        }
        block = 32 * self.wpb["vjp"]
        launch_kernel(self._kernels["vjp"], nb, block, self.smem["vjp"], stream, self._arguments("vjp", values))
        flags = self._run_guard(values, stream)
        values |= {**flags, "max_steps": ctypes.c_int(REC_STEPS * max_steps), "rec": self._scratch["rec_recovery"]}
        launch_kernel(self._kernels["vjp_rec"], nb, block, self.smem["vjp"], stream, self._arguments("vjp_rec", values))
        launch_kernel(self._kernels["arrh"], self.nr, ARRH_BLOCK_THREADS, 0, stream, self._arguments("arrh", values))
        self.total += self.count
        self.flag = flags["redo"]
        self._refuse(x, T, P, dt, J, done, self.flag)
        return xo, ho, done.bool(), nok, g
