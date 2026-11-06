"""Derivatives of one solver call in CUDA kernels: a tangent (JVP) and an adjoint (VJP).

Public names: ``ADSolver``, and the constants ``CAP``, ``DMAX``. The method and the two contracts, frozen and
adaptive, are in docs/method.md.
"""

import ctypes
from typing import Any, Self

import numpy as np
import torch

from ._ad_tables import ad_tables, layout, supported
from ._constants import (
    ATOL,
    CAP,
    COUNT_SLOTS,
    COUNTS,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
    DMAX,
    GUARD_BLOCK_THREADS,
    KERNEL_ARGUMENTS,
    PHOTOLYSIS_ARGUMENT,
    REC_STEPS,
    T_MIN,
)
from ._cuda import (
    blocks_per_multiprocessor,
    compile_cubin,
    device_of,
    launch_kernel,
    load_module,
    on_device,
    opt_in_shared_memory,
    unload_at_deletion,
    unload_modules,
)
from ._kernel_header import derivative_header, kernel_sources
from ._tables import pack_small_tables, tables_for_device
from .kincuda import check_inputs
from .kinvulcan import VulcanNetwork

__all__ = ["CAP", "DMAX", "ADSolver"]
MAX_WARPS_PER_BLOCK = 32  # the derivative kernels' queue and shared memory hold at most this many
ARRH_BLOCK_THREADS = 256  # the arrh kernel's block: its reduction array in vjp.cuh has this size


class ADSolver:
    """One solver call with its derivatives on one GPU.

    Each call returns the result of ``CudaSolver.launch``, bit for bit, and the derivative of that result. After each
    kernel the guard lists the cells the solver would recompute, and those are redone on the recovery's path with
    their derivatives. A cell that still fails has flag 2 and NaN derivatives. Calls run one at a time on torch's
    current stream. See docs/api.md.

    Parameters
    ----------
    net : VulcanNetwork
        The network; at most 96 species and 2,047 reactions.
    device : torch.device or str
        The CUDA device.
    cap : int
        Attempts stored per cell before the adjoint switches to checkpoints.
    drift : float, optional
        The element-drift allowance, relative; the default is ``DRIFT``.
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
        if photo and (not net.photo or net.photo_missing):
            raise ValueError(
                "photo=True needs a network with photolysis branches whose species are all in its species "
                f"list (branches {len(net.photo)}, missing species {net.photo_missing})"
            )
        self.dev = device_of(device)
        self._modules: list[Any] = []  # every module loaded, also when the setup fails part of the way
        self._setup(net, cap, drift, photo)

    @on_device
    def _setup(self, net: VulcanNetwork, cap: int, drift: float | None, photo: bool) -> None:
        """Build the kernels and their tables on the solver's device."""
        torch.zeros(1, device=self.dev)
        prop = torch.cuda.get_device_properties(self.dev)
        tab = tables_for_device(net, photo, prop.shared_memory_per_block_optin)  # the call kernel's warps per cell
        supported(tab)
        ext = ad_tables(net, tab)
        self.ns, self.nr, self.cap, self.nj, self.rcw = tab["ns"], tab["nr"], cap, tab["nj"], tab["rc"].shape[1]
        blob, _ = pack_small_tables(tab)
        self.wpb: dict[str, int] = {}
        self.smem: dict[str, int] = {}
        self.max_blocks: dict[str, int] = {}
        self._fn: dict[str, Any] = {}
        self.capr = 2 * cap  # CAPR of the source: the recovery's stored attempts
        for kind in ("jvp", "vjp"):
            _, perwa = layout(tab, ext, kind)
            wpb = min(MAX_WARPS_PER_BLOCK, (prop.shared_memory_per_block_optin - len(blob)) // (8 * perwa))
            if wpb < 1:
                raise RuntimeError(f"the {kind} kernel's work arrays do not fit in shared memory")
            self.wpb[kind], self.smem[kind] = wpb, 8 * wpb * perwa + len(blob)
            sources = kernel_sources(derivative_header(tab, ext, kind, wpb, cap, drift))
            cubin = compile_cubin(f"{kind}.cu", sources, f"{prop.major}{prop.minor}")
            names = (kind, f"{kind}_rec", "arrh" if kind == "vjp" else "guard")
            _, functions = load_module(cubin, names, self._modules)
            self._fn |= dict(zip(names, functions, strict=True))
            blocks = blocks_per_multiprocessor(self._fn[kind], 32 * wpb, self.smem[kind])
            opt_in_shared_memory(self._fn[f"{kind}_rec"], self.smem[kind])
            self.max_blocks[kind] = prop.multi_processor_count * blocks
        self._upload_tables(tab, ext, blob)

    def _upload_tables(self, tab: dict[str, Any], ext: dict[str, Any], blob: bytes) -> None:
        """Copy the kernels' tables to the device, and allocate the scratch and the counts."""

        def to_device(values: Any, view: Any = None) -> torch.Tensor:
            return torch.tensor(values if view is None else np.asarray(values).view(view), device=self.dev)

        def doubles(size: int) -> torch.Tensor:
            return torch.empty(size, dtype=torch.float64, device=self.dev)

        self.rc0 = to_device(tab["rc"])
        self._nw = {k: self.max_blocks[k] * self.wpb[k] for k in ("jvp", "vjp")}
        self._scratch = doubles(self._nw["vjp"] * (14 * self.nr + 2))
        self._jscr = self._scratch
        self._rec = doubles(self._nw["vjp"] * 2 * self.cap * (self.ns + 6))
        self._rrec = doubles(self._nw["vjp"] * 2 * self.capr * (self.ns + 6))
        self._tables = {
            "perm": to_device(tab["perm"]),
            "blob": to_device(np.frombuffer(blob, np.int32).copy()),
            "rf": to_device(tab["rf"]),
            "nasa": to_device(tab["nasa"]),
            "sst": to_device(tab["sst"].ravel(), np.int32),
            "jt": to_device(tab["jt"].ravel(), np.int64),
            "lu": to_device(tab["lu"], np.int64),
            "optr": to_device(tab["optr"]),
            "gt": to_device(ext["gt"].ravel(), np.int64),
            "xg": to_device(ext["xg"]),
            "sround": ctypes.c_int(tab["sst"].shape[0]),
            "jround": ctypes.c_int(tab["jround"]),
            "ground": ctypes.c_int(ext["gt"].shape[0]),
            "wscr": doubles(max(self._nw.values()) * self.ns**2),
            "didx": to_device(tab["didx"], np.int16),
            "atoms": to_device(tab["atoms"]),
        }
        self.count = torch.zeros(len(COUNT_SLOTS), dtype=torch.int32, device=self.dev)
        self.total = torch.zeros(len(COUNT_SLOTS), dtype=torch.int64, device=self.dev)
        self.flag: torch.Tensor | None = None

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
        if not self._modules:
            raise RuntimeError("this solver was closed; build a new one")
        B = check_inputs(self.dev, self.ns, x, T, P, dt, h, rtol, atol, max_steps)
        if max_steps > self.cap * self.cap or REC_STEPS * max_steps > self.capr**2:
            raise ValueError(f"max_steps above {self.cap**2} (CAP^2) or the recovery's tries above {self.capr**2}")
        lm = None if lm is None else lm.contiguous()
        rc = self.rc0 if rc is None else rc.contiguous()
        if (
            (lm is not None and tuple(lm.shape) != (B, self.nr))
            or tuple(rc.shape) != (self.nr, self.rcw)
            or not all(a.dtype == torch.float64 and a.device == x.device for a in (lm, rc) if a is not None)
        ):
            raise ValueError(f"lm [B, {self.nr}], rc [{self.nr}, {self.rcw}]: float64 on the device of x")
        if J is not None and not self.nj:
            raise ValueError("J given, but the solver was built without photolysis (photo=False)")
        if self.nj:
            J = torch.zeros(B, self.nj, dtype=torch.float64, device=self.dev) if J is None else J.contiguous()
            if J.dtype != torch.float64 or J.device != x.device or tuple(J.shape) != (B, self.nj):
                raise ValueError(f"J [B, {self.nj}]: float64 on the device of x")
        return B, [a.contiguous() for a in (x, T, P, dt, h)], lm, rc, J

    def _ptr(self, a: torch.Tensor | None) -> Any:
        """Return a tensor argument, or a null pointer for an optional one left out."""
        return ctypes.c_void_p(0) if a is None else a

    def _guard(self, values: dict[str, Any], J: torch.Tensor | None, stream: Any) -> dict[str, torch.Tensor]:
        """Run the call kernel's guard on a kernel's result: the redo and list of the cells for the recovery's path."""
        cells = values["x0"].shape[0]
        flags = {name: torch.empty(cells, dtype=torch.int32, device=self.dev) for name in ("redo", "list")}
        arguments = self._arguments("guard", {**values, **flags}) + ([J] if self.nj else [])
        launch_kernel(
            self._fn["guard"], -(-cells // GUARD_BLOCK_THREADS) or 1, GUARD_BLOCK_THREADS, 0, stream, arguments
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
            "x0": x,
            "Ta": T,
            "Pa": P,
            "dta": dt,
            "hin": h,
            "lm": self._ptr(lm),
            "rc": rc,
            "count": self.count,
            "B": ctypes.c_int(x.shape[0]),
            "rtol": ctypes.c_double(rtol),
            "atol": ctypes.c_double(atol),
            "max_steps": ctypes.c_int(max_steps),
            "adapt": ctypes.c_int(int(adapt)),
            PHOTOLYSIS_ARGUMENT: self._ptr(J),
        }

    @staticmethod
    def _arguments(kernel: str, values: dict[str, Any]) -> list[Any]:
        """Return a kernel's arguments, in its signature's order, from the values by parameter name."""
        return [values[name] for name in KERNEL_ARGUMENTS[kernel]]

    def _finish(self, redo: torch.Tensor) -> None:
        """Add the counts to self.total and keep the per-cell flag, on the device."""
        self.total += self.count
        self.flag = redo

    @property
    def status(self) -> dict[str, int]:
        """Return the last call's counts (one small copy from the device)."""
        return dict(zip(COUNTS, self.count[1:-1].tolist(), strict=True))

    def check(self) -> dict[str, int]:
        """Return the counts since construction (one small copy); raise if any cell was below T_MIN."""
        n = dict(zip(COUNTS, self.total[1:-1].tolist(), strict=True))
        if n["cold"]:
            raise ValueError(f"{n['cold']} cells below {T_MIN:g} K: outside the chemistry solver's domain")
        return n

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
        need = self._nw["jvp"] * (5 * self.nr + 2 + directions * (2 * self.nr + self.ns))
        if self._jscr.numel() < need:
            self._jscr = torch.empty(need, dtype=torch.float64, device=self.dev)

    def _check_tangents(self, tangents: dict[str, Any], cells: int, device: torch.device) -> int:
        """Refuse tangents of the wrong form and return their directions (1 without a leading axis)."""
        dx = tangents["dx"]
        one = dx.dim() == 2
        directions = 1 if one else dx.shape[0]
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
        given = {name: a for name, a in tangents.items() if a is not None}
        if (
            not 1 <= directions <= DMAX
            or any(tuple(a.shape) != shapes[name] for name, a in given.items())
            or ("dJ" in given and not self.nj)
        ):
            raise ValueError(
                f"tangents: dx [D, B, S], dT, dP, ddt, dh [D, B], dlm [D, B, R], drc [D, R, 9], "
                f"dJ [D, B, branches] (photo=True only) (D <= {DMAX}), or the same without the D axis"
            )
        if not all(a.dtype == torch.float64 and a.device == device for a in given.values()):
            raise ValueError("tangents: float64 on the device of x")
        return directions

    @on_device
    def jvp(
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

        Inputs are those of ``CudaSolver.launch``, plus lm [B, R] (log rate multipliers), rc [R, W] (the rate table),
        adapt (the contract) and the tangents dx [B, S], dT (K), dP (dyn cm^-2), ddt (s), dh (s) [B], dlm [B, R], drc
        [R, 9], dJ [B, branches] (s^-1); dlm, drc and dJ are optional. With a leading axis of D <= ``DMAX`` directions
        on every tangent, the D tangents share one call.

        Returns
        -------
        tuple of torch.Tensor
            x, h, done, accepted steps, dx, dh; the last two have the leading axis if the inputs do.
        """
        B, (x, T, P, dt, h), lm, rc, J = self._prep(x, T, P, dt, h, lm, rc, J, max_steps, rtol, atol)
        tangents = {"dx": dx, "dT": dT, "dP": dP, "ddt": ddt, "dh": dh, "dlm": dlm, "drc": drc, "dJ": dJ}
        D = self._check_tangents(tangents, B, x.device)
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
            "dlm": self._ptr(dlm),
            "drc": self._ptr(drc),
            "D": ctypes.c_int(D),
            "xout": xo,
            "hout": ho,
            "doneo": done,
            "noko": nok,
            "dxout": dxo,
            "dhout": dho,
            "scratch": self._jscr,
            "dJin": self._ptr(dJ),
        }
        block = 32 * self.wpb["jvp"]
        launch_kernel(self._fn["jvp"], nb, block, self.smem["jvp"], stream, self._arguments("jvp", values))
        flags = self._guard(values, J, stream)
        values |= {**flags, "max_steps": ctypes.c_int(REC_STEPS * max_steps)}
        launch_kernel(self._fn["jvp_rec"], nb, block, self.smem["jvp"], stream, self._arguments("jvp_rec", values))
        redo = flags["redo"]
        self._finish(redo)
        return xo, ho, done.bool(), nok, dxo[0] if one else dxo, dho[0] if one else dho

    @on_device
    def vjp(
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

        Inputs are those of ``CudaSolver.launch``, plus gx [B, S] and gh [B] (the cotangents of x_out and h_out; gh is
        optional), lm [B, R], rc [R, W] and adapt.

        Returns
        -------
        tuple
            x, h, done, accepted steps, and a dict of the cotangents x, T, P, dt, h, lm [B, R], rc [R, 9] and, with
            ``photo=True``, J [B, branches].
        """
        B, (x, T, P, dt, h), lm, rc, J = self._prep(x, T, P, dt, h, lm, rc, J, max_steps, rtol, atol)
        if (
            tuple(gx.shape) != (B, self.ns)
            or (gh is not None and tuple(gh.shape) != (B,))
            or not all(a.dtype == torch.float64 and a.device == x.device for a in (gx, gh) if a is not None)
        ):
            raise ValueError(f"cotangents gx [B, {self.ns}], gh [B]: float64 on the device of x")
        gx = gx.contiguous()
        gh = None if gh is None else gh.contiguous()
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
            "gh": self._ptr(gh),
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
            "scratch": self._scratch,
            "rec": self._rec,
            "gJ": self._ptr(g.get("J")),
            "gA": g_ln_a,
        }
        block = 32 * self.wpb["vjp"]
        launch_kernel(self._fn["vjp"], nb, block, self.smem["vjp"], stream, self._arguments("vjp", values))
        flags = self._guard(values, J, stream)
        values |= {**flags, "max_steps": ctypes.c_int(REC_STEPS * max_steps), "rec": self._rrec}
        launch_kernel(self._fn["vjp_rec"], nb, block, self.smem["vjp"], stream, self._arguments("vjp_rec", values))
        launch_kernel(self._fn["arrh"], self.nr, ARRH_BLOCK_THREADS, 0, stream, self._arguments("arrh", values))
        redo = flags["redo"]
        self._finish(redo)
        return xo, ho, done.bool(), nok, g
