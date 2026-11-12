"""What the two solvers share: the kernels' lifetime, the counts, the input checks and the tables every kernel reads."""

import math
import operator
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch

from ._constants import (
    COUNT_SLOTS,
    COUNTS,
    DRIFT,
    KERNEL_ARGUMENTS,
    PHOTOLYSIS_ARGUMENT,
    PHOTOLYSIS_KERNEL_NAMES,
    REC_STEPS,
    T_MAX,
    T_MIN,
    X_SUM_TOL,
)
from ._cuda import device_of, unload_at_deletion, unload_modules
from .kinvulcan import VulcanNetwork

if TYPE_CHECKING:
    import ctypes


def _first(mask: torch.Tensor) -> int:
    """Return the first index where mask is true (the caller knows there is one)."""
    return int(torch.nonzero(mask)[0, 0])


def check_tensor(name: str, a: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> None:
    """Refuse a tensor that is not float64 of this shape on this device, naming it and what it is."""
    if a.dtype != torch.float64 or a.device != device or tuple(a.shape) != shape:
        raise ValueError(
            f"{name}: float64 {list(shape)} on {device} expected, got {a.dtype} {list(a.shape)} on {a.device}"
        )


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
    """Refuse a call's inputs of the wrong form or range, before any launch, and return the number of cells."""
    if x.dim() != 2 or x.shape[1] != ns:
        raise ValueError(f"x: [B, {ns}] expected, got {list(x.shape)}")
    B = x.shape[0]
    check_tensor("x", x, (B, ns), dev)
    for name, a in (("T", T), ("P", P), ("dt", dt), ("h", h)):
        check_tensor(name, a, (B,), dev)
    if not (math.isfinite(rtol) and rtol > 0.0 and math.isfinite(atol) and atol > 0.0):
        raise ValueError(f"rtol and atol finite and positive expected, got {rtol}, {atol}")
    try:
        ms = operator.index(max_steps)
    except TypeError:
        raise ValueError(f"max_steps: an integer expected, got {max_steps!r}") from None
    most = int(np.iinfo(np.int32).max)  # the kernels count tries in an int, up to REC_STEPS * max_steps
    if ms < 1 or REC_STEPS * ms > most:
        raise ValueError(f"max_steps from 1 to {most // REC_STEPS} expected, got {ms}")
    return int(B)


def check_photolysis(J: torch.Tensor | None, cells: int, branches: int, device: torch.device) -> torch.Tensor | None:
    """Return the call's J, None without photolysis; refuse a J that breaks the rule of docs/api.md.

    Without photolysis the call takes no J or an empty one, [B, 0]; with it, J [B, branches] is required. The C++
    caller (cpp/maxichem.cpp) gives the same two rule messages; its form message differs in the dtype's name.
    """
    if not branches:
        if J is None:
            return None
        if J.dim() != 2 or J.shape[1] != 0:
            raise ValueError(
                f"J: none or an empty [{cells}, 0] one expected, since this solver has no photolysis; got "
                f"{list(J.shape)}"
            )
        check_tensor("J", J, (cells, 0), device)
        return None
    if J is None:
        raise ValueError(f"J [B, {branches}] is required with photolysis: pass zeros for dark cells")
    check_tensor("J", J, (cells, branches), device)
    return J.contiguous()


def to_device(values: Any, device: torch.device, view: Any = None) -> torch.Tensor:
    """Copy a table to the device, read as dtype view if given."""
    return torch.tensor(values if view is None else np.asarray(values).view(view), device=device)


def shared_tables(tab: dict[str, Any], blob: bytes, device: torch.device) -> dict[str, torch.Tensor]:
    """Copy to the device the tables every kernel reads, by the kernels' parameter names."""
    return {
        "perm": to_device(tab["perm"], device),
        "blob": to_device(np.frombuffer(blob, np.int32).copy(), device),
        "rc": to_device(tab["rc"], device),
        "rf": to_device(tab["rf"], device),
        "nasa": to_device(tab["nasa"], device),
        "sst": to_device(tab["sst"].ravel(), device, np.int32),
        "jt": to_device(tab["jt"], device, np.int64),
        "lu": to_device(tab["lu"], device, np.int64),
        "optr": to_device(tab["optr"], device),
        "didx": to_device(tab["didx"], device, np.int16),
        "atoms": to_device(tab["atoms"], device),
    }


class Solver:
    """What both solvers share: the kernels' lifetime, the counts on the device, and the checks of a new solver."""

    nj: int  # photolysis branches, set when a subclass builds its tables

    def __init__(self, net: VulcanNetwork, device: torch.device | str, drift: float | None, photo: bool) -> None:
        if photo and (not net.photo or net.photo_missing):
            raise ValueError(
                "photo=True needs a network with photolysis branches whose species are all in its species "
                f"list (branches {len(net.photo)}, missing species {net.photo_missing})"
            )
        value = float(DRIFT if drift is None else drift)
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"drift: a finite allowance of at least 0 expected, got {drift}")
        self.drift = value
        self.dev = device_of(device)
        self._modules: list[ctypes.c_void_p] = []  # every module loaded, also when the setup fails part of the way
        self._tables: dict[str, torch.Tensor] = {}  # by the kernels' parameter names
        self._rounds: dict[str, ctypes.c_int] = {}
        self._scratch: dict[str, torch.Tensor] = {}
        self.count = torch.zeros(len(COUNT_SLOTS), dtype=torch.int32, device=self.dev)
        self.total = torch.zeros(len(COUNT_SLOTS), dtype=torch.int64, device=self.dev)
        # the per-cell flag [B], int32, of the last call if it returned; None before the first and after one that raised
        self.flag: torch.Tensor | None = None

    def close(self) -> None:
        """Unload the kernels and drop the tables and scratch, after the device's work is done.

        The solver cannot be called afterwards. Its counts stay readable.
        """
        unload_modules(self.dev, self._modules)
        self._tables = {}
        self._scratch = {}

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

    def _require_open(self) -> None:
        """Refuse work on a closed solver."""
        if not self._modules:
            raise RuntimeError("this solver was closed; build a new one")

    def _arguments(self, kernel: str, values: dict[str, Any]) -> list[Any]:
        """Return a kernel's arguments, in its signature's order, from the values by parameter name.

        The call and recovery kernels take J last only with photolysis, the guard never; the derivative kernels' lists
        always hold it, null without.
        """
        extra = [values[PHOTOLYSIS_ARGUMENT]] if self.nj and kernel in PHOTOLYSIS_KERNEL_NAMES else []
        return [values[name] for name in KERNEL_ARGUMENTS[kernel]] + extra

    @property
    def fallback(self) -> dict[str, int]:
        """Return the last call's counts (one small copy from the device)."""
        return dict(zip(COUNTS, self.count[1:-1].tolist(), strict=True))

    def _refuse(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        dt: torch.Tensor,
        J: torch.Tensor | None,
        status: torch.Tensor,
    ) -> None:
        """Raise ValueError if the call just launched refused a cell (one small copy from the device, a host sync).

        The kernel that steps a cell screens it once and writes its status: 0, or why it is refused (bad input, cold
        or hot). The guard counts the refused cells from it, and the host names the first of each kind from it.
        """
        n = self.fallback
        if not (n["bad_input"] or n["cold"] or n["hot"]):
            return
        refused = status != 0
        cold, hot = refused & (T < T_MIN), refused & (T > T_MAX)
        problems = []
        for kind, cells, side in (("cold", cold, f"below {T_MIN:g} K"), ("hot", hot, f"above {T_MAX:g} K")):
            if n[kind]:
                i = _first(cells)
                problems.append(
                    f"{n[kind]} cells {side}, outside the solver's range {T_MIN:g}-{T_MAX:g} K (first: cell {i}, "
                    f"T = {float(T[i]):g} K); leave them out of the call"
                )
        if n["bad_input"]:
            i = _first(refused & torch.logical_not(cold | hot))
            where = (
                f"cell {i}: sum(x) - 1 = {float(x[i].sum() - 1.0):g} (added in another order than the kernel's, so "
                f"it can differ in the last bits), min(x) = {float(x[i].min()):g}, "
                f"T = {float(T[i]):g} K, P = {float(P[i]):g} dyn cm^-2, dt = {float(dt[i]):g} s"
                + ("" if J is None else f", min(J) = {float(J[i].min()):g} s^-1")
            )
            problems.append(
                f"{n['bad_input']} cells of bad input: x negative or non-finite, |sum(x) - 1| above {X_SUM_TOL:g} "
                "or no x above 0; T non-finite, P non-finite, not above 0 or so large that P / (k_B T) overflows, dt "
                f"negative or non-finite, or J negative or non-finite (first: {where}); pass x normalized to sum 1 in "
                "float64, clamped at 0"
            )
        raise ValueError("the call refused cells: " + "; ".join(problems))

    def check(self) -> dict[str, int]:
        """Return the counts since construction (one small copy); raise if any cell was outside [T_MIN, T_MAX]."""
        n = dict(zip(COUNTS, self.total[1:-1].tolist(), strict=True))
        if n["cold"] or n["hot"]:
            raise ValueError(
                f"{n['cold']} cells below {T_MIN:g} K and {n['hot']} above {T_MAX:g} K: outside the chemistry "
                "solver's range"
            )
        return n
