"""The edge to the chemistry solver: its pressure unit, and the stop when a call leaves cells unfinished.

A cell the call did not finish holds a state for the wrong elapsed time, so the run stops; the message names the cause
from the call's own counts, and the fix that matches it. The call itself raises on refused cells (cold, hot, bad
input) before it returns, so after a real call those counts are 0 here, and the cells left are those still failing
after the recovery. The causes named for those three counts serve counts from a solver whose raise is replaced, as in
the tests. docs/architecture.md places this module.
"""

from collections.abc import Mapping

import torch

from maxichem.kincuda import REC_STEPS, T_MAX, T_MIN, X_SUM_TOL

__all__ = ["PA_TO_DYN_CM2", "UnfinishedCellsError", "require_finished"]

PA_TO_DYN_CM2 = 10.0  # the solver takes pressure in dyn cm^-2 (VULCAN's unit): 1 Pa = 10 dyn cm^-2


class UnfinishedCellsError(RuntimeError):
    """A chemistry call returned cells that are not done."""


def _causes(counts: Mapping[str, int]) -> list[str]:
    """Return the cause and fix of each count that leaves a cell not done.

    The call's other counts (non-finite, unfinished, drift) lead to still_failed, or to nothing once recovered.
    """
    out = []
    for name, side, edge in (("cold", "below", T_MIN), ("hot", "above", T_MAX)):
        if counts.get(name):
            out.append(
                f"{counts[name]} cell(s) {side} the solver's {T_MIN:g}-{T_MAX:g} K domain (refused at {edge:g} K). "
                f"No solver setting helps: the flow made them this {name}; look at the deck (its lid, its radiation) "
                "where they are"
            )
    if counts.get("bad_input"):
        out.append(
            f"{counts['bad_input']} cell(s) with bad input (non-finite or negative mole fractions, |sum(x) - 1| above "
            f"{X_SUM_TOL:g}, none above 0, P non-finite, P <= 0 or P / (k_B T) overflowing, a non-finite T, dt < 0 "
            "or a bad J). The driver's inputs to the "
            "call are wrong there: check the cells' tracers, their normalization, T and P"
        )
    if counts.get("still_failed"):
        out.append(
            f"{counts['still_failed']} cell(s) still failing the guard after the recovery's {REC_STEPS} times longer "
            f"try (in the call: {counts.get('unfinished', 0)} out of steps, {counts.get('nonfinite', 0)} non-finite, "
            f"{counts.get('drift', 0)} element drift). Run with a shorter --dt"
        )
    return out or ["no count of the solver explains it; report this with the counts above"]


def require_finished(
    done: torch.Tensor, T: torch.Tensor, P: torch.Tensor, counts: Mapping[str, int], where: str
) -> None:
    """Raise when the chemistry call left any cell not done, naming the cause and its fix.

    Parameters
    ----------
    done : torch.Tensor [cells]
        The call's done flags.
    T : torch.Tensor [cells]
        The call's temperatures, K.
    P : torch.Tensor [cells]
        The call's pressures, Pa.
    counts : mapping of str to int
        This call's counts, as ``CudaSolver.fallback`` gives them (keys of ``maxichem.kincuda.COUNTS``).
    where : str
        Where in the run the call was, for the message, for example "at cycle 12".
    """
    if not (done.dim() == 1 and done.shape == T.shape == P.shape):
        shapes = [tuple(a.shape) for a in (done, T, P)]
        raise ValueError(f"done, T and P must all be [cells], got shapes {shapes}")
    bad = ~done.bool()
    if not bool(bad.any()):
        return
    i = int(torch.nonzero(bad)[0])
    risen = {k: int(v) for k, v in counts.items() if v}
    raise UnfinishedCellsError(
        f"{int(bad.sum())} chemistry cell(s) not done {where}; the first, cell {i}: T {float(T[i]):.1f} K, "
        f"P {float(P[i]):.4g} Pa. The call's counts: {risen}. " + ". ".join(_causes(risen)) + "."
    )
