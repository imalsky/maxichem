"""The kernels' data header: one network's sizes, offsets and bit masks as preprocessor macros.

Public names: ``call_header``, ``derivative_header``, ``kernel_sources``. The kernels themselves are the files in
``kernels/``; see docs/architecture.md.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import kinvulcan
from ._ad_tables import layout
from ._constants import (
    CAP,
    COUNT_SLOTS,
    DMAX,
    DRIFT,
    K_COLL,
    NO_SLOT,
    RATE_LAW_FLAGS,
    RATE_LAW_SHIFTS,
    T_MIN,
)
from ._tables import pack_small_tables, rates_fit_in_block, shared_doubles_per_cell, shared_doubles_recover

_KERNELS = Path(__file__).parent / "kernels"
_HEADER = "network.h"


def kernel_sources(header: str) -> dict[str, str]:
    """Return every kernel file's text by name, with the generated header under its include name."""
    return {path.name: path.read_text() for path in sorted(_KERNELS.iterdir())} | {_HEADER: header}


def _bit_set(name: str, members: Iterable[int], size: int) -> str:
    """Return a macro for bit k of a set of integers below size, held in 64-bit words."""
    members = set(members)
    words = [
        sum(1 << (k - 64 * word) for k in members if 64 * word <= k < 64 * word + 64)
        for word in range(max(1, -(-size // 64)))
    ]
    macro = f"(({words[-1]:#x}ull >> ((k) & 63)) & 1ull)"
    for word in range(len(words) - 2, -1, -1):
        macro = f"((k) < {64 * word + 64} ? (({words[word]:#x}ull >> ((k) & 63)) & 1ull) : {macro})"
    return f"#define {name}(k) {macro}"


def _two_bit_field(name: str, values: list[int]) -> str:
    """Return a macro for the 2-bit value values[q], held in 64-bit words."""
    words = [
        sum(int(v) << (2 * (q - 32 * word)) for q, v in enumerate(values) if 32 * word <= q < 32 * word + 32)
        for word in range(max(1, -(-len(values) // 32)))
    ]
    macro = f"((int)(({words[-1]:#x}ull >> (2 * ((q) & 31))) & 3ull))"
    for word in range(len(words) - 2, -1, -1):
        macro = f"((q) < {32 * word + 32} ? ((int)(({words[word]:#x}ull >> (2 * ((q) & 31))) & 3ull)) : {macro})"
    return f"#define {name}(q) {macro}"


def _pattern_macros(tab: dict[str, Any]) -> list[str]:
    """Return the macros that describe the LU pattern and the widest reaction sides.

    Column k of the sparse leading part has L entries in the leading rows (LPRE), L entries in the trailing rows (LTAIL)
    and U entries (UPRE); trailing column K0 + j has U entries in the leading rows (UTAIL). VSA(q), VSB(q): the widest
    forward and reverse side of round q of NL reactions.
    """
    ns, k0, pattern = tab["ns"], tab["k0"], tab["pattern"]
    return [
        _bit_set("LPRE", (k for k in range(k0) if any((i, k) in pattern for i in range(k + 1, k0))), k0),
        _bit_set("LTAIL", (k for k in range(k0) if any((i, k) in pattern for i in range(k0, ns))), k0),
        _bit_set("UPRE", (k for k in range(k0) if any((i, k) in pattern for i in range(k))), k0),
        _bit_set("UTAIL", (j for j in range(ns - k0) if any((i, k0 + j) in pattern for i in range(k0))), ns - k0),
        _two_bit_field("VSA", tab["vs"][0]),
        _two_bit_field("VSB", tab["vs"][1]),
    ]


def _stream_fields(widths: dict[str, int]) -> dict[str, int | str]:
    """Return the bit positions and masks of the packed S v, Jacobian and LU stream words."""

    def mask(bits: int) -> int:
        return (1 << bits) - 1

    reaction, rate, species, slot, part = (widths[k] for k in ("r", "k", "s", "slot", "part"))
    j_a = rate + 2
    return {
        "SV_REACTION_MASK": f"{mask(reaction)}u",
        "SV_SIGN": reaction,
        "SV_DOUBLE": reaction + 1,
        "SV_PIECE": reaction + 2,
        "SV_PIECE_MASK": f"{mask(part)}u",
        "SV_END": reaction + 2 + part,
        "J_RATE_MASK": f"{mask(rate)}u",
        "J_SIGN": rate,
        "J_DOUBLE": rate + 1,
        "J_A": j_a,
        "J_B": j_a + species,
        "J_SPECIES_MASK": f"{mask(species)}u",
        "J_END": j_a + 2 * species,
        "LU_SLOT_MASK": f"{mask(slot)}ull",
        "LU_B": slot,
        "LU_C": 2 * slot,
    }


def call_header(tab: dict[str, Any], cpb: int, wsh: bool = True, drift: float | None = None) -> str:
    """Return the data header of the call, guard and recovery kernels.

    Parameters
    ----------
    tab : dict
        The network's tables, from ``_tables.tables``.
    cpb : int
        Cells per block.
    wsh : bool
        The recovery kernel holds W in shared memory.
    drift : float, optional
        The element-drift allowance, relative; the default is ``DRIFT``.

    Returns
    -------
    str
        The text of ``network.h``.
    """
    ns, nnz, tb0, law = tab["ns"], tab["nnz"], tab["tb0"], tab["law"]
    blob, offsets = pack_small_tables(tab)
    macros = {
        "NS": ns,
        "NR": tab["nr"],
        "NRT": tab["nrt"],
        "NJ": tab["nj"],
        "NNZ": nnz,
        "TB0": tb0,
        "NPART": tab["npart"],
        "K0": tab["k0"],
        "NT": ns - tab["k0"],
        "RP": max(1, -(-tab["k0"] // 32)),
        "RCW": tab["rc"].shape[1],
        "CPB": cpb,
        "W": tab["warps"],
        "PERC": shared_doubles_per_cell(tab),
        "VOFF": tb0 if rates_fit_in_block(tab) else nnz + 10 * ns + 3,
        "TBYTES": len(blob),
        "JSMAX": tab["jsmax"],
        "NFIX": tab["nfix"],
        "NEL": tab["atoms"].shape[1],
        "ETOT": repr(float(tab["atoms"].sum())),
        "TMIN": repr(T_MIN),
        "DRIFT": repr(float(DRIFT if drift is None else drift)),
        "KCOLL": repr(K_COLL),
        "KBOLTZ": repr(kinvulcan.KB),
        "P0BAR": repr(kinvulcan.P0),
        "WSH": int(wsh),
        "VR": max(tab["nr"], ns + 1),
        "RPD": -(-ns // 32),
        "PERWR": shared_doubles_recover(tab, wsh),
        "HAS_TROE": int("troe" in law),
        "HAS_EFF": int("eff" in law),
        **{f"O_{name.upper()}": offset for name, offset in offsets.items()},
        "NO_SLOT": f"{NO_SLOT:#x}u",
        **RATE_LAW_FLAGS,
        **RATE_LAW_SHIFTS,
        **{f"COUNT_{name.upper()}": slot for slot, name in enumerate(COUNT_SLOTS)},
        **_stream_fields(tab["widths"]),
    }
    lines = [f"#define {name} {value}" for name, value in macros.items()]
    return "\n".join([*lines, *_pattern_macros(tab)]) + "\n"


def derivative_header(
    tab: dict[str, Any], ext: dict[str, Any], kind: str, wpb: int, cap: int = CAP, drift: float | None = None
) -> str:
    """Return the data header of the tangent ("jvp") or adjoint ("vjp") kernels.

    They are built on the call kernels with one cell per block, so this header starts with ``call_header``'s.

    Parameters
    ----------
    tab, ext : dict
        The network's tables, from ``_tables.tables`` and ``_ad_tables.ad_tables``.
    kind : str
        "jvp" or "vjp".
    wpb : int
        Warps per block.
    cap : int
        Attempts stored per cell before the adjoint switches to checkpoints.
    drift : float, optional
        The element-drift allowance, relative; the default is ``DRIFT``.

    Returns
    -------
    str
        The text of ``network.h``.
    """
    offsets, perwa = layout(tab, ext, kind)
    macros = {
        "WPB": wpb,
        "PERWA": perwa,
        "CAP": cap,
        "DMAX": DMAX,
        "NPG": ext["npg"],
        "UROW": f"{ext['urow']}ull",
        "LROW": f"{ext['lrow']}ull",
        "PX_SLOTS": ", ".join(map(str, ext["px"])),
        **{f"A_{name}": offset for name, offset in offsets.items()},
    }
    return call_header(tab, 1, drift=drift) + "\n".join(f"#define {k} {v}" for k, v in macros.items()) + "\n"
