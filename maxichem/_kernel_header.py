"""The kernels' data header: one network's sizes, offsets and bit masks as preprocessor macros.

The kernels themselves are the files in ``kernels/``; see docs/architecture.md.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._ad_tables import GATHER_FIELDS, layout, scratch_layouts, scratch_sizes
from ._constants import (
    ARRH_BLOCK_THREADS,
    CALL_END,
    COUNT_SLOTS,
    DMAX,
    ELEMENT_PIVOT_MIN,
    ERR_FLOOR,
    FAC_AFTER_REJECT,
    FAC_NONFINITE,
    FACACC,
    FACMAX,
    FACREJ,
    GMAX,
    H_MIN,
    K_COLL,
    KB,
    NASA_SPLIT_T,
    NO_SLOT,
    P0,
    PIVTAU,
    RATE_LAW_FLAGS,
    RATE_LAW_SHIFTS,
    SECANT_MAX_ORDER,
    SECANT_MIN_ORDER,
    SECANT_TARGET,
    SECMIN,
    STEP_SAFETY,
    T_MAX,
    T_MIN,
    WARP_LANES,
    X_SUM_TOL,
)
from ._tables import (
    call_layout,
    call_rates_layout,
    dense_w_doubles,
    pack_small_tables,
    rate_slots,
    recovery_int_layout,
    recovery_layout,
    regions,
    retry_layout,
    scratch_layout,
)

_KERNELS = Path(__file__).parent / "kernels"
_HEADER = "network.h"
# The method's settings, which the kernels' constexprs of the same names read as NAME_VALUE
_METHOD_SETTINGS = {
    "H_MIN": H_MIN,
    "CALL_END": CALL_END,
    "STEP_SAFETY": STEP_SAFETY,
    "ERR_FLOOR": ERR_FLOOR,
    "FAC_NONFINITE": FAC_NONFINITE,
    "FACACC": FACACC,
    "FACREJ": FACREJ,
    "FACMAX": FACMAX,
    "FAC_AFTER_REJECT": FAC_AFTER_REJECT,
    "SECMIN": SECMIN,
    "SECANT_MIN_ORDER": SECANT_MIN_ORDER,
    "SECANT_MAX_ORDER": SECANT_MAX_ORDER,
    "SECANT_TARGET": SECANT_TARGET,
    "PIVTAU": PIVTAU,
    "ELEMENT_PIVOT_MIN": ELEMENT_PIVOT_MIN,
    "GMAX": GMAX,
}


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


def _offset_macros(layouts: dict[str, dict[str, int]]) -> dict[str, int]:
    """Return each region's array offsets as REGION_ARRAY macros (_tables.regions)."""
    return {f"{region}_{name}": offset for region, names in layouts.items() for name, offset in names.items()}


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


def call_header(tab: dict[str, Any], cpb: int, wsh: bool, drift: float) -> str:
    """Return network.h for the call, guard and recovery kernels; wsh: the recovery holds W in shared memory."""
    ns, nnz, tb0, law = tab["ns"], tab["nnz"], tab["tb0"], tab["law"]
    blob, offsets = pack_small_tables(tab)
    # each region's offsets, named after its region, and its size
    rate_offsets, rate_size = regions(rate_slots(tab["nr"]))
    call, perc = call_layout(tab)
    recovery, perwr = recovery_layout(tab, wsh)
    call_scratch, call_stride = scratch_layout(tab, False)
    recovery_scratch, recover_stride = scratch_layout(tab, not wsh)
    layouts = {
        "RATE": rate_offsets,
        "C": call,
        "CR": call_rates_layout(tab)[0],
        "R": recovery,
        "RI": recovery_int_layout(tab)[0],
        "S": call_scratch,
        "SR": recovery_scratch,
        "RSC": retry_layout(tab)[0],
    }
    macros = {
        "CALL_STRIDE": call_stride,
        "RECOVER_STRIDE": recover_stride,
        "RATE_SLOTS": rate_size,
        "W_DOUBLES": dense_w_doubles(tab),
        **_offset_macros(layouts),
        "NS": ns,
        "NR": tab["nr"],
        "NRT": tab["nrt"],
        "NJ": tab["nj"],
        "NNZ": nnz,
        "TB0": tb0,
        "NPART": tab["npart"],
        "K0": tab["k0"],
        "NT": ns - tab["k0"],
        "RP": max(1, -(-tab["k0"] // WARP_LANES)),
        "RCW": tab["rc"].shape[1],
        "CPB": cpb,
        "W": tab["warps"],
        "PERC": perc,
        "TBYTES": len(blob),
        "JSMAX": tab["jsmax"],
        "NFIX": tab["nfix"],
        "NEL": tab["atoms"].shape[1],
        "ETOT": repr(float(tab["atoms"].sum())),
        "TMIN": repr(T_MIN),
        "TMAX": repr(T_MAX),
        "XSUMTOL": repr(X_SUM_TOL),
        "DRIFT": repr(float(drift)),
        "KCOLL": repr(K_COLL),
        "KBOLTZ": repr(KB),
        "NASA_SPLIT_T_K": repr(NASA_SPLIT_T),
        "P0BAR": repr(P0),
        # repr gives the double back exactly
        **{f"{name}_VALUE": repr(float(value)) for name, value in _METHOD_SETTINGS.items()},
        "WSH": int(wsh),
        "RPD": -(-ns // WARP_LANES),
        "PERWR": perwr,
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


def derivative_header(tab: dict[str, Any], ext: dict[str, Any], kind: str, wpb: int, cap: int, drift: float) -> str:
    """Return the text of network.h for the tangent ("jvp") or adjoint ("vjp") kernels.

    They are built on the call kernels with one cell per block, so this header starts with ``call_header``'s.
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
        "ARRH_BLOCK_THREADS": ARRH_BLOCK_THREADS,
        **scratch_sizes(tab, cap),
        **_offset_macros({region: names for region, (names, _) in scratch_layouts(tab).items()}),
        **GATHER_FIELDS,
        **{f"A_{name}": offset for name, offset in offsets.items()},
    }
    return call_header(tab, 1, True, drift) + "\n".join(f"#define {k} {v}" for k, v in macros.items()) + "\n"
