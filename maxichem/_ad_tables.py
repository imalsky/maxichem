"""The derivative kernels' tables and shared-memory layout, beyond the call kernels' tables."""

from typing import Any

import numpy as np

from ._constants import WARP_LANES
from ._tables import balance_lanes, rate_slots, regions
from .kinvulcan import VulcanNetwork

# Each kernel's per-warp work arrays (name, length in units of NS unless layout says otherwise), and the arrays sharing
# their space (free while they are used).
_LAYOUT = {
    "jvp": (
        ("M", "DINV", "N0", "N1", "V", "PART", "NN", "Y", "F", "SV", "U", "K1", "K2", "K3", "K4",
         "ND", "YD", "SV1N", "KD1", "KD2", "KD3", "KD4", "YS", "YDS"),
        {"FD": "F", "SVN": "SV", "T1": "YS", "T2": "YDS", "N2": "YD", "N12": "F"},
    ),
    "vjp": (
        ("M", "DINV", "N0", "N1", "V", "PART", "NN", "Y", "F", "SV", "U", "K1", "K2", "K3", "K4",
         "NB", "KB1", "KB2", "KB3", "Z", "YB", "YB2", "NBD"),
        {"YS": "F"},
    ),
}  # fmt: skip
# A name its kernel does not use sits at offset 0, so one header compiles and lints both kernels' files (tools/).
_ALL_NAMES = sorted({name for names, aliases in _LAYOUT.values() for name in (*names, *aliases)})


# The transposed Jacobian's words: rate (12 bits), species a and b (7 each), piece (9), end of piece, used.
_GT_A, _GT_B, _GT_PIECE, _GT_END, _GT_USED = 12, 19, 26, 35, 37
_MAX_RATES, _MAX_SPECIES_SLOTS, _MAX_PIECES = 1 << 12, 1 << 7, 1 << 9
_MAX_LEADING_ROWS = 64  # the row masks of the transposed solves are 64-bit words
GATHER_FIELDS = {
    "GT_A": _GT_A,
    "GT_B": _GT_B,
    "GT_PIECE": _GT_PIECE,
    "GT_END": _GT_END,
    "GT_USED": _GT_USED,
    "GT_RATE_MASK": f"{_MAX_RATES - 1}ull",
    "GT_SPECIES_MASK": f"{_MAX_SPECIES_SLOTS - 1}ull",
    "GT_PIECE_MASK": f"{_MAX_PIECES - 1}ull",
}


def supported(tab: dict[str, Any]) -> None:
    """Raise NotImplementedError unless the derivative kernels take the network of these tables."""
    if tab["k0"] > _MAX_LEADING_ROWS or 2 * tab["nr"] >= _MAX_RATES:
        raise NotImplementedError(
            f"the derivative kernels take at most {_MAX_LEADING_ROWS} leading rows and {_MAX_RATES // 2 - 1} reactions "
            f"(photolysis branches included): this network has {tab['k0']} and {tab['nr']}"
        )


def scratch_layouts(tab: dict[str, Any]) -> dict[str, tuple[dict[str, int], int]]:
    """Return the derivative kernels' global scratch regions by their names' prefix in network.h (_tables.regions).

    JS, per warp of the tangent kernels: the rates, a per-reaction work array and the collision limit's weights; the
    directions' rate and state tangents follow, JVP_DIRECTION_STRIDE each. VS, per warp of the adjoint kernels: the
    rates, their cotangents, rho (4 x 2 NR) and the limit's weights. REC, one stored attempt: N, h, t, accepted, then
    the previous attempt's error, size and rejection.
    """
    ns, nr = tab["ns"], tab["nr"]
    rates = ("RATES", regions(rate_slots(nr))[1])
    record = [("N", ns), ("H", 1), ("T", 1), ("OK", 1), ("PEN", 1), ("PHA", 1), ("PREJ", 1)]
    return {
        "JS": regions([rates, ("VA", nr), ("CW", 2 * nr)]),
        "VS": regions([rates, ("LBK", 2 * nr), ("RHO", 8 * nr), ("CW", 2 * nr)]),
        "REC": regions(record),
    }


def scratch_sizes(tab: dict[str, Any], cap: int) -> dict[str, int]:
    """Return the global scratch of the derivative kernels, in doubles, by the name the kernels read it.

    The strides of scratch_layouts' regions; JVP_DIRECTION_STRIDE, per direction: the rate and state tangents; CAPR:
    attempts stored per recovered cell, enough for its tries.
    """
    sizes = {name: size for name, (_, size) in scratch_layouts(tab).items()}
    return {
        "VJP_STRIDE": sizes["VS"],
        "JVP_STRIDE": sizes["JS"],
        "JVP_DIRECTION_STRIDE": 2 * tab["nr"] + tab["ns"],
        "REC": sizes["REC"],
        "CAPR": 2 * cap,
    }


def ad_tables(net: VulcanNetwork, tab: dict[str, Any]) -> dict[str, Any]:
    """Build the derivative kernels' tables beyond those of ``_tables.tables``.

    Returns
    -------
    dict
        gt and xg: the stream of the transposed concentration Jacobian and its piece offsets. urow and lrow: the row
        masks of the transposed triangular solves. px: the photolysis branches' product slots. npg: the gather pieces.
    """
    ns, nr, nrt, k0, pattern = tab["ns"], tab["nr"], tab["nrt"], tab["k0"], tab["pattern"]
    pos = {int(s): k for k, s in enumerate(tab["perm"])}
    reac = net.reac + (net.photo if nr > nrt else [])
    S = (
        np.hstack([np.asarray(net.S, np.float64), np.asarray(net.Sp, np.float64)])
        if nr > nrt
        else np.asarray(net.S, np.float64)
    )
    rows: list[list[tuple[int, int, int]]] = [[] for _ in range(ns)]
    for r in range(nr):
        if not np.any(S[:, r] != 0):
            continue
        for side, k in ((reac[r]["re"], r), (reac[r]["pr"], nr + r))[: 1 if r >= nrt else 2]:
            sl = [pos[q] for q in side]
            for m in range(len(sl)):
                rest = sl[:m] + sl[m + 1 :]
                rest = rest + [ns] * (2 - len(rest))
                rows[sl[m]].append((k, rest[0], rest[1]))
    ch = -(-sum(len(t) for t in rows) // WARP_LANES)
    pieces, xg, nx = [], [0], 0
    for i, t in enumerate(rows):
        for c in range(0, len(t), ch):
            pieces.append((i if c == 0 else ns + nx, t[c : c + ch]))
            nx += c > 0
        xg.append(nx)
    npg = ns + nx
    if npg >= _MAX_PIECES:
        raise ValueError(
            f"too many gather pieces: {npg}, below {_MAX_PIECES}; this network is too large for the adjoint"
        )
    lanes, ground = balance_lanes(range(len(pieces)), lambda q: len(pieces[q][1]), WARP_LANES)
    gt = np.zeros((ground, WARP_LANES), np.uint64)
    for l_, its in enumerate(lanes):
        q = 0
        for it in its:
            dst, terms = pieces[it]
            for e, (k, a, b) in enumerate(terms):
                end = int(e == len(terms) - 1)
                gt[q, l_] = int(k) | a << _GT_A | b << _GT_B | dst << _GT_PIECE | end << _GT_END | 1 << _GT_USED
                q += 1
    urow = sum(1 << k for k in range(k0) if any((k, j) in pattern for j in range(k + 1, ns)))
    lrow = sum(1 << k for k in range(k0) if any((k, j) in pattern for j in range(k)))
    px = [pos.get(q, ns) for r in reac[nrt:] for q in list(r["pr"]) + [None] * (3 - len(r["pr"]))] or [ns]
    return {"gt": gt, "xg": np.array(xg, np.int32), "npg": npg, "urow": urow, "lrow": lrow, "px": px}


def layout(tab: dict[str, Any], ext: dict[str, Any], kind: str) -> tuple[dict[str, int], int]:
    """Return the shared-memory offsets (doubles) of kernel kind's per-warp work arrays and their total."""
    ns, nr = tab["ns"], tab["nr"]
    size = {
        "M": tab["nnz"],
        "N0": ns + 1,
        "N1": ns + 1,
        "V": max(nr, ns + 1),
        "F": ns + 1,
        "YD": ns + 1,
        "YB": ns + 1,
        "YB2": ns + 1,
        "NBD": ns + 1,
        "PART": 2 * tab["npart"] if kind == "jvp" else max(tab["npart"], ext["npg"]),
    }
    names, alias = _LAYOUT[kind]
    off, o = dict.fromkeys(_ALL_NAMES, 0), 0
    for k in names:
        off[k] = o
        o += size.get(k, ns)
    for k, a in alias.items():
        off[k] = off[a]
    return off, o
