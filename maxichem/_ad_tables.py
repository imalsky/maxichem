"""The derivative kernels' tables and shared-memory layout, beyond the call kernels' tables."""

from typing import Any

import numpy as np

from ._tables import balance_lanes
from .kinvulcan import VulcanNetwork

# per-warp work arrays of each kernel (name, length in units of NS unless given) and the arrays sharing their space
# (free while they are used); names a kernel does not use sit at offset 0.
_LAYOUT = {
    "jvp": (
        [
            "M",
            "DINV",
            "N0",
            "N1",
            "V",
            "PART",
            "NN",
            "Y",
            "F",
            "SV",
            "U",
            "K1",
            "K2",
            "K3",
            "K4",
            "ND",
            "YD",
            "SV1N",
            "KD1",
            "KD2",
            "KD3",
            "KD4",
            "YS",
            "YDS",
        ],
        {"FD": "F", "SVN": "SV", "T1": "YS", "T2": "YDS", "N2": "YD", "N12": "F"},
    ),
    "vjp": (
        [
            "M",
            "DINV",
            "N0",
            "N1",
            "V",
            "PART",
            "NN",
            "Y",
            "F",
            "SV",
            "U",
            "K1",
            "K2",
            "K3",
            "K4",
            "NB",
            "KB1",
            "KB2",
            "KB3",
            "Z",
            "YB",
            "YB2",
            "NBD",
        ],
        {"YS": "F"},
    ),
}
NAMES = (
    "M",
    "DINV",
    "N0",
    "N1",
    "N2",
    "N12",
    "V",
    "PART",
    "NN",
    "Y",
    "F",
    "SV",
    "U",
    "K1",
    "K2",
    "K3",
    "K4",
    "ND",
    "YD",
    "FD",
    "SVN",
    "SV1N",
    "KD1",
    "KD2",
    "KD3",
    "KD4",
    "YS",
    "YDS",
    "T1",
    "T2",
    "NB",
    "KB1",
    "KB2",
    "KB3",
    "Z",
    "YB",
    "YB2",
    "NBD",
)


def supported(tab: dict[str, Any]) -> None:
    """Raise NotImplementedError unless the derivative kernels take the network of these tables.

    At most 64 leading rows (with the 32-row trailing block: at most 96 species), at most 2,047 reactions (photolysis
    branches included).
    """
    if tab["k0"] > 64 or 2 * tab["nr"] >= 4096:
        raise NotImplementedError(
            "the derivative kernels take networks of at most 96 species (64 leading rows) "
            "and at most 2,047 reactions (photolysis branches included)"
        )


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
    ch = -(-sum(len(t) for t in rows) // 32)
    pieces, xg, nx = [], [0], 0
    for i, t in enumerate(rows):
        for c in range(0, len(t), ch):
            pieces.append((i if c == 0 else ns + nx, t[c : c + ch]))
            nx += c > 0
        xg.append(nx)
    npg = ns + nx
    if npg > 511:
        raise ValueError("too many gather pieces")
    lanes, ground = balance_lanes(range(len(pieces)), lambda q: len(pieces[q][1]), 32)
    gt = np.zeros((ground, 32), np.uint64)
    for l_, its in enumerate(lanes):
        q = 0
        for it in its:
            dst, terms = pieces[it]
            for e, (k, a, b) in enumerate(terms):
                gt[q, l_] = int(k) | a << 12 | b << 19 | dst << 26 | int(e == len(terms) - 1) << 35 | 1 << 37
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
    off, o = dict.fromkeys(NAMES, 0), 0
    for k in names:
        off[k] = o
        o += size.get(k, ns)
    for k, a in alias.items():
        off[k] = off[a]
    return off, o
