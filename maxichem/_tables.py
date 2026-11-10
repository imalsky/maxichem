"""The call kernels' tables from a network: species order, rate laws, term streams and LU pattern."""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ._constants import NO_SLOT, RATE_LAW_FLAGS, RATE_LAW_SHIFTS
from .kinvulcan import VulcanNetwork, _check_third_body

MAX_SPECIES = 254  # species slots are uint8, with one value left for the padding
NTAIL = 32  # rows of the trailing block factored in registers (at most 32)


def warps_per_cell(fit: int) -> int:
    """Return the warps per cell for a network whose per-cell arrays fit fit times in shared memory.

    4 when fit <= 3 (at 4 warps x 32 lanes and 160-190 registers per thread the register file already caps a block at
    2-3 cells, so the extra lanes cost no cells), else 2.
    """
    return 4 if fit <= 3 else 2


def _order(pattern: Iterable[tuple[int, int]], n: int) -> list[int]:
    """Greedy minimum-degree elimination order on the symmetrized pattern (ties by index)."""
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in pattern:
        if i != j:
            adj[i].add(j)
            adj[j].add(i)
    left, order = set(range(n)), []
    while left:
        k = min(left, key=lambda q: (len(adj[q] & left), q))
        order.append(k)
        nb = adj[k] & left
        for a in nb:
            adj[a] |= nb - {a}
        left.remove(k)
    return order


def balance_lanes(items: Iterable[int], size: Callable[[int], int], nl: int) -> tuple[list[list[int]], int]:
    """Items spread over nl lanes, longest first onto the least-loaded lane: (items per lane, rounds)."""
    lanes: list[list[int]] = [[] for _ in range(nl)]
    load = np.zeros(nl, int)
    for it in sorted(items, key=lambda q: -size(q)):
        l_ = int(np.argmin(load))
        lanes[l_].append(it)
        load[l_] += size(it)
    return lanes, int(load.max())


def _bits(n: int) -> int:
    """Bits that hold the values 0..n."""
    return max(1, int(n).bit_length())


def _law(r: dict[str, Any]) -> tuple[list[float], int]:
    """Return a rate law's kernel row and flags (``RATE_LAW_FLAGS``).

    The row: ln A0, B0, C0, ln A1, B1, C1 (second k0 term), ln A_inf, B_inf, C_inf, Troe a, T3, T1, T2, efficiency.
    """

    def ln(a: float) -> float:
        return np.log(a) if a > 0 else -np.inf

    if r["kind"] == 2:
        a0, a1, ai = (r["A"], r["B"], r["C"]), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    else:
        a0 = (r["A0"], r["B0"], r["C0"])
        a1 = (r["A1"], r["B1"], r["C1"]) if "A1" in r else (0.0, 0.0, 0.0)
        ai = (r["Ai"], r["Bi"], r["Ci"]) if r["kind"] in (3, 6) else (1.0, 0.0, 0.0)
    if min(a0[0], a1[0], ai[0]) < 0:
        raise ValueError(f"negative Arrhenius prefactor in {r['eq']}: the kernel takes ln A")
    row = [
        ln(a0[0]),
        a0[1],
        a0[2],
        ln(a1[0]),
        a1[1],
        a1[2],
        ln(ai[0]),
        ai[1],
        ai[2],
        *r.get("troe", (0.0, 1.0, 1.0, 0.0)),
        r.get("eff", 1.0),
    ]
    fl = (
        int(r["kind"] in (3, 6)) * RATE_LAW_FLAGS["FL_FALLOFF"]
        | int("A1" in r) * RATE_LAW_FLAGS["FL_SECOND_K0"]
        | int("troe" in r) * RATE_LAW_FLAGS["FL_TROE"]
        | int("eff" in r) * RATE_LAW_FLAGS["FL_EFF"]
        | int(r.get("width") == "vm11") * RATE_LAW_FLAGS["FL_VM11"]
    )
    return row, fl


def _rate_derivative_terms(
    reactants: list[list[int]], products: list[list[int]], live: list[int], n_reactions: int, n_thermal: int
) -> dict[tuple[int, int], list[Any]]:
    """Return d v_r / d n_q: (sign, rate index (kf r, kr nr + r), other species) by (r, q)."""
    terms: dict[tuple[int, int], list[Any]] = {}
    for r in live:
        sides = ((reactants[r], r, 1.0), (products[r], n_reactions + r, -1.0))
        for side, k, sgn in sides[: 1 if r >= n_thermal else 2]:
            for m in range(len(side)):
                terms.setdefault((r, side[m]), []).append((sgn, k, side[:m] + side[m + 1 :]))
    return terms


def _jacobian_entries(rate_terms: dict[tuple[int, int], list[Any]], S: np.ndarray) -> dict[tuple[int, int], list[Any]]:
    """Return each entry (p, q) of Jn: a list of (S[p, r], (r, q))."""
    entries: dict[tuple[int, int], list[Any]] = {}
    for r, q in rate_terms:
        for p in np.flatnonzero(S[:, r]):
            entries.setdefault((int(p), q), []).append((S[p, r], (r, q)))
    return entries


def _widest_sides(rx: np.ndarray, ns: int, lanes: int) -> list[list[int]]:
    """Return the widest forward and the widest reverse side of each round of reactions."""
    widths = [(rx[:, o : o + 3] != ns).sum(1) for o in (0, 3)]
    return [[int(q[c : c + lanes].max()) for c in range(0, len(rx), lanes)] for q in widths]


def _symbolic_lu(
    entries: set[tuple[int, int]], ns: int
) -> tuple[list[int], dict[int, int], set[tuple[int, int]], list[list[tuple[int, int]]]]:
    """Return the elimination order, the species' positions, the pattern after fill and each pivot's updates."""
    perm = _order(entries, ns)
    pos = {s: k for k, s in enumerate(perm)}
    pattern = {(pos[i], pos[j]) for i, j in entries}
    ops: list[list[tuple[int, int]]] = [[] for _ in range(ns)]  # without exchanges: ops[k] = (i, j) updated by pivot k
    for k in range(ns):
        rows = sorted(i for (i, j) in pattern if j == k and i > k)
        cols = sorted(j for (i, j) in pattern if i == k and j > k)
        for i in rows:
            for j in cols:
                pattern.add((i, j))
                ops[k].append((i, j))
    return perm, pos, pattern, ops


def _slots(
    pattern: set[tuple[int, int]], k0: int, ns: int
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]], int]:
    """Return the slot of each (row, column), the leading entries and the first slot of the trailing block."""
    nt = ns - k0
    lead = sorted(e for e in pattern if not (e[0] >= k0 and e[1] >= k0))  # row-major, then the trailing block row-major
    slot = {e: n for n, e in enumerate(lead)}
    tb0 = len(lead)
    for i in range(k0, ns):
        for j in range(k0, ns):
            slot[(i, j)] = tb0 + (i - k0) * nt + j - k0
    return slot, lead, tb0


def _reaction_tables(
    reac: list[dict[str, Any]],
    reactants: list[list[int]],
    products: list[list[int]],
    pos: dict[int, int],
    ns: int,
    n_thermal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return the reactions' species slots (padding ns: n[ns] = 1, g[ns] = 0) and rate laws."""
    nr = len(reac)
    rx = np.full((nr, 6), ns, np.uint8)
    rc = np.zeros((nr, 14), np.float64)
    rf = np.zeros(nr, np.int32)
    for j, r in enumerate(reac):
        if "rev" in r:
            raise ValueError(
                f"reaction {r.get('eq', j)} has an explicit reverse law, which would be ignored: "
                "every reverse comes from NASA-9; remove its 'rev' entry"
            )
        rx[j, : len(reactants[j])] = [pos[q] for q in reactants[j]]
        if j >= n_thermal:  # photolysis: no reverse side, no law (kf = J, kr = 0 in the kernel)
            continue
        _check_third_body(r["eq"], r["kind"], f"reaction {j}")  # also a reaction built or changed by hand
        rx[j, 3 : 3 + len(products[j])] = [pos[q] for q in products[j]]
        rc[j], rf[j] = _law(r)
        rf[j] |= (
            (r["kind"] in (3, 5, 6)) * RATE_LAW_FLAGS["FL_THREE_BODY"]
            | (len(products[j]) - len(reactants[j]) + 4) << RATE_LAW_SHIFTS["FL_DMU_SHIFT"]
            | len(reactants[j]) << RATE_LAW_SHIFTS["FL_FORWARD_SHIFT"]
            | len(products[j]) << RATE_LAW_SHIFTS["FL_REVERSE_SHIFT"]
        )
    law = sorted({"troe" for r in reac if "troe" in r} | {"eff" for r in reac if "eff" in r})
    if not law:
        rc = rc[:, :9]
    return rx, rc, rf, law


def _sv_stream(
    stoichiometry: np.ndarray, perm: list[int], live: list[int], lanes: int, widths: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the S v lane stream, each row's extra pieces and the number of pieces; adds the piece's width to widths.

    Each row's terms are cut into pieces of at most an even share; its first piece writes part[i], the others
    part[ns + e]. A word holds the reaction, sign, x2, destination piece and end of piece.
    """
    ns = len(perm)
    rows = [[(r, stoichiometry[perm[i], r]) for r in live if stoichiometry[perm[i], r] != 0] for i in range(ns)]
    share = -(-sum(len(t) for t in rows) // lanes)
    pieces, xp, nx = [], [0], 0
    for i, t in enumerate(rows):
        for c in range(0, len(t), share):
            pieces.append((i if c == 0 else ns + nx, t[c : c + share]))
            nx += c > 0
        xp.append(nx)
    npart = ns + nx
    if nx > 255:
        raise ValueError(f"too many S v pieces: {nx}, at most 255; the reaction terms are too unevenly spread")
    widths["part"] = _bits(npart - 1)
    if widths["r"] + widths["part"] + 4 > 32 or widths["k"] + 2 * widths["s"] + 3 > 32 or 3 * widths["slot"] > 64:
        raise ValueError(f"descriptor widths {widths} do not fit their words: the network is too large")
    assigned, rounds = balance_lanes(range(len(pieces)), lambda q: len(pieces[q][1]), lanes)
    sst = np.zeros((rounds, lanes), np.uint32)
    o_dst = widths["r"] + 2
    for lane, its in enumerate(assigned):
        q = 0
        for it in its:
            dst, terms = pieces[it]
            for e, (r, c) in enumerate(terms):
                sst[q, lane] = (
                    r
                    | int(c < 0) << widths["r"]
                    | int(abs(c) == 2) << (widths["r"] + 1)
                    | dst << o_dst
                    | int(e == len(terms) - 1) << (o_dst + widths["part"])
                    | 1 << 31
                )
                q += 1
    return sst, np.array(xp, np.uint8), npart


def _jacobian_terms(
    entries: dict[tuple[int, int], list[Any]],
    rate_terms: dict[tuple[int, int], list[Any]],
    slot: dict[tuple[int, int], int],
    pos: dict[int, int],
    ns: int,
    nnz: int,
) -> list[list[Any]]:
    """Return each slot's terms: (rate index, coefficient, species a, species b), for coefficient * k * n_a * n_b."""
    terms: list[list[Any]] = [[] for _ in range(nnz)]
    for (p, q), lst in entries.items():
        s = slot[(pos[p], pos[q])]
        for c, (r, q_) in lst:
            for sgn, k, rest in rate_terms[(r, q_)]:
                ab = [pos[o] for o in rest] + [ns] * (2 - len(rest))
                terms[s].append((k, c * sgn, ab[0], ab[1]))
    return terms


def _cut_slots(terms: list[list[Any]], share: int) -> tuple[list[Any], list[int]]:
    """Cut each slot's terms into pieces of at most share: the pieces (destination, terms), each extra piece's slot.

    A destination at or above the number of slots means Y[destination - slots].
    """
    pieces: list[Any] = []
    extra: list[int] = []
    for s, slot_terms in enumerate(terms):
        for c in range(0, len(slot_terms), share):
            pieces.append((s if c == 0 else len(terms) + len(extra), slot_terms[c : c + share]))
            if c:
                extra.append(s)
    return pieces, extra


def _jacobian_stream(
    terms: list[list[Any]], nr: int, ns: int, lanes: int, widths: dict[str, int]
) -> tuple[np.ndarray, int, int, int]:
    """Return the Jacobian's lane stream as packed words, with its rounds, the slots per lane and the extra pieces.

    A word holds the rate, sign, x2, species a, b and end of slot; padding words take the zero rate at index 2 nr. A
    slot longer than an even share is cut into pieces: the first writes the slot, the others Y[e] (free during the
    Jacobian), added to the slot afterwards in order. Y holds ns doubles, so the share grows until the extra pieces fit:
    a network with few terms per lane gets longer rounds, not a refusal. Each lane's finished slots follow the words,
    then each extra piece's slot.
    """
    nnz = len(terms)
    share = -(-sum(len(t) for t in terms) // lanes)
    pieces, extra = _cut_slots(terms, share)
    while len(extra) > ns:
        share += 1
        pieces, extra = _cut_slots(terms, share)
    if nnz + len(extra) > NO_SLOT:
        raise ValueError(
            f"too many Jacobian slots: {nnz} and {len(extra)} extra pieces, at most {NO_SLOT}; the kernels' slots are "
            "uint16, so this network is too large"
        )
    filled = [q for q in range(len(pieces)) if pieces[q][1]]
    assigned, rounds = balance_lanes(filled, lambda q: len(pieces[q][1]), lanes)
    o_a = widths["k"] + 2
    o_end = o_a + 2 * widths["s"]
    words = np.zeros((rounds, lanes), np.uint32)
    for lane, its in enumerate(assigned):
        q = 0
        for it in its:
            piece = pieces[it][1]
            for e, (k, c, a, b) in enumerate(piece):
                words[q, lane] = (
                    int(k)
                    | int(c < 0) << widths["k"]
                    | int(abs(c) == 2) << (widths["k"] + 1)
                    | a << o_a
                    | b << (o_a + widths["s"])
                    | int(e == len(piece) - 1) << o_end
                )
                q += 1
        words[q:, lane] = 2 * nr | ns << o_a | ns << (o_a + widths["s"])
    finished = [[pieces[it][0] for it in its] for its in assigned]
    most = max(len(its) for its in finished)
    per_lane = np.zeros((lanes, most), np.uint16)
    for lane, its in enumerate(finished):
        per_lane[lane, : len(its)] = its
    packed = words.tobytes() + per_lane.tobytes() + np.array(extra, np.uint16).tobytes()
    return np.frombuffer(packed + b"\0" * (-len(packed) % 8), np.uint64), rounds, most, len(extra)


def _lu_stream(
    slot: dict[tuple[int, int], int],
    ops: list[list[tuple[int, int]]],
    pattern: set[tuple[int, int]],
    k0: int,
    widths: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the LU updates of the sparse leading part, each pivot's first update and the pivot batches.

    Updates of pivots below k0 reach the trailing block too. A batch is a run of consecutive mutually independent
    pivots; blen[k] is the length of the batch starting at pivot k, else 0.
    """
    lu = [
        (slot[(i, j)] | slot[(i, k)] << widths["slot"] | slot[(k, j)] << 2 * widths["slot"])
        for k in range(k0)
        for (i, j) in ops[k]
    ]
    optr = np.cumsum([0] + [len(ops[k]) for k in range(k0)]).astype(np.int32)
    blen, cur = np.zeros(k0, np.uint8), None
    for k in range(k0):
        if cur is None or blen[cur] == 255 or any((j, k) in pattern or (k, j) in pattern for j in range(cur, k)):
            cur = k
        blen[cur] += 1
    return np.array(lu, np.uint64), optr, blen


def _slot_lookups(slot: dict[tuple[int, int], int], lead: list[tuple[int, int]], ns: int) -> dict[str, np.ndarray]:
    """Return the tables that lead from a (row, column) to its slot and back."""
    smap = np.full((ns, ns), NO_SLOT, np.uint16)
    didx = np.zeros(len(slot), np.uint16)  # dense position of each slot (recovery)
    for (i, j), s in slot.items():
        smap[i, j] = s
        didx[s] = i * ns + j
    return {
        "smap": smap,
        "rowptr": np.searchsorted([i for (i, j) in lead], np.arange(ns + 1)).astype(np.uint16),
        "colof": np.array([j for (i, j) in lead], np.uint8),
        "diag": np.array([slot[(i, i)] for i in range(ns)], np.uint16),
        "didx": didx,
    }


def tables(net: VulcanNetwork, warps: int, photo: bool = False) -> dict[str, Any]:
    """Build the kernels' tables: species order (perm[i] = network index of row i), rate laws, streams, LU pattern.

    Warps per cell at least 2. With photo, the network's photolysis branches follow the thermal reactions: irreversible,
    their forward rate coefficient the call's per-cell input J.
    """
    reac = net.reac + (net.photo if photo else [])
    S = np.asarray(net.S, np.float64)
    if photo:
        S = np.hstack([S, np.asarray(net.Sp, np.float64)])
    ns, nr = S.shape
    nrt = len(net.reac)
    lanes = 32 * warps
    reactants = [list(r["re"]) for r in reac]
    products = [list(r["pr"]) for r in reac]
    if ns > MAX_SPECIES or max(len(q) for q in reactants + products) > 3 or np.abs(S).max() > 2:
        raise ValueError(
            f"the kernels hold at most {MAX_SPECIES} species, 3 per side and |S| <= 2; this has {ns} species"
        )
    live = [r for r in range(nr) if np.any(S[:, r] != 0)]
    rate_terms = _rate_derivative_terms(reactants, products, live, nr, nrt)
    entries = _jacobian_entries(rate_terms, S)
    perm, pos, pattern, ops = _symbolic_lu(set(entries) | {(i, i) for i in range(ns)}, ns)
    nt = min(NTAIL, ns)
    k0 = ns - nt  # rows < k0: sparse leading part; rows >= k0: the trailing block
    pattern |= {(i, j) for i in range(k0, ns) for j in range(k0, ns)}
    slot, lead, tb0 = _slots(pattern, k0, ns)
    nnz = len(slot)
    rx, rc, rf, law = _reaction_tables(reac, reactants, products, pos, ns, nrt)
    # descriptor widths: reaction, rate (kf, kr, the zero rate 2 nr), species (padding ns), LU slot, S v piece
    widths = {"r": _bits(nr - 1), "k": _bits(2 * nr), "s": _bits(ns), "slot": _bits(nnz - 1)}
    sst, xp, npart = _sv_stream(S, perm, live, lanes, widths)
    terms = _jacobian_terms(entries, rate_terms, slot, pos, ns, nnz)
    jt, jround, jsmax, nfix = _jacobian_stream(terms, nr, ns, lanes, widths)
    lu, optr, blen = _lu_stream(slot, ops, pattern, k0, widths)
    return {
        **_slot_lookups(slot, lead, ns),
        "ns": ns,
        "nr": nr,
        "nrt": nrt,
        "nj": nr - nrt,
        "nnz": nnz,
        "k0": k0,
        "tb0": tb0,
        "warps": warps,
        "pattern": pattern,
        "perm": np.array(perm, np.int32),
        "rx": rx,
        "rc": rc,
        "rf": rf,
        "law": law,
        "nasa": np.concatenate([np.asarray(net.lo, np.float64)[perm], np.asarray(net.hi, np.float64)[perm]], 0),
        "npart": npart,
        "xp": xp,
        "sst": sst,
        "jt": jt,
        "jround": jround,
        "jsmax": jsmax,
        "nfix": nfix,
        "lu": lu,
        "optr": optr,
        "blen": blen,
        "vs": _widest_sides(rx, ns, lanes),
        "atoms": np.asarray(net.atoms, np.float64),
        "widths": widths,
    }


def tables_for_device(net: VulcanNetwork, photo: bool, shared_memory: int) -> dict[str, Any]:
    """Build the tables with the warps per cell that fit a device with this shared memory per block (bytes)."""
    tab = tables(net, 2, photo)
    warps = warps_per_cell(cells_per_block(tab, shared_memory))
    if warps != 2:
        tab = tables(net, warps, photo)
    if cells_per_block(tab, shared_memory) < 1:
        blob = pack_small_tables(tab)[0]
        raise ValueError(
            f"one cell needs {8 * shared_doubles_per_cell(tab) + len(blob)} B of shared memory per block; "
            f"the device has {shared_memory}"
        )
    return tab


def cells_per_block(tab: dict[str, Any], shared_memory: int) -> int:
    """Return the cells per block the shared memory (bytes) holds."""
    return int((shared_memory - len(pack_small_tables(tab)[0])) // (8 * shared_doubles_per_cell(tab)))


def pack_small_tables(tab: dict[str, Any]) -> tuple[bytes, dict[str, int]]:
    """Return the small tables as one byte string for shared memory, and the byte offset of each."""
    blob, offs = b"", {}
    for name in ("smap", "rowptr", "diag", "colof", "rx", "xp", "blen"):
        blob += b"\0" * (-len(blob) % 2)
        offs[name] = len(blob)
        blob += np.ascontiguousarray(tab[name]).tobytes()
    return blob + b"\0" * (-len(blob) % 8), offs


def rates_fit_in_block(tab: dict[str, Any]) -> bool:
    """Say whether the rates (or G / RT) and the S v pieces fit in the trailing block's slots."""
    nt = tab["ns"] - tab["k0"]
    return bool(max(tab["nr"], tab["ns"] + 1) + tab["npart"] <= nt * nt)


def scratch_strides(tab: dict[str, Any], wsh: bool) -> tuple[int, int]:
    """Return the doubles of global scratch per cell slot of the call kernel and per block of the recover kernel.

    Each holds the rate coefficients (2 NR + 2) and the copy of F, u and -Jn kept for the try after a rejection; the
    recover kernel also W when it is not in shared memory.
    """
    ns, nr = tab["ns"], tab["nr"]
    call = 2 * nr + 2 + 2 * ns + tab["nnz"]
    return call, call + (0 if wsh else ns * ns)


def shared_doubles_per_cell(tab: dict[str, Any]) -> int:
    """Return the shared-memory doubles per cell.

    M (leading slots and the trailing block), dinv (and c0), n, N, Y, F, u, K1-K4, one for the queue's next cell; v and
    the S v pieces when they do not fit in the trailing block.
    """
    ns = tab["ns"]
    rates = 0 if rates_fit_in_block(tab) else max(tab["nr"], ns + 1) + tab["npart"]
    return int(tab["nnz"] + ns + 1 + ns + 1 + 8 * ns + 1 + rates)


def shared_doubles_recover(tab: dict[str, Any], wsh: bool) -> int:
    """Return the shared-memory doubles of the recover kernel's cell (PERWR).

    W when wsh (else in global memory), dinv, n, v (or the G / RT of the species), N, Y, F, u, K1-K4, the S v pieces,
    the row order of the pivoting and the next list entry (as ints).
    """
    ns = tab["ns"]
    return int((ns * ns if wsh else 0) + 10 * ns + 1 + max(tab["nr"], ns + 1) + tab["npart"] + (ns + 2) // 2)
