"""Fast test runner for the GPU chemistry solver against VULCAN truth on NCHO_photo; misses first.

The tiers and what they score are in docs/testing.md. Exit status 1 when a path or box has a miss, a call is unfinished
or leaves a non-finite or negative state, or a cell still fails after the recovery.

    python bench/kintest.py quick|full [--save out.npz] [--ref ref.npz] [--force]

--save writes per-path summaries, the 12 gases of every cell at intervals and at the end, the per-box results and
timings (it refuses to overwrite). --ref compares with a saved run of the same tier and the same truth.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from kintest_truth import FLOOR, HOLD, check_jobs, load_p, load_paths

from maxichem import kincuda, kinvulcan, paths

GASES = ["H2O", "CO", "CO2", "CH4", "NH3", "HCN", "C2H2", "H", "OH", "O", "H2", "N2"]
MISS, LARGE = 0.1, 1.0
DAY = 86400.0
READ = 1000  # ticks between read-backs of the counters
CHUNK = 65536  # most boxes per chunk
CAP_H = 5.0  # full: stop above this estimate (hours) unless --force
META = ("id", "panel", "kind", "stratum", "start", "comp")
KEYS_FB = (
    "bad_input",
    "nonfinite",
    "unfinished",
    "drift",
    "recomputed",
    "recovered",
    "still_failed",
)  # CudaSolver.fallback
NEVER = 1 << 62


class Block:
    """One panel on one schedule.

    One panel on one schedule: call j runs on tick j * stride and uses forcing row (j // rep) % cyc of each path (rep
    calls per 1500-s hold; cyc = the row count of a cycled panel); a path takes part in calls s0 <= j < e; a mix keyed
    to row r comes before call rep * r, a stored state of row r is the state after call rep * r + rep - 1.
    """

    def __init__(self, name, cells, stride, ncall, rep=1, dt=None, cyc=None):
        self.name, self.cells, self.stride, self.ncall, self.rep, self.dt = name, cells, stride, ncall, rep, dt
        self.cyc = cyc or NEVER
        self.n = len(cells)
        self.truth = cells[0]["truth"] is not None
        lengths = np.array([len(c["T"]) for c in cells])
        assert cyc is None or (cyc == lengths).all()
        self.s0 = np.array([rep * c["s0"] for c in cells])
        self.e = np.full(self.n, ncall) if cyc else np.minimum(rep * lengths, ncall)
        self.nrow = lengths if cyc else np.minimum(lengths, -(-ncall // rep))  # forcing rows used per path
        self.end = int(self.e.max())
        self.ev = np.unique(np.concatenate([self.s0, self.e]))
        self.next, self.nlive = -1, 0

    def setup(self, c0, off, dev):
        """Global cell numbers c0.., forcing offsets off; mixes by row and stored truth by call on the GPU."""
        self.c0, self.off = c0, off
        mr, mc, mf, my, sk, sc, st = [], [], [], [], [], [], []
        self.mixed = np.zeros(self.n, bool)
        for i, c in enumerate(self.cells):
            for r, f, y in c["mix"]:
                if self.cyc < NEVER or self.rep * r < self.e[i]:
                    mr.append(r), mc.append(c0 + i), mf.append(f), my.append(y)
                    self.mixed[i] = True
            if self.truth:
                k = self.rep * c["rows"] + self.rep - 1
                keep = k < self.e[i]
                sk.append(k[keep]), sc.append(np.full(int(keep.sum()), c0 + i)), st.append(c["truth"][keep])
        nr = int(self.nrow.max())
        mr = np.asarray(mr, np.int64)
        o = np.argsort(mr, kind="stable")
        self.mb = np.searchsorted(mr[o], np.arange(nr + 1))
        self.mc = torch.as_tensor(np.asarray(mc, np.int64)[o], device=dev)
        self.mf = torch.as_tensor(np.asarray(mf, np.float64)[o], device=dev)[:, None]
        self.my = torch.as_tensor(np.asarray(my, np.float64).reshape(-1, len(self.cells[0]["x0"]))[o], device=dev)
        if self.truth:
            sk = np.concatenate(sk)
            o = np.argsort(sk, kind="stable")
            self.sb = np.searchsorted(sk[o], np.arange(self.ncall + 1))
            self.sc = torch.as_tensor(np.concatenate(sc)[o], device=dev)
            self.st = torch.as_tensor(np.concatenate(st)[o], device=dev)

    def live(self, j, dev):
        """Refresh the scheduled cells (global numbers, forcing offsets) when a path starts or ends at call j."""
        if j < self.next:
            return
        on = np.flatnonzero((self.s0 <= j) & (j < self.e))
        self.gidx = torch.as_tensor(self.c0 + on, device=dev)
        self.goff = torch.as_tensor(self.off[on], device=dev)
        self.nlive = len(on)
        later = self.ev[self.ev > j]
        self.next = int(later[0]) if len(later) else NEVER

    def mix(self, row, x_state):
        """Apply the mixes keyed to this forcing row, before the row's first call."""
        a, z = self.mb[row], self.mb[row + 1]
        if z > a:
            c, f = self.mc[a:z], self.mf[a:z]
            x_state[c] = (1.0 - f) * x_state[c] + f * self.my[a:z]


def score(x, c, truth, j, best, cnt, gi, fl):
    """Score the states x[c] against truth rows.

    Score the states x[c] against truth rows: per cell the worst error with its call j and gas (best), and the counts
    of stored states, misses and large misses (cnt).
    """
    lx = torch.log10(x[c][:, gi].clamp(min=1e-300)).maximum(fl)
    e = (lx - truth).abs()
    ep, g = e.max(1)
    ep = torch.where(torch.isfinite(lx).all(1) & torch.isfinite(truth).all(1), ep, torch.inf)  # bad truth: a failure
    w = best[0][c]
    up = ep > w
    best[0][c] = torch.where(up, ep, w)
    best[1][c] = torch.where(up, j, best[1][c])
    best[2][c] = torch.where(up, g, best[2][c])
    cnt.index_add_(0, c, torch.stack((torch.ones_like(ep), (ep > MISS).double(), (ep > LARGE).double()), 1))


def add_fb(fb, solver):
    """Add the guard's counts of the last call to fb."""
    for k, v in solver.fallback.items():
        fb[k] = fb.get(k, 0) + v


def _complete_box_set(sp):
    """Read the boxes' manifest and refuse a set that is not complete: (manifest, shards, first shard)."""
    man = json.loads((paths.BOXES / "manifest.json").read_text())
    shards = [paths.BOXES / f for f in man.get("shards", [])]
    bad = {k: len(man.get(k, [])) for k in ("errors", "bad_states", "drift_over_1e-6")}
    ids = np.concatenate([np.load(f)["box"] for f in shards]) if shards else np.zeros(0, int)
    cover = np.zeros(man["boxes"], int)
    np.add.at(cover, np.concatenate([ids, np.asarray(man.get("quarantine", []), int)]), 1)
    if man.get("pending", 1) or any(bad.values()) or len(ids) != man["accepted"] or (cover != 1).any():
        raise SystemExit(
            f"{paths.BOXES}: not complete (manifest: {man.get('pending')} boxes not run, {bad}, "
            f"{man['accepted']} accepted; shards hold {len(ids)} boxes, design boxes not held exactly once "
            f"(accepted or quarantined) {int((cover != 1).sum())})"
        )
    print(f"boxes: {man['accepted']:,} of {man['boxes']:,} accepted, {len(man['quarantine'])} quarantined (left out)")
    z0 = np.load(shards[0])
    for f in shards:
        z = np.load(f)
        if (
            [str(q) for q in z["species"]] != sp
            or not np.array_equal(z["stops"], z0["stops"])
            or not np.array_equal(z["lines"], z0["lines"])
        ):
            raise SystemExit(f"{f}: species, stops or photolysis lines differ from the paths' and the other shards'")
    return man, shards, z0


def boxes(solver, net, sp, gh, gi, fl, cap, dev):
    """Run the VULCAN truth boxes.

    The VULCAN truth boxes (paths.BOXES, made by the truth generator's sobol.py: constant T, P and photolysis rates,
    stored at fixed stops from 0 to 1e8 s): the first cap boxes (None: all), each from its stored start in chained
    calls whose lengths are the stop intervals, every stored state scored (gh: the scored gases' columns, gi: the
    same on the device). Refuses a set whose manifest lists boxes not run, failed runs, bad states or element drift, or
    whose shards do not hold the accepted count; reports the quarantined boxes (a step forced at the generator's dt_min,
    left out). b_set: a digest of the scored truth and inputs. Returns the per-box results (b_*).
    """
    man, shards, z0 = _complete_box_set(sp)
    stops, lines = z0["stops"], [str(q) for q in z0["lines"]]
    col = [lines.index(f"{q['absorber']},{q['branch']}") for q in net.photo]
    keys = ("box", "X", "T", "P", "J", "lit", "kind")
    bset = {k: [] for k in keys}
    n = 0
    for f in shards:
        if cap is not None and n >= cap:
            break
        z = np.load(f)
        for k in keys:
            bset[k].append(z[k])
        n += len(z["box"])
    bset = {k: np.concatenate(v)[:cap] for k, v in bset.items()}
    n_boxes, nc = len(bset["box"]), len(stops) - 1
    bres = {
        "b_box": bset["box"],
        "b_calls": np.full(n_boxes, nc),
        "b_worst": np.zeros(n_boxes),
        "b_kworst": np.full(n_boxes, -1),
        "b_gworst": np.full(n_boxes, -1),
        "b_saved": np.zeros(n_boxes),
        "b_nmiss": np.zeros(n_boxes),
        "b_nlarge": np.zeros(n_boxes),
        "b_unfinished": np.zeros(n_boxes),
        "b_nonfinite": np.zeros(n_boxes),
        "b_steps": np.zeros(n_boxes),
        "b_lx": np.zeros((n_boxes, 12), np.float32),
    }
    fb = {}
    t0 = time.time()

    def tt(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)

    for p0 in range(0, n_boxes, CHUNK):
        sl = slice(p0, min(n_boxes, p0 + CHUNK))
        m = sl.stop - p0
        x, T, P, J = tt(bset["X"][sl, 0]), tt(bset["T"][sl]), tt(bset["P"][sl]), tt(bset["J"][sl][:, col])
        h = torch.full_like(T, 1e-10)
        acc = torch.zeros(m, 3, dtype=torch.float64, device=dev)  # unfinished, non-finite, steps
        cnt = torch.zeros(m, 3, dtype=torch.float64, device=dev)  # saved states, misses, large misses
        best = (
            torch.zeros(m, dtype=torch.float64, device=dev),
            torch.full((m,), -1, device=dev),
            torch.full((m,), -1, device=dev),
        )
        c = torch.arange(m, device=dev)
        for k in range(1, nc + 1):
            x, h, done, nok = solver(x, T, P, torch.full_like(T, stops[k] - stops[k - 1]), h, J=J)
            add_fb(fb, solver)
            bad = torch.logical_not(torch.isfinite(x).all(1))
            acc += torch.stack((torch.logical_not(done).double(), bad.double(), nok), 1)
            xt = tt(bset["X"][sl, k][:, gh])
            truth = torch.where(torch.isfinite(xt), torch.log10(xt.clamp(min=1e-300)).maximum(fl), torch.nan)
            score(x, c, truth, k, best, cnt, gi, fl)
        bres["b_worst"][sl], bres["b_kworst"][sl], bres["b_gworst"][sl] = (q.cpu().numpy() for q in best)
        bres["b_saved"][sl], bres["b_nmiss"][sl], bres["b_nlarge"][sl] = cnt.cpu().numpy().T
        bres["b_unfinished"][sl], bres["b_nonfinite"][sl], bres["b_steps"][sl] = acc.cpu().numpy().T
        bres["b_lx"][sl] = torch.log10(x[:, gi].clamp(min=1e-300)).maximum(fl).float().cpu().numpy()
    torch.cuda.synchronize()
    bres["b_s"], bres["b_cellcalls"] = time.time() - t0, n_boxes * nc
    h = hashlib.sha256(json.dumps(man["stamp"], sort_keys=True).encode())
    for a in [bset[k] for k in keys] + [stops, np.array(lines), np.array(sp)]:
        h.update(np.ascontiguousarray(a).tobytes())
    bres["b_set"] = np.array(f"{man['network']} rtol {man.get('rtol')} {n_boxes} boxes sha256 {h.hexdigest()[:16]}")
    bres["fallback_boxes"] = np.array([fb.get(q, 0) for q in KEYS_FB])
    box_report(bres, bset, stops)
    return bres


def box_report(bres, bset, stops):
    """Misses-first lines of the box pass."""
    w = bres["b_worst"]
    print(
        f"boxes: {len(w):,} boxes x {len(stops) - 1} calls in {bres['b_s']:.0f} s "
        f"({1e6 * bres['b_s'] / bres['b_cellcalls']:.3f} us per cell-call), steps per call "
        f"{bres['b_steps'].sum() / bres['b_calls'].sum():.3f}; guard "
        f"{dict(zip(KEYS_FB, bres['fallback_boxes'].tolist(), strict=True))}"
    )

    def line(name, s):
        if s.any():
            print(
                f"  {name:20s} {int(s.sum()):7,d} boxes: miss {100 * (w[s] > MISS).mean():6.3f} %  large "
                f"{100 * (w[s] > LARGE).mean():6.3f} %  worst {w[s].max():.3f} | stored states "
                f"{int(bres['b_saved'][s].sum()):,}: miss {int(bres['b_nmiss'][s].sum()):,} large "
                f"{int(bres['b_nlarge'][s].sum()):,} | boxes with an unfinished call "
                f"{int((bres['b_unfinished'][s] > 0).sum())}"
                f", with a non-finite state {int((bres['b_nonfinite'][s] > 0).sum())}"
            )

    line("all boxes", np.ones(len(w), bool))
    line("dark", bset["lit"] == 0)
    line("lit", bset["lit"] == 1)
    for k, name in enumerate(("unreacted", "reheat", "mixed", "scrambled", "hot-H")):
        line(f"start {name}", bset["kind"] == k)
    for lo, hi in ((300, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 3001)):
        line(f"T {lo}-{hi} K", (bset["T"] >= lo) & (bset["T"] < hi))
    print("  worst boxes:")
    for i in np.argsort(-w)[:5]:
        print(
            f"    box {bset['box'][i]} T {bset['T'][i]:.0f} K P {bset['P'][i] / 1e6:.1e} bar "
            f"{'lit' if bset['lit'][i] else 'dark'} "
            f"start {int(bset['kind'][i])}: {w[i]:.3f} dex in {GASES[bres['b_gworst'][i]]} at stop "
            f"{stops[bres['b_kworst'][i]]:.3g} s"
        )


def make_blocks(tier, sp, gi):
    """Return the tier's blocks (see the module docstring)."""
    t0 = time.time()
    qc = load_paths(("QC", "QCs", "twins/QC"), sp, gi, hold=False)
    qh = load_paths(("QH", "QHs", "twins/QH"), sp, gi, hold=True)
    ql = load_paths(("QL",), sp, gi, hold=True)
    qlc = load_paths(("QLc",), sp, gi, hold=False)
    pp = load_p(sp)
    nqc, nh, nl, nlc, npp = (max(len(c["T"]) for c in v) for v in (qc, qh, ql, qlc, pp))
    print(f"truth and forcing loaded in {time.time() - t0:.0f} s", flush=True)
    if tier == "quick":
        return [
            Block("QC family, per call", qc, 1, nqc),
            Block("QH family, 1500-s holds", qh, 1, nh, dt=HOLD),
            Block("QL, dt 500 s", ql, 1, 10000, rep=3, dt=500.0),
            Block("QLc, dt 500 s", qlc, 1, 10000),
            *[Block(f"P, dt {d:g} s", pp, 1, npp, dt=d, cyc=npp) for d in (25.0, 50.0, 500.0)],
        ]
    return [
        Block("QL, dt 25 s", ql, 1, 60 * nl, rep=60, dt=25.0),
        Block("QL, dt 500 s", ql, 20, 3 * nl, rep=3, dt=500.0),
        Block("QLc, dt 500 s", qlc, 20, nlc),
        Block("QH family, dt 50 s", qh, 2, 30 * nh, rep=30, dt=50.0),
        Block("QC family, per call", qc, 1, nqc),
        Block("QH family, 1500-s holds", qh, 1, nh, dt=HOLD),
        *[
            Block(f"P, dt {d:g} s", pp, s, int(1000 * DAY / d), dt=d, cyc=npp)
            for d, s in ((25.0, 1), (50.0, 2), (500.0, 20))
        ],
    ]


def assemble(blocks, net, dev):
    """Global state, flat forcing (T, P in dyn cm^-2, dt per forcing row) and per-cell counters for all blocks."""
    x_start, t_forcing, p_forcing, dt_forcing, off0 = [], [], [], [], 0
    meta = {q: [] for q in META}
    c0 = 0
    for b in blocks:
        off = np.zeros(b.n, np.int64)
        for i, c in enumerate(b.cells):
            m = int(b.nrow[i])
            off[i] = off0
            off0 += m
            t_forcing.append(c["T"][:m]), p_forcing.append(c["P"][:m] * 1e6)
            dt_forcing.append(c["dt"][:m] if b.dt is None else np.full(m, b.dt))
            x_start.append(c["x0"] / c["x0"].sum())
            for q in META:
                meta[q].append(c["meta"][q])
        b.setup(c0, off, dev)
        c0 += b.n

    def tt(a):
        return torch.as_tensor(np.concatenate(a), dtype=torch.float64, device=dev)

    x_state = torch.as_tensor(np.stack(x_start), device=dev)
    atoms = torch.as_tensor(net.atoms, device=dev)
    i_h = net.elements.index("H")
    el = x_state @ atoms
    N = len(x_start)

    def z(*s, dt=torch.float64):
        return torch.zeros(*s, dtype=dt, device=dev)

    return {
        "X": x_state,
        "H": torch.full((N,), 1e-10, dtype=torch.float64, device=dev),
        "T": tt(t_forcing),
        "P": tt(p_forcing),
        "dt": tt(dt_forcing),
        "host": (np.concatenate(t_forcing), np.concatenate(p_forcing), np.concatenate(dt_forcing)),
        "atoms": atoms,
        "iH": i_h,
        "ratio0": el / el[:, i_h : i_h + 1],
        "meta": {q: np.array(v) for q, v in meta.items()},
        "acc": z(N, 5),  # calls, accepted steps, unfinished, non-finite, negative
        "cnt": z(N, 3),  # stored states, misses, large misses
        "best": (z(N), torch.full((N,), -1, device=dev), torch.full((N,), -1, device=dev)),
        "drift": z(N),
        "N": N,
    }


def _print_tick(t, ticks, elapsed, since_last, cells, w, tot, fb):
    """Print the loop's progress after tick t."""
    print(
        f"tick {t + 1:,} of {ticks:,}: {elapsed:.0f} s, {1e3 * since_last / ((t % READ) + 1):.2f} ms per "
        f"tick (last {(t % READ) + 1:,}), {cells} cells this tick, "
        f"paths with a miss "
        f"{int((w > MISS).sum())}, worst {float(w.max()):.3f} dex, unfinished {tot[2]:.0f}, non-finite "
        f"{tot[3]:.0f}, negative {tot[4]:.0f}, recomputed by the guard {fb.get('recomputed', 0)}",
        flush=True,
    )


def _stop_if_too_long(total, loop, ticks, force):
    """Print the full tier's estimate after the first READ ticks; stop above CAP_H hours unless forced."""
    est = total + (ticks - READ) * loop / READ
    print(
        f"estimate: {est / 3600:.2f} h in all ({total / 60:.1f} min so far, then {ticks - READ:,} "
        f"ticks x {1e3 * loop / READ:.2f} ms per tick of ticks 1-{READ:,})",
        flush=True,
    )
    if est > CAP_H * 3600 and not force:
        print(f"stopped: the estimate exceeds {CAP_H:g} h (--force runs it anyway)", flush=True)
        sys.exit(2)


def _schedule(blocks, t, x_state, dev):
    """Return the cells, forcing rows and blocks of tick t, after the mixes of its forcing rows."""
    ids, offs, sched = [], [], []
    for b in blocks:
        if t % b.stride:
            continue
        j = t // b.stride
        if j >= b.end:
            continue
        b.live(j, dev)
        if not b.nlive:
            continue
        row = (j // b.rep) % b.cyc
        if j % b.rep == 0:
            b.mix(row, x_state)
        ids.append(b.gidx), offs.append(b.goff + row), sched.append((b, j))
    return ids, offs, sched


def run(blocks, S, solver, ticks, snap, full, force, t_all, gi, fl, dev):
    """Run the tick loop; return timings, snapshots and the guard's counts."""
    x_state, h_state, acc, best, drift = S["X"], S["H"], S["acc"], S["best"], S["drift"]
    act, fb, snaps, snap_t, cellcalls = list(blocks), {}, [], [], 0

    def lx():
        return torch.log10(x_state[:, gi].clamp(min=1e-300)).maximum(fl).float().cpu().numpy()

    t0 = tl = time.time()
    for t in range(ticks):
        ids, offs, sched = _schedule(act, t, x_state, dev)
        if ids:
            idx = ids[0] if len(ids) == 1 else torch.cat(ids)
            fo = offs[0] if len(offs) == 1 else torch.cat(offs)
            dark = torch.zeros(fo.shape[0], solver.nj, dtype=torch.float64, device=dev)  # these paths run without light
            xo, ho, done, nok = solver(x_state[idx], S["T"][fo], S["P"][fo], S["dt"][fo], h_state[idx], J=dark)
            add_fb(fb, solver)
            x_state[idx], h_state[idx] = xo, ho
            cellcalls += idx.numel()
            fin = torch.isfinite(xo).all(1)
            acc.index_add_(
                0,
                idx,
                torch.stack(
                    (torch.ones_like(nok), nok, (~done).double(), (~fin).double(), (xo.min(1).values < 0).double()), 1
                ),
            )
            el = xo @ S["atoms"]
            d = (
                torch.log10(el / el[:, S["iH"] : S["iH"] + 1] / S["ratio0"][idx])
                .abs()
                .nan_to_num(nan=0.0)
                .max(1)
                .values
            )
            drift[idx] = torch.maximum(drift[idx], d)
            for b, j in sched:
                if b.truth:
                    a, z = b.sb[j], b.sb[j + 1]
                    if z > a:
                        score(x_state, b.sc[a:z], b.st[a:z], j, best, S["cnt"], gi, fl)
        if (t + 1) % READ == 0 or t + 1 == ticks:
            w = best[0]
            now = time.time()
            if not full or (t + 1) % (100 * READ) == 0 or t + 1 in (READ, ticks):
                _print_tick(t, ticks, now - t0, now - tl, sum(b.nlive for b, _ in sched), w, acc.sum(0).tolist(), fb)
            tl = now
            if (t + 1) % snap == 0:
                snaps.append(lx()), snap_t.append(t + 1)
            act = [b for b in act if (b.end - 1) * b.stride > t]
            if full and t + 1 == READ:
                _stop_if_too_long(now - t_all, now - t0, ticks, force)
    torch.cuda.synchronize()
    return {
        "loop_s": time.time() - t0,
        "ticks": ticks,
        "cellcalls": cellcalls,
        "snap_tick": np.array(snap_t),
        "snap_lx": np.stack(snaps) if snaps else np.zeros((0, S["N"], 12), np.float32),
        "final_lx": lx(),
        "fallback_loop": np.array([fb.get(q, 0) for q in KEYS_FB]),
    }


def results(blocks, S):
    """Per-cell counters and metadata as numpy arrays (the saved per-path summaries)."""
    acc, cnt = S["acc"].cpu().numpy(), S["cnt"].cpu().numpy()
    res = dict(
        block_names=np.array([b.name for b in blocks]),
        truth_block=np.array([b.truth for b in blocks]),
        block_of=np.concatenate([np.full(b.n, i) for i, b in enumerate(blocks)]),
        mixed=np.concatenate([b.mixed for b in blocks]),
        **S["meta"],
    )
    res["worst"], res["kworst"], res["gworst"] = (q.cpu().numpy() for q in S["best"])
    res["stored"], res["nmiss"], res["nlarge"] = cnt.T
    res["calls"], res["steps"], res["unfinished"], res["nonfinite"], res["negative"] = acc.T
    res["drift"] = S["drift"].cpu().numpy()
    return res


def line(name, s, res, truth):
    """One misses-first line for the paths s."""
    w, calls = res["worst"][s], res["calls"][s].sum()
    um = s & ~res["mixed"]
    dr = f"{res['drift'][um].max():.1e} dex" if um.any() else "-"
    head = f"  {name:10s} {int(s.sum()):5d} paths"
    if truth:
        head += (
            f": miss {100 * (w > MISS).mean():5.1f} %  large {100 * (w > LARGE).mean():5.1f} %  worst {w.max():.3f} | "
            f"stored states {int(res['stored'][s].sum()):,}: miss {int(res['nmiss'][s].sum())} large "
            f"{int(res['nlarge'][s].sum())}"
        )
    print(
        head + f" | calls {int(calls):,}: unfinished {int(res['unfinished'][s].sum())} non-finite "
        f"{int(res['nonfinite'][s].sum())} negative {int(res['negative'][s].sum())} | "
        f"element drift after the correction (unmixed) {dr} | "
        f"steps per call {res['steps'][s].sum() / max(calls, 1):.3f}"
    )


def report(blocks, res, lres, S):
    """Print the report, misses first.

    Misses first: totals over the truth paths, then per block (split by panel and kind; breakdowns by stratum,
    start and composition; the worst paths), then timings and the guard's counts.
    """
    tr = res["truth_block"][res["block_of"]]
    w = np.where(tr, res["worst"], -1.0)
    c = int(np.argmax(w))
    print(
        f"\nMISSES FIRST. qual truth paths {int(tr.sum()):,}: with a miss {int((w > MISS).sum())}, with a large miss "
        f"{int((w > LARGE).sum())}; stored states {int(res['stored'].sum()):,}: with a miss "
        f"{int(res['nmiss'].sum())}, large "
        f"{int(res['nlarge'].sum())}; worst {w[c]:.3f} dex "
        f"({res['block_names'][res['block_of'][c]]}, path {res['id'][c]}); "
        f"all blocks: unfinished calls {int(res['unfinished'].sum())}, "
        f"non-finite states {int(res['nonfinite'].sum())}, "
        f"negative {int(res['negative'].sum())}"
    )
    t_forcing, p_forcing, dt_forcing = S["host"]
    for bi, b in enumerate(blocks):
        s = res["block_of"] == bi
        print(f"{b.name}: {b.n} paths, up to {b.ncall:,} calls (every {b.stride} tick{'s' if b.stride > 1 else ''})")
        lab = np.char.add(np.char.add(res["panel"], " "), res["kind"])
        for v in dict.fromkeys(lab[s]):
            line(v, s & (lab == v), res, b.truth)
        if not b.truth:
            continue
        for key in ("stratum", "start", "comp"):
            parts = []
            for v in np.unique(res[key][s]):
                q = s & (res[key] == v)
                parts.append(
                    f"{v} {int(q.sum())} (miss {int((res['worst'][q] > MISS).sum())}, "
                    f"worst {res['worst'][q].max():.3f})"
                )
            print(f"    by {key}: " + "; ".join(parts))
        print("    worst paths:")
        for c in b.c0 + np.argsort(-res["worst"][s])[:5]:
            i, j = c - b.c0, int(res["kworst"][c])
            if j < 0:
                continue
            f = b.off[i] + (j // b.rep) % b.cyc
            seg = b.cells[i]["seg"]
            where = f" seg {seg[1][int(np.searchsorted(seg[0], j // b.rep, side='right')) - 1]}" if seg else ""
            print(
                f"      {res['id'][c]} {res['panel'][c]} {res['kind'][c]} {res['stratum'][c]} "
                f"{res['start'][c]} {res['comp'][c]}: "
                f"{res['worst'][c]:.3f} dex in {GASES[res['gworst'][c]]} at call {j}{where} (T {t_forcing[f]:.0f} K, P "
                f"{p_forcing[f] / 1e6:.2e} bar, dt {dt_forcing[f]:.1f} s); "
                f"stored states with a miss {int(res['nmiss'][c])}"
            )
    print(
        f"loop: {lres['ticks']:,} ticks in {lres['loop_s']:.0f} s = "
        f"{1e3 * lres['loop_s'] / lres['ticks']:.3f} ms per tick; "
        f"{lres['cellcalls']:,} cell-calls = {1e6 * lres['loop_s'] / lres['cellcalls']:.3f} us per cell-call; guard "
        f"{dict(zip(KEYS_FB, lres['fallback_loop'].tolist(), strict=True))}"
    )


def compare(new, ref):
    """Paired comparison of two runs of the same tier (mappings as written by --save)."""
    for k in ("tier", "id", "block_of", "b_box", "b_set", "q_set"):
        assert np.array_equal(np.asarray(new[k]), np.asarray(ref[k])), f"runs differ in {k}: not comparable"
    assert np.array_equal(new["snap_tick"], ref["snap_tick"])

    def lat(z):
        return (1e3 * float(z["loop_s"]) / int(z["ticks"]), 1e6 * float(z["b_s"]) / int(z["b_cellcalls"]))

    (m1, y1), (m0, y0) = lat(new), lat(ref)
    print(
        f"\nPAIRED with the reference (tier {new['tier']}): latency loop {m1:.3f} vs {m0:.3f} ms per tick "
        f"({m1 / m0:.2f}x), boxes {y1:.3f} vs {y0:.3f} us per cell-call ({y1 / y0:.2f}x)"
    )

    def spc(z, s):
        return z["steps"][s].sum() / max(z["calls"][s].sum(), 1)

    def dif(a, b):
        return np.nan_to_num(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)), nan=np.inf)

    ds = np.maximum(
        dif(new["final_lx"], ref["final_lx"]).max(1),
        dif(new["snap_lx"], ref["snap_lx"]).max((0, 2)) if len(new["snap_tick"]) else 0.0,
    )

    def row(name, s, truth, w1, w0, st, c1, c0, u1, u0, ids):
        txt = f"  {name:24s} {int(s.sum()):7,d} paths"
        if truth:
            dw = np.nan_to_num(w1[s] - w0[s], nan=np.inf, posinf=np.inf, neginf=-np.inf)
            i, k = int(np.argmax(dw)), int(np.argmin(dw))
            txt += (
                f" | worst error new - ref: mean {dw[np.isfinite(dw)].mean() if np.isfinite(dw).any() else 0:+.5f}, "
                f"largest rise {dw[i]:+.4f} ({ids[s][i]}), largest fall {dw[k]:+.4f} ({ids[s][k]}); |change| "
                f"> 0.01: {int((np.abs(dw) > 0.01).sum())}, > 0.03: {int((np.abs(dw) > 0.03).sum())}"
            )
        print(
            txt + f" | state change > 0.01: {int((st[s] > 0.01).sum())}, > 0.03: {int((st[s] > 0.03).sum())} (largest "
            f"{st[s].max():.2e}) | steps per call {c1:.3f} vs {c0:.3f} | unfinished {int(u1)} vs {int(u0)}"
        )

    for bi, name in enumerate(new["block_names"]):
        s = new["block_of"] == bi

        row(
            str(name),
            s,
            bool(new["truth_block"][bi]),
            new["worst"],
            ref["worst"],
            ds,
            spc(new, s),
            spc(ref, s),
            new["unfinished"][s].sum(),
            ref["unfinished"][s].sum(),
            new["id"],
        )
    bs = np.ones(len(new["b_box"]), bool)
    row(
        "boxes",
        bs,
        True,
        new["b_worst"],
        ref["b_worst"],
        dif(new["b_lx"], ref["b_lx"]).max(1),
        new["b_steps"].sum() / new["b_calls"].sum(),
        ref["b_steps"].sum() / ref["b_calls"].sum(),
        new["b_unfinished"].sum(),
        ref["b_unfinished"].sum(),
        new["b_box"],
    )


def main():
    """Run the tier, print the report and exit 1 on any miss or failed cell."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tier", choices=("quick", "full"))
    ap.add_argument("--save", help="write this npz (refuses to overwrite)")
    ap.add_argument("--ref", help="compare with this saved run of the same tier")
    ap.add_argument("--force", action="store_true", help="full: run even if the estimate exceeds 5 h")
    a = ap.parse_args()
    if a.save and Path(a.save).exists():
        sys.exit(f"{a.save} exists; not overwriting")
    paths.need(paths.QUAL / "QC", paths.DESIGNS / "decision/P", paths.BOXES, *([a.ref] if a.ref else []))
    check_jobs()
    ref = np.load(a.ref) if a.ref else None
    t_all = time.time()
    dev = torch.device("cuda")
    quick = a.tier == "quick"
    sp = [str(q) for q in np.load(min((paths.QUAL / "QC").glob("*.npz")))["species"]]
    gi = [sp.index(g) for g in GASES]
    net = kinvulcan.VulcanNetwork(
        f"{paths.VULCAN_THERMO}/NCHO_photo_network.txt", f"{paths.VULCAN_THERMO}/NASA9", species=sp
    )
    solver = kincuda.CudaSolver(net, dev, photo=True)
    gi_t, fl = torch.tensor(gi, device=dev), torch.tensor(FLOOR, device=dev)
    print(
        f"{a.tier}: {torch.cuda.get_device_name(dev)}, kernel ready ({solver.wpb} warps per block), "
        f"{time.time() - t_all:.0f} s",
        flush=True,
    )
    box_out = boxes(solver, net, sp, gi, gi_t, fl, 32768 if quick else None, dev)
    blocks = make_blocks(a.tier, sp, gi)
    qh = hashlib.sha256()  # the paths' truth and forcing, for paired runs
    for b in blocks:
        for c in b.cells:
            for v in (c["T"], c["P"], c["dt"], c["x0"], c["rows"], c["truth"]):
                if v is not None:
                    qh.update(np.ascontiguousarray(v).tobytes())
            for at, fr, y in c["mix"]:
                qh.update(np.array([at, fr]).tobytes() + np.ascontiguousarray(y).tobytes())
    S = assemble(blocks, net, dev)
    ticks = max(b.stride * (b.end - 1) + 1 for b in blocks)
    print(f"{len(blocks)} blocks, {S['N']:,} cells, {ticks:,} ticks; set up in {time.time() - t_all:.0f} s", flush=True)
    lres = run(blocks, S, solver, ticks, READ if quick else 10 * READ, not quick, a.force, t_all, gi_t, fl, dev)
    solver.check()  # raises if any cell was below 300 K
    res = results(blocks, S)
    report(blocks, res, lres, S)
    res.update(lres, **box_out, tier=np.array(a.tier), q_set=np.array(qh.hexdigest()[:16]))
    print(
        f"wall time {time.time() - t_all:.0f} s (boxes {box_out['b_s']:.0f} s, loop {lres['loop_s']:.0f} s)",
        flush=True,
    )
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.save, **res)
        print(f"saved {a.save}")
    if ref is not None:
        compare(res, ref)
    tr = res["truth_block"][res["block_of"]]
    fail = int(
        (np.where(tr, res["worst"], 0.0) > MISS).sum()
        + (box_out["b_worst"] > MISS).sum()
        + res["unfinished"].sum()
        + res["nonfinite"].sum()
        + res["negative"].sum()
        + box_out["b_unfinished"].sum()
        + box_out["b_nonfinite"].sum()
        + box_out["fallback_boxes"][KEYS_FB.index("still_failed")]
        + res["fallback_loop"][KEYS_FB.index("still_failed")]
    )
    if fail:
        sys.exit(
            f"FAIL: {fail} paths or boxes with a miss, unfinished calls, non-finite or negative states, or cell-calls "
            "still failed after the recovery"
        )


if __name__ == "__main__":
    main()
