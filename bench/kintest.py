"""Fast test runner for the GPU chemistry solver (maxichem/kincuda.py) on VULCAN-JAX's NCHO_photo network: the VULCAN truth
boxes (paths.BOXES), the VULCAN truth paths (paths.QUAL) and decision panel P in one process on one GPU, misses first.
The solver runs with photolysis (photo=True): the boxes with their J, the paths with J = 0.

Every panel is a block of cells with its own forcing (T, P, dt per call), mixes (x <- (1 - f) x + f y before a call),
kick-twin starts and VULCAN truth. One loop of ticks: a block's call j runs on tick j * stride; its paths that have not
started or have finished are left out; all scheduled cells go to the solver in one call per tick. Counters stay on the
GPU (read back every 1,000 ticks): per path the worst floored error against VULCAN over the 12 scored gases (floors
1e-10 H2O CO CO2 CH4 NH3 HCN C2H2, 1e-8 H OH O, 1e-6 H2 N2; a non-finite state is an infinite error) with its call and
gas, stored states with a miss (> 0.1 dex) / large miss (> 1 dex), unfinished calls, non-finite and negative states,
the largest change of an element ratio (to H) since the path's start (reported on paths without mixes), accepted
steps; the guard's counts of every call (CudaSolver.fallback).
  quick  the first 32,768 boxes (chained calls between their stored stops, 0 to 1e8 s); then the paths: QC + QCs + QC
         kick twins (T, P and dt every call, every state scored); QH + QHs + QH kick twins, one call per 1500-s hold;
         QL at dt 500 s (T, P held per hold, truth at hold ends) and QLc at dt 500 s; panel P's 256 paths cycled at dt
         25, 50 and 500 s for 10,000 calls each (no truth: stability counters only; starts mapped by species name).
         The paths are as long as the truth set holds: its replay of the old qual set's first rows (QC family 1,000
         calls instead of 10,000; QH family 128 of the 1,664 paths, 1,000 holds; QL 3,500 holds; QLc 3,000 calls).
         Less trajectory coverage than the old set; the boxes add 131,072 states from 0 to 1e8 s, lit and dark.
  full   every box; then a 25-s clock of 3,456,000 ticks (1000 days): QL at dt 25 s (every tick)
         and 500 s (every 20th), QLc (every 20th), the QH family at dt 50 s (every 2nd), the QC family and the QH family
         at 1500-s holds as in quick, panel P at dt 25 / 50 / 500 s (every 1st / 2nd / 20th) for 1000 days. After 1,000
         ticks it prints the estimate (time so far + remaining ticks x ms per tick of ticks 1-1,000) and stops above
         5 h unless --force.
  --save out.npz  per-path summaries, the 12 gases of every cell every 1,000 (quick) / 10,000 (full) ticks and at the
                  end, the per-box results, timings (refuses to overwrite).
  --ref ref.npz   paired comparison with a saved run of the same tier and the same truth (b_set, q_set: digests of the
                  scored truth and forcing): per path the change of the worst error and the largest state change,
                  counts above 0.01 / 0.03 dex, steps per call, unfinished calls, latency.
Exit status 1 when a path or box has a miss, or a call is unfinished or leaves a non-finite or negative state.
    python bench/kintest.py quick|full [--save out.npz] [--ref ref.npz] [--force]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from maxichem import kincuda as KC  # noqa: E402
from maxichem import kinvulcan as KV  # noqa: E402
from maxichem import paths  # noqa: E402

GASES = ["H2O", "CO", "CO2", "CH4", "NH3", "HCN", "C2H2", "H", "OH", "O", "H2", "N2"]
FLOOR = np.log10([1e-10] * 7 + [1e-8] * 3 + [1e-6] * 2)
MISS, LARGE = 0.1, 1.0
HOLD, DAY = 1500.0, 86400.0
READ = 1000                 # ticks between read-backs of the counters
CHUNK = 65536               # most boxes per chunk
CAP_H = 5.0                 # full: stop above this estimate (hours) unless --force
META = ("id", "panel", "kind", "stratum", "start", "comp")
KEYS_FB = ("bad_input", "nonfinite", "unfinished", "drift", "recomputed")   # CudaSolver.fallback
NEVER = 1 << 62


def floored(x):
    """Floored log10 of the scored gases' mole fractions (last axis = the 12 gases); a negative or non-finite value is
    NaN (a failure when scored)."""
    return np.maximum(np.log10(np.maximum(np.where(np.isfinite(x) & (x >= 0), x, np.nan), 1e-300)), FLOOR)


def check_jobs():
    """Stops unless the truth paths on disk are exactly those that paths.QUAL's jobs.json lists, once each, and each
    holds the rows its job asked for (min(n, the source's rows after the start)): a missing or short path would shrink
    the coverage without notice."""
    jobs = json.loads((paths.QUAL / "jobs.json").read_text())
    rel = [j["out"].split(f"/{paths.QUAL.name}/", 1)[-1] for j in jobs]
    want = set(rel)
    have = {str(f.relative_to(paths.QUAL)) for f in paths.QUAL.rglob("*.npz")}
    if want != have or len(want) != len(rel):
        raise SystemExit(f"{paths.QUAL}: {len(rel) - len(want)} duplicate jobs, {len(want - have)} of {len(want)} listed "
                         f"truth paths missing {sorted(want - have)[:3]}, {len(have - want)} not listed "
                         f"{sorted(have - want)[:3]}")
    short = []
    for j, r in zip(jobs, rel):
        z = np.load(paths.QUAL / r)
        hold = "saved_h" in z.files
        s0 = int(z["h0"] if hold else z["k0"]) if str(z["kind"]) == "twin" else 0
        rows = len(z["saved_h"]) if hold else len(z["x_call"])
        if rows != min(int(j["n"]), len(np.load(j["src"])["T"]) - s0):
            short.append(r)
    if short:
        raise SystemExit(f"{paths.QUAL}: {len(short)} truth paths shorter than their jobs, e.g. {short[:3]}")


def load_paths(dirs, sp, gi, hold):
    """Qual truth paths of the folders dirs (load_path)."""
    return [load_path(f, sp, gi, hold) for d in dirs for f in sorted((paths.QUAL / d).glob("*.npz"))]


def load_path(f, sp, gi, hold):
    """One qual truth path: forcing rows (calls, or 1500-s holds if hold), start, mixes keyed to rows, stored states
    (row r = the state after row r) as floored log10 of the 12 gases. A kick twin starts at its kick row from x_kicked;
    its mix at that row is already in x_kicked."""
    z = np.load(f)
    assert [str(q) for q in z["species"]] == sp and bool(z["accepted"]), f
    twin = str(z["kind"]) == "twin"
    s0 = int(z["h0"] if hold else z["k0"]) if twin else 0
    if hold:
        rows, xs = z["saved_h"], z["x_saved"]
    elif "x_call" in z.files:
        xs = z["x_call"]
        rows = np.arange(s0, s0 + len(xs))
        assert s0 + len(xs) == len(z["T"]), f
    else:
        rows, xs = z["saved_call"], z["x_saved"]
    return dict(T=z["T"], P=z["Pbar"], dt=None if hold else z["dt"], s0=s0, x0=z["x_kicked"] if twin else z["x0"],
                mix=[(int(a), float(fr), y / y.sum()) for a, fr, y in zip(z["mix_at"], z["mix_f"], z["mix_x"])
                     if int(a) > s0 or not twin],
                rows=rows.astype(np.int64), truth=floored(xs[:, gi]).astype(np.float32),
                seg=(z["seg_start"], [str(q) for q in z["seg_names"]]) if "seg_start" in z.files else None,
                meta=dict(id=int(z["id"]), panel=str(z["panel"]), kind=str(z["kind"]),
                          stratum=str(z["stratum"]), start=str(z["start"]), comp=str(z["comp"])))


def load_p(sp):
    """Decision panel P: GCM-like paths (T, P every call) with their recorded mixes; no truth at a fixed dt."""
    D = paths.DESIGNS / "decision/P"
    r, s, F = (np.load(D / f"{f}.npz") for f in ("rows", "starts", "forcing"))
    s53 = [str(q) for q in s["species53"]]
    to = lambda a: np.stack([a[..., s53.index(q)] if q in s53 else np.zeros(a.shape[:-1]) for q in sp], -1)  # noqa
    T, Pb, mr, mk, mf, mx = (F[k] for k in ("T", "Pbar", "mix_row", "mix_k", "mix_f", "mix_x"))
    x53, mx = to(s["x53"]), to(mx)                       # by species name onto the network (the others 0)
    out = []
    for p in range(len(r["N"])):
        q = np.flatnonzero(mr == p)
        out.append(dict(T=T[p], P=Pb[p], dt=None, s0=0, x0=x53[r["unit"][p]],
                        mix=[(int(mk[i]), float(mf[i]), mx[i] / mx[i].sum()) for i in q], rows=None, truth=None,
                        seg=None, meta=dict(id=int(r["job"][p]), panel="P", kind="main", stratum="-",
                                            start=f"age{int(r['age_idx'][p])}", comp="-")))
    return out


class Block:
    """One panel on one schedule: call j runs on tick j * stride and uses forcing row (j // rep) % cyc of each path (rep
    calls per 1500-s hold; cyc = the row count of a cycled panel); a path takes part in calls s0 <= j < e; a mix keyed
    to row r comes before call rep * r, a stored state of row r is the state after call rep * r + rep - 1."""

    def __init__(self, name, cells, stride, ncall, rep=1, dt=None, cyc=None):
        self.name, self.cells, self.stride, self.ncall, self.rep, self.dt = name, cells, stride, ncall, rep, dt
        self.cyc = cyc or NEVER
        self.n = len(cells)
        self.truth = cells[0]["truth"] is not None
        L = np.array([len(c["T"]) for c in cells])
        assert cyc is None or (L == cyc).all()
        self.s0 = np.array([rep * c["s0"] for c in cells])
        self.e = np.full(self.n, ncall) if cyc else np.minimum(rep * L, ncall)
        self.nrow = L if cyc else np.minimum(L, -(-ncall // rep))      # forcing rows used per path
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

    def mix(self, row, X):
        """The mixes keyed to this forcing row, before the row's first call."""
        a, z = self.mb[row], self.mb[row + 1]
        if z > a:
            c, f = self.mc[a:z], self.mf[a:z]
            X[c] = (1.0 - f) * X[c] + f * self.my[a:z]


def score(x, c, truth, j, best, cnt, gi, fl):
    """Score the states x[c] against truth rows: per cell the worst error with its call j and gas (best), and the counts
    of stored states, misses and large misses (cnt)."""
    lx = torch.log10(x[c][:, gi].clamp(min=1e-300)).maximum(fl)
    e = (lx - truth).abs()
    ep, g = e.max(1)
    ep = torch.where(torch.isfinite(lx).all(1) & torch.isfinite(truth).all(1), ep, torch.inf)   # bad truth: a failure
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


def boxes(solver, net, sp, gh, gi, fl, cap, dev):
    """The VULCAN truth boxes (paths.BOXES, made by the truth generator's sobol.py: constant T, P and photolysis rates,
    stored at fixed stops from 0 to 1e8 s): the first cap boxes (None: all), each from its stored start in chained
    calls whose lengths are the stop intervals, every stored state scored (gh: the scored gases' columns, gi: the
    same on the device). Refuses a set whose manifest lists boxes not run, failed runs, bad states or element drift, or
    whose shards do not hold the accepted count; reports the quarantined boxes (a step forced at the generator's dt_min,
    left out). b_set: a digest of the scored truth and inputs. Returns the per-box results (b_*)."""
    man = json.loads((paths.BOXES / "manifest.json").read_text())
    shards = [paths.BOXES / f for f in man.get("shards", [])]
    bad = {k: len(man.get(k, [])) for k in ("errors", "bad_states", "drift_over_1e-6")}
    ids = np.concatenate([np.load(f)["box"] for f in shards]) if shards else np.zeros(0, int)
    cover = np.zeros(man["boxes"], int)
    np.add.at(cover, np.concatenate([ids, np.asarray(man.get("quarantine", []), int)]), 1)
    if man.get("pending", 1) or any(bad.values()) or len(ids) != man["accepted"] or (cover != 1).any():
        raise SystemExit(f"{paths.BOXES}: not complete (manifest: {man.get('pending')} boxes not run, {bad}, "
                         f"{man['accepted']} accepted; shards hold {len(ids)} boxes, design boxes not held exactly once "
                         f"(accepted or quarantined) {int((cover != 1).sum())})")
    print(f"boxes: {man['accepted']:,} of {man['boxes']:,} accepted, {len(man['quarantine'])} quarantined (left out)")
    z0 = np.load(shards[0])
    for f in shards:
        z = np.load(f)
        if [str(q) for q in z["species"]] != sp or not np.array_equal(z["stops"], z0["stops"]) or \
                not np.array_equal(z["lines"], z0["lines"]):
            raise SystemExit(f"{f}: species, stops or photolysis lines differ from the paths' and the other shards'")
    stops, lines = z0["stops"], [str(q) for q in z0["lines"]]
    col = [lines.index(f"{q['absorber']},{q['branch']}") for q in net.photo]
    keys = ("box", "X", "T", "P", "J", "lit", "kind")
    Z = {k: [] for k in keys}
    n = 0
    for f in shards:
        if cap is not None and n >= cap:
            break
        z = np.load(f)
        for k in keys:
            Z[k].append(z[k])
        n += len(z["box"])
    Z = {k: np.concatenate(v)[:cap] for k, v in Z.items()}
    R, nc = len(Z["box"]), len(stops) - 1
    B = dict(b_box=Z["box"], b_calls=np.full(R, nc), b_worst=np.zeros(R), b_kworst=np.full(R, -1),
             b_gworst=np.full(R, -1), b_saved=np.zeros(R), b_nmiss=np.zeros(R), b_nlarge=np.zeros(R),
             b_unfinished=np.zeros(R), b_nonfinite=np.zeros(R), b_steps=np.zeros(R), b_lx=np.zeros((R, 12), np.float32))
    fb = {}
    t0 = time.time()
    tt = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)  # noqa: E731
    for p0 in range(0, R, CHUNK):
        sl = slice(p0, min(R, p0 + CHUNK))
        m = sl.stop - p0
        x, T, P, J = tt(Z["X"][sl, 0]), tt(Z["T"][sl]), tt(Z["P"][sl]), tt(Z["J"][sl][:, col])
        h = torch.full_like(T, 1e-10)
        acc = torch.zeros(m, 3, dtype=torch.float64, device=dev)              # unfinished, non-finite, steps
        cnt = torch.zeros(m, 3, dtype=torch.float64, device=dev)              # saved states, misses, large misses
        best = (torch.zeros(m, dtype=torch.float64, device=dev), torch.full((m,), -1, device=dev),
                torch.full((m,), -1, device=dev))
        c = torch.arange(m, device=dev)
        for k in range(1, nc + 1):
            x, h, done, nok = solver(x, T, P, torch.full_like(T, stops[k] - stops[k - 1]), h, J=J)
            add_fb(fb, solver)
            bad = torch.logical_not(torch.isfinite(x).all(1))
            acc += torch.stack((torch.logical_not(done).double(), bad.double(), nok), 1)
            xt = tt(Z["X"][sl, k][:, gh])
            truth = torch.where(torch.isfinite(xt), torch.log10(xt.clamp(min=1e-300)).maximum(fl), torch.nan)
            score(x, c, truth, k, best, cnt, gi, fl)
        B["b_worst"][sl], B["b_kworst"][sl], B["b_gworst"][sl] = (q.cpu().numpy() for q in best)
        B["b_saved"][sl], B["b_nmiss"][sl], B["b_nlarge"][sl] = cnt.cpu().numpy().T
        B["b_unfinished"][sl], B["b_nonfinite"][sl], B["b_steps"][sl] = acc.cpu().numpy().T
        B["b_lx"][sl] = torch.log10(x[:, gi].clamp(min=1e-300)).maximum(fl).float().cpu().numpy()
    torch.cuda.synchronize()
    B["b_s"], B["b_cellcalls"] = time.time() - t0, R * nc
    h = hashlib.sha256(json.dumps(man["stamp"], sort_keys=True).encode())
    for a in [Z[k] for k in keys] + [stops, np.array(lines), np.array(sp)]:
        h.update(np.ascontiguousarray(a).tobytes())
    B["b_set"] = np.array(f"{man['network']} rtol {man.get('rtol')} {R} boxes sha256 {h.hexdigest()[:16]}")
    B["fallback_boxes"] = np.array([fb.get(q, 0) for q in KEYS_FB])
    box_report(B, Z, stops)
    return B


def box_report(B, Z, stops):
    """Misses-first lines of the box pass."""
    w = B["b_worst"]
    print(f"boxes: {len(w):,} boxes x {len(stops) - 1} calls in {B['b_s']:.0f} s "
          f"({1e6 * B['b_s'] / B['b_cellcalls']:.3f} us per cell-call), steps per call "
          f"{B['b_steps'].sum() / B['b_calls'].sum():.3f}; guard {dict(zip(KEYS_FB, B['fallback_boxes'].tolist()))}")

    def line(name, s):
        if s.any():
            print(f"  {name:20s} {int(s.sum()):7,d} boxes: miss {100 * (w[s] > MISS).mean():6.3f} %  large "
                  f"{100 * (w[s] > LARGE).mean():6.3f} %  worst {w[s].max():.3f} | stored states "
                  f"{int(B['b_saved'][s].sum()):,}: miss {int(B['b_nmiss'][s].sum()):,} large "
                  f"{int(B['b_nlarge'][s].sum()):,} | boxes with an unfinished call {int((B['b_unfinished'][s] > 0).sum())}"
                  f", with a non-finite state {int((B['b_nonfinite'][s] > 0).sum())}")

    line("all boxes", np.ones(len(w), bool))
    line("dark", Z["lit"] == 0)
    line("lit", Z["lit"] == 1)
    for k, name in enumerate(("unreacted", "reheat", "mixed", "scrambled", "hot-H")):
        line(f"start {name}", Z["kind"] == k)
    for lo, hi in ((300, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 3001)):
        line(f"T {lo}-{hi} K", (Z["T"] >= lo) & (Z["T"] < hi))
    print("  worst boxes:")
    for i in np.argsort(-w)[:5]:
        print(f"    box {Z['box'][i]} T {Z['T'][i]:.0f} K P {Z['P'][i] / 1e6:.1e} bar {'lit' if Z['lit'][i] else 'dark'} "
              f"start {int(Z['kind'][i])}: {w[i]:.3f} dex in {GASES[B['b_gworst'][i]]} at stop "
              f"{stops[B['b_kworst'][i]]:.3g} s")


def make_blocks(tier, sp, gi):
    """The tier's blocks (see the module docstring)."""
    t0 = time.time()
    qc = load_paths(("QC", "QCs", "twins/QC"), sp, gi, hold=False)
    qh = load_paths(("QH", "QHs", "twins/QH"), sp, gi, hold=True)
    ql = load_paths(("QL",), sp, gi, hold=True)
    qlc = load_paths(("QLc",), sp, gi, hold=False)
    pp = load_p(sp)
    nqc, nh, nl, nlc, npp = (max(len(c["T"]) for c in v) for v in (qc, qh, ql, qlc, pp))
    print(f"truth and forcing loaded in {time.time() - t0:.0f} s", flush=True)
    if tier == "quick":
        return [Block("QC family, per call", qc, 1, nqc),
                Block("QH family, 1500-s holds", qh, 1, nh, dt=HOLD),
                Block("QL, dt 500 s", ql, 1, 10000, rep=3, dt=500.0),
                Block("QLc, dt 500 s", qlc, 1, 10000),
                *[Block(f"P, dt {d:g} s", pp, 1, npp, dt=d, cyc=npp) for d in (25.0, 50.0, 500.0)]]
    return [Block("QL, dt 25 s", ql, 1, 60 * nl, rep=60, dt=25.0),
            Block("QL, dt 500 s", ql, 20, 3 * nl, rep=3, dt=500.0),
            Block("QLc, dt 500 s", qlc, 20, nlc),
            Block("QH family, dt 50 s", qh, 2, 30 * nh, rep=30, dt=50.0),
            Block("QC family, per call", qc, 1, nqc),
            Block("QH family, 1500-s holds", qh, 1, nh, dt=HOLD),
            *[Block(f"P, dt {d:g} s", pp, s, int(1000 * DAY / d), dt=d, cyc=npp)
              for d, s in ((25.0, 1), (50.0, 2), (500.0, 20))]]


def assemble(blocks, net, dev):
    """Global state, flat forcing (T, P in dyn cm^-2, dt per forcing row) and per-cell counters for all blocks."""
    X0, Tf, Pf, DTf, off0 = [], [], [], [], 0
    meta = {q: [] for q in META}
    c0 = 0
    for b in blocks:
        off = np.zeros(b.n, np.int64)
        for i, c in enumerate(b.cells):
            m = int(b.nrow[i])
            off[i] = off0
            off0 += m
            Tf.append(c["T"][:m]), Pf.append(c["P"][:m] * 1e6)
            DTf.append(c["dt"][:m] if b.dt is None else np.full(m, b.dt))
            X0.append(c["x0"] / c["x0"].sum())
            for q in META:
                meta[q].append(c["meta"][q])
        b.setup(c0, off, dev)
        c0 += b.n
    tt = lambda a: torch.as_tensor(np.concatenate(a), dtype=torch.float64, device=dev)  # noqa: E731
    X = torch.as_tensor(np.stack(X0), device=dev)
    atoms = torch.as_tensor(net.atoms, device=dev)
    iH = net.elements.index("H")
    el = X @ atoms
    N = len(X0)
    z = lambda *s, dt=torch.float64: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
    return dict(X=X, H=torch.full((N,), 1e-10, dtype=torch.float64, device=dev), T=tt(Tf), P=tt(Pf), dt=tt(DTf),
                host=(np.concatenate(Tf), np.concatenate(Pf), np.concatenate(DTf)), atoms=atoms, iH=iH,
                ratio0=el / el[:, iH:iH + 1], meta={q: np.array(v) for q, v in meta.items()},
                acc=z(N, 5),                # calls, accepted steps, unfinished, non-finite, negative
                cnt=z(N, 3),                # stored states, misses, large misses
                best=(z(N), torch.full((N,), -1, device=dev), torch.full((N,), -1, device=dev)),
                drift=z(N), N=N)


def run(blocks, S, solver, ticks, snap, full, force, t_all, gi, fl, dev):
    """The tick loop; returns timings, snapshots and the guard's counts."""
    X, H, acc, best, drift = S["X"], S["H"], S["acc"], S["best"], S["drift"]
    act, fb, snaps, snap_t, cellcalls = list(blocks), {}, [], [], 0
    lx = lambda: torch.log10(X[:, gi].clamp(min=1e-300)).maximum(fl).float().cpu().numpy()  # noqa: E731
    t0 = tl = time.time()
    for t in range(ticks):
        ids, offs, sched = [], [], []
        for b in act:
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
                b.mix(row, X)
            ids.append(b.gidx), offs.append(b.goff + row), sched.append((b, j))
        if ids:
            idx = ids[0] if len(ids) == 1 else torch.cat(ids)
            fo = offs[0] if len(offs) == 1 else torch.cat(offs)
            xo, ho, done, nok = solver(X[idx], S["T"][fo], S["P"][fo], S["dt"][fo], H[idx])
            add_fb(fb, solver)
            X[idx], H[idx] = xo, ho
            cellcalls += idx.numel()
            fin = torch.isfinite(xo).all(1)
            acc.index_add_(0, idx, torch.stack((torch.ones_like(nok), nok, (~done).double(), (~fin).double(),
                                                (xo.min(1).values < 0).double()), 1))
            el = xo @ S["atoms"]
            d = torch.log10(el / el[:, S["iH"]:S["iH"] + 1] / S["ratio0"][idx]).abs().nan_to_num(nan=0.0).max(1).values
            drift[idx] = torch.maximum(drift[idx], d)
            for b, j in sched:
                if b.truth:
                    a, z = b.sb[j], b.sb[j + 1]
                    if z > a:
                        score(X, b.sc[a:z], b.st[a:z], j, best, S["cnt"], gi, fl)
        if (t + 1) % READ == 0 or t + 1 == ticks:
            tot = acc.sum(0).tolist()
            w = best[0]
            now = time.time()
            if not full or (t + 1) % (100 * READ) == 0 or t + 1 in (READ, ticks):
                print(f"tick {t + 1:,} of {ticks:,}: {now - t0:.0f} s, {1e3 * (now - tl) / ((t % READ) + 1):.2f} ms per "
                      f"tick (last {(t % READ) + 1:,}), {sum(b.nlive for b, _ in sched)} cells this tick, paths with a miss "
                      f"{int((w > MISS).sum())}, worst {float(w.max()):.3f} dex, unfinished {tot[2]:.0f}, non-finite "
                      f"{tot[3]:.0f}, negative {tot[4]:.0f}, recomputed by the guard {fb.get('recomputed', 0)}", flush=True)
            tl = now
            if (t + 1) % snap == 0:
                snaps.append(lx()), snap_t.append(t + 1)
            act = [b for b in act if (b.end - 1) * b.stride > t]
            if full and t + 1 == READ:
                est = (now - t_all) + (ticks - READ) * (now - t0) / READ
                print(f"estimate: {est / 3600:.2f} h in all ({(now - t_all) / 60:.1f} min so far, then {ticks - READ:,} "
                      f"ticks x {1e3 * (now - t0) / READ:.2f} ms per tick of ticks 1-{READ:,})", flush=True)
                if est > CAP_H * 3600 and not force:
                    print(f"stopped: the estimate exceeds {CAP_H:g} h (--force runs it anyway)", flush=True)
                    sys.exit(2)
    torch.cuda.synchronize()
    return dict(loop_s=time.time() - t0, ticks=ticks, cellcalls=cellcalls, snap_tick=np.array(snap_t),
                snap_lx=np.stack(snaps) if snaps else np.zeros((0, S["N"], 12), np.float32), final_lx=lx(),
                fallback_loop=np.array([fb.get(q, 0) for q in KEYS_FB]))


def results(blocks, S):
    """Per-cell counters and metadata as numpy arrays (the saved per-path summaries)."""
    acc, cnt = S["acc"].cpu().numpy(), S["cnt"].cpu().numpy()
    R = dict(block_names=np.array([b.name for b in blocks]), truth_block=np.array([b.truth for b in blocks]),
             block_of=np.concatenate([np.full(b.n, i) for i, b in enumerate(blocks)]),
             mixed=np.concatenate([b.mixed for b in blocks]), **S["meta"])
    R["worst"], R["kworst"], R["gworst"] = (q.cpu().numpy() for q in S["best"])
    R["stored"], R["nmiss"], R["nlarge"] = cnt.T
    R["calls"], R["steps"], R["unfinished"], R["nonfinite"], R["negative"] = acc.T
    R["drift"] = S["drift"].cpu().numpy()
    return R


def line(name, s, R, truth):
    """One misses-first line for the paths s."""
    w, calls = R["worst"][s], R["calls"][s].sum()
    um = s & ~R["mixed"]
    dr = f"{R['drift'][um].max():.1e} dex" if um.any() else "-"
    head = f"  {name:10s} {int(s.sum()):5d} paths"
    if truth:
        head += (f": miss {100 * (w > MISS).mean():5.1f} %  large {100 * (w > LARGE).mean():5.1f} %  worst {w.max():.3f} | "
                 f"stored states {int(R['stored'][s].sum()):,}: miss {int(R['nmiss'][s].sum())} large "
                 f"{int(R['nlarge'][s].sum())}")
    print(head + f" | calls {int(calls):,}: unfinished {int(R['unfinished'][s].sum())} non-finite "
          f"{int(R['nonfinite'][s].sum())} negative {int(R['negative'][s].sum())} | drift (unmixed) {dr} | "
          f"steps per call {R['steps'][s].sum() / max(calls, 1):.3f}")


def report(blocks, R, L, S):
    """Misses first: totals over the truth paths, then per block (split by panel and kind; breakdowns by stratum,
    start and composition; the worst paths), then timings and the guard's counts."""
    tr = R["truth_block"][R["block_of"]]
    w = np.where(tr, R["worst"], -1.0)
    c = int(np.argmax(w))
    print(f"\nMISSES FIRST. qual truth paths {int(tr.sum()):,}: with a miss {int((w > MISS).sum())}, with a large miss "
          f"{int((w > LARGE).sum())}; stored states {int(R['stored'].sum()):,}: with a miss {int(R['nmiss'].sum())}, large "
          f"{int(R['nlarge'].sum())}; worst {w[c]:.3f} dex ({R['block_names'][R['block_of'][c]]}, path {R['id'][c]}); "
          f"all blocks: unfinished calls {int(R['unfinished'].sum())}, non-finite states {int(R['nonfinite'].sum())}, "
          f"negative {int(R['negative'].sum())}")
    Tf, Pf, DTf = S["host"]
    for bi, b in enumerate(blocks):
        s = R["block_of"] == bi
        print(f"{b.name}: {b.n} paths, up to {b.ncall:,} calls (every {b.stride} tick{'s' if b.stride > 1 else ''})")
        lab = np.char.add(np.char.add(R["panel"], " "), R["kind"])
        for v in dict.fromkeys(lab[s]):
            line(v, s & (lab == v), R, b.truth)
        if not b.truth:
            continue
        for key in ("stratum", "start", "comp"):
            parts = []
            for v in np.unique(R[key][s]):
                q = s & (R[key] == v)
                parts.append(f"{v} {int(q.sum())} (miss {int((R['worst'][q] > MISS).sum())}, worst {R['worst'][q].max():.3f})")
            print(f"    by {key}: " + "; ".join(parts))
        print("    worst paths:")
        for c in b.c0 + np.argsort(-R["worst"][s])[:5]:
            i, j = c - b.c0, int(R["kworst"][c])
            if j < 0:
                continue
            f = b.off[i] + (j // b.rep) % b.cyc
            seg = b.cells[i]["seg"]
            where = f" seg {seg[1][int(np.searchsorted(seg[0], j // b.rep, side='right')) - 1]}" if seg else ""
            print(f"      {R['id'][c]} {R['panel'][c]} {R['kind'][c]} {R['stratum'][c]} {R['start'][c]} {R['comp'][c]}: "
                  f"{R['worst'][c]:.3f} dex in {GASES[R['gworst'][c]]} at call {j}{where} (T {Tf[f]:.0f} K, P "
                  f"{Pf[f] / 1e6:.2e} bar, dt {DTf[f]:.1f} s); stored states with a miss {int(R['nmiss'][c])}")
    print(f"loop: {L['ticks']:,} ticks in {L['loop_s']:.0f} s = {1e3 * L['loop_s'] / L['ticks']:.3f} ms per tick; "
          f"{L['cellcalls']:,} cell-calls = {1e6 * L['loop_s'] / L['cellcalls']:.3f} us per cell-call; guard "
          f"{dict(zip(KEYS_FB, L['fallback_loop'].tolist()))}")


def compare(new, ref):
    """Paired comparison of two runs of the same tier (mappings as written by --save)."""
    for k in ("tier", "id", "block_of", "b_box", "b_set", "q_set"):
        assert np.array_equal(np.asarray(new[k]), np.asarray(ref[k])), f"runs differ in {k}: not comparable"
    assert np.array_equal(new["snap_tick"], ref["snap_tick"])
    lat = lambda z: (1e3 * float(z["loop_s"]) / int(z["ticks"]), 1e6 * float(z["b_s"]) / int(z["b_cellcalls"]))  # noqa
    (m1, y1), (m0, y0) = lat(new), lat(ref)
    print(f"\nPAIRED with the reference (tier {new['tier']}): latency loop {m1:.3f} vs {m0:.3f} ms per tick "
          f"({m1 / m0:.2f}x), boxes {y1:.3f} vs {y0:.3f} us per cell-call ({y1 / y0:.2f}x)")
    dif = lambda a, b: np.nan_to_num(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)), nan=np.inf)  # noqa
    ds = np.maximum(dif(new["final_lx"], ref["final_lx"]).max(1),
                    dif(new["snap_lx"], ref["snap_lx"]).max((0, 2)) if len(new["snap_tick"]) else 0.0)

    def row(name, s, truth, w1, w0, st, c1, c0, u1, u0, ids):
        txt = f"  {name:24s} {int(s.sum()):7,d} paths"
        if truth:
            dw = np.nan_to_num(w1[s] - w0[s], nan=np.inf, posinf=np.inf, neginf=-np.inf)
            i, k = int(np.argmax(dw)), int(np.argmin(dw))
            txt += (f" | worst error new - ref: mean {dw[np.isfinite(dw)].mean() if np.isfinite(dw).any() else 0:+.5f}, "
                    f"largest rise {dw[i]:+.4f} ({ids[s][i]}), largest fall {dw[k]:+.4f} ({ids[s][k]}); |change| "
                    f"> 0.01: {int((np.abs(dw) > 0.01).sum())}, > 0.03: {int((np.abs(dw) > 0.03).sum())}")
        print(txt + f" | state change > 0.01: {int((st[s] > 0.01).sum())}, > 0.03: {int((st[s] > 0.03).sum())} (largest "
              f"{st[s].max():.2e}) | steps per call {c1:.3f} vs {c0:.3f} | unfinished {int(u1)} vs {int(u0)}")

    for bi, name in enumerate(new["block_names"]):
        s = new["block_of"] == bi
        spc = lambda z: z["steps"][s].sum() / max(z["calls"][s].sum(), 1)  # noqa: E731
        row(str(name), s, bool(new["truth_block"][bi]), new["worst"], ref["worst"], ds, spc(new), spc(ref),
            new["unfinished"][s].sum(), ref["unfinished"][s].sum(), new["id"])
    bs = np.ones(len(new["b_box"]), bool)
    row("boxes", bs, True, new["b_worst"], ref["b_worst"], dif(new["b_lx"], ref["b_lx"]).max(1),
        new["b_steps"].sum() / new["b_calls"].sum(), ref["b_steps"].sum() / ref["b_calls"].sum(),
        new["b_unfinished"].sum(), ref["b_unfinished"].sum(), new["b_box"])


def main():
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
    sp = [str(q) for q in np.load(sorted((paths.QUAL / "QC").glob("*.npz"))[0])["species"]]
    gi = [sp.index(g) for g in GASES]
    net = KV.VulcanNetwork(f"{paths.VULCAN_THERMO}/NCHO_photo_network.txt", f"{paths.VULCAN_THERMO}/NASA9", species=sp)
    solver = KC.CudaSolver(net, dev, photo=True)
    gi_t, fl = torch.tensor(gi, device=dev), torch.tensor(FLOOR, device=dev)
    print(f"{a.tier}: {torch.cuda.get_device_name(dev)}, kernel ready ({solver.wpb} warps per block), "
          f"{time.time() - t_all:.0f} s", flush=True)
    Y = boxes(solver, net, sp, gi, gi_t, fl, 32768 if quick else None, dev)
    blocks = make_blocks(a.tier, sp, gi)
    qh = hashlib.sha256()                                 # the paths' truth and forcing, for paired runs
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
    L = run(blocks, S, solver, ticks, READ if quick else 10 * READ, not quick, a.force, t_all, gi_t, fl, dev)
    solver.check()                                                      # raises if any cell was below 300 K
    R = results(blocks, S)
    report(blocks, R, L, S)
    R.update(L, **Y, tier=np.array(a.tier), q_set=np.array(qh.hexdigest()[:16]))
    print(f"wall time {time.time() - t_all:.0f} s (boxes {Y['b_s']:.0f} s, loop {L['loop_s']:.0f} s)", flush=True)
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.save, **R)
        print(f"saved {a.save}")
    if ref is not None:
        compare(R, ref)
    tr = R["truth_block"][R["block_of"]]
    fail = int((np.where(tr, R["worst"], 0.0) > MISS).sum() + (Y["b_worst"] > MISS).sum() + R["unfinished"].sum()
               + R["nonfinite"].sum() + R["negative"].sum() + Y["b_unfinished"].sum() + Y["b_nonfinite"].sum())
    if fail:
        sys.exit(f"FAIL: {fail} paths or boxes with a miss, unfinished calls, non-finite or negative states")


if __name__ == "__main__":
    main()
