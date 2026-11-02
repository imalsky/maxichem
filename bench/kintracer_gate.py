"""The quick gate of a reduced-tracer design (maxichem/kintracer.py) against VULCAN and the full mechanism, misses first.

The design runs paired with the full mechanism (all species advected; kincuda.CudaSolver) on the same truth paths: the
same forcing and the same recorded mixes, applied to both by mass (partner gas y: f_m = f M_y / ((1 - f) M_x + f M_y)
with M_x the full mechanism's mean molar mass, which is VULCAN's molar mixing for the full state; the design mixes its
tracers, its memory stays). Scored per path, at every call against the full mechanism and at the stored states against
VULCAN (12 gases, floors 1e-10 H2O CO CO2 CH4 NH3 HCN C2H2, 1e-8 H OH O, 1e-6 H2 N2):
  miss    after the path's first mix: a call more than 0.1 dex off the full mechanism, or a stored state more than 0.1
          dex off VULCAN; where the full mechanism itself is more than 0.1 dex off VULCAN the state is open and counts
          only if the design is more than 0.03 dex off the full mechanism. Large miss: above 1 dex.
  defect  before the first mix (the whole run on a path without mixes): a call more than 0.01 dex off the full
          mechanism.
Also reported: reconstruction validity (the layer's defects: an advected row not reproduced, a negative He remainder, a
non-finite or negative state; chemistry cells still failed after the recovery), the projection's spill path by its
effect (cell-calls and the worst difference from the full mechanism there), element drift against the full mechanism
(largest |log10| ratio of an element's amount, every call), wall time per call.
  (a) The kill gate: 16 QC / QCs paths to call 1,000, QL 4240020001 to hold 3,500, QLc 4240070001 to call 3,000. A path
      with a miss or a defect kills the design.
  (b) If (a) passes, the panels (folders under the truth data; default qual/QC, qual/QCs, qual/twins/QC, qual/QL,
      qual/QLc), each path to --days of model time (default 100). Panels with hold files (stored per 1500-s hold) run
      one call per hold; the others call by call with their dt. A kick twin starts at its kick call from x_kicked.
    python bench/kintracer_gate.py --design FILE [--panels a,b,...] [--days D] [--save out.json]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maxichem import kincuda as KC  # noqa: E402
import kintest as KX  # noqa: E402
from maxichem import kintracer as KT  # noqa: E402
from maxichem import kinvulcan as KV  # noqa: E402
from maxichem import paths  # noqa: E402

GASES, FLOOR = KX.GASES, KX.FLOOR
MISS, LARGE, OPEN_TOL, DEFECT = 0.1, 1.0, 0.03, 0.01
HOLD, DAY = 1500.0, 86400.0
KILL = (4240050031, 4240000443, 4240050008, 4240050009, 4240000107, 4240050030, 4240000025, 4240000130, 4240000131,
        4240000338, 4240000339, 4240050007, 4240050037, 4240050042, 4240050054, 4240050055)
KILL_CALLS, KILL_QL, KILL_QLC = 1000, (4240020001, 3500), (4240070001, 3000)
PANELS = "qual/QC,qual/QCs,qual/twins/QC,qual/QL,qual/QLc"
READ = 500                                   # calls between progress lines


def load(files, sp, gi, ncap=None, days=None):
    """Truth paths (kintest.load_path) with the calls to run n (to ncap, or to days of model time)."""
    out = []
    for f in files:
        hold = "saved_h" in np.load(f).files
        c = KX.load_path(f, sp, gi, hold)
        c["dt"] = np.full(len(c["T"]), HOLD) if hold else c["dt"]
        n = len(c["T"])
        if days is not None:
            n = min(n, int((np.cumsum(c["dt"]) < days * DAY - 1e-6).sum()) + 1)
        c["n"] = n if ncap is None else min(n, ncap)
        c["meta"]["src"] = f.parent.relative_to(paths.DATA / "truth").as_posix()
        out.append(c)
    return out


def gate(cells, ts, kc, net, dev, label):
    """Run the paths (design and full mechanism paired) and return per-path results (numpy) and timings."""
    cells = sorted(cells, key=lambda c: -c["n"])            # live paths = a prefix (paths end in this order)
    N = len(cells)
    ncall = cells[0]["n"]
    nend = np.array([c["n"] for c in cells])
    tt = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=dev)  # noqa: E731
    pad = lambda k: np.stack([np.pad(c[k][:c["n"]], (0, ncall - c["n"]), mode="edge") for c in cells])  # noqa: E731
    T, P, DT = tt(pad("T")), tt(pad("P")) * 1e6, tt(pad("dt"))
    s0 = np.array([c["s0"] for c in cells])
    x0 = tt(np.stack([c["x0"] / c["x0"].sum() for c in cells]))
    n0 = x0 / (x0 @ ts.MW)[:, None]
    # mixes keyed to calls (a hold path's row = its call), kept inside each path's run
    mk, mc, mf, my = [], [], [], []
    first = np.full(N, np.iinfo(np.int64).max)
    sk, sc, st = [], [], []
    for i, c in enumerate(cells):
        for r, f, y in c["mix"]:
            if s0[i] <= r < c["n"]:
                mk.append(r), mc.append(i), mf.append(f), my.append(y)
                first[i] = min(first[i], r)
        keep = (c["rows"] >= s0[i]) & (c["rows"] < c["n"])
        sk.append(c["rows"][keep]), sc.append(np.full(int(keep.sum()), i)), st.append(c["truth"][keep])
    o = np.argsort(np.asarray(mk, np.int64), kind="stable")
    mb = np.searchsorted(np.asarray(mk, np.int64)[o], np.arange(ncall + 1))
    MC = torch.as_tensor(np.asarray(mc, np.int64)[o], device=dev)
    MF = tt(np.asarray(mf)[o])[:, None] if len(mf) else tt(np.zeros((0, 1)))
    MY = tt(np.asarray(my).reshape(-1, len(ts.species))[o])
    sk = np.concatenate(sk)
    o = np.argsort(sk, kind="stable")
    sb = np.searchsorted(sk[o], np.arange(ncall + 1))
    SC = torch.as_tensor(np.concatenate(sc)[o], device=dev)
    ST = torch.as_tensor(np.concatenate(st)[o], device=dev).double()
    starts = {}
    for i in np.flatnonzero(s0 > 0):
        starts.setdefault(int(s0[i]), []).append(i)
    first_t = torch.as_tensor(first, device=dev)
    s0_t = torch.as_tensor(s0, device=dev)
    gi = torch.as_tensor([ts.species.index(g) for g in GASES], device=dev)
    fl = torch.as_tensor(FLOOR, device=dev)
    EL = ts.EL
    # state: the design in specific moles (carried, rows, memory), the full mechanism's n; step sizes
    C, R, G = n0[:, ts.ic].contiguous(), torch.where(ts.hid, n0, torch.zeros_like(n0)) @ ts.A, n0[:, ts.ih].contiguous()
    nc = n0.clone()
    h, hc = (torch.full((N,), 1e-10, dtype=torch.float64, device=dev) for _ in range(2))
    z = lambda dt=torch.float64: torch.zeros(N, dtype=dt, device=dev)  # noqa: E731
    A = dict(wc=z(), kc=z(torch.long), gc=z(torch.long), wv=z(), kv=z(torch.long), gv=z(torch.long), wcv=z(), pre=z(),
             cmiss=z(), clarge=z(), smiss=z(), slarge=z(), sopen=z(), stored=z(), spill=z(), wspill=z(), defect=z(),
             failed=z(), drift=z(), nonfinite=z())
    t0 = time.time()
    m = N
    for k in range(ncall):
        while m and nend[m - 1] <= k:
            m -= 1
        for i in starts.get(k, []):                         # a twin starts here from x_kicked
            C[i], R[i], G[i], nc[i] = n0[i, ts.ic], (torch.where(ts.hid, n0[i], torch.zeros_like(n0[i])) @ ts.A), \
                n0[i, ts.ih], n0[i]
            h[i] = hc[i] = 1e-10
        a, b = mb[k], mb[k + 1]
        if b > a:                                           # mixes before call k (all paths of these mixes are live)
            p, f, y = MC[a:b], MF[a:b], MY[a:b]
            yn = y / (y @ ts.MW)[:, None]
            Mx, My = 1.0 / nc[p].sum(1), y @ ts.MW
            fm = f * My[:, None] / ((1.0 - f) * Mx[:, None] + f * My[:, None])
            nc[p] = (1.0 - fm) * nc[p] + fm * yn
            C[p] = (1.0 - fm) * C[p] + fm * yn[:, ts.ic]
            R[p] = (1.0 - fm) * R[p] + fm * (torch.where(ts.hid, yn, torch.zeros_like(yn)) @ ts.A)
        L = slice(0, m)
        Tk, Pk, Dk = T[L, k], P[L, k], DT[L, k]
        C[L], R[L], G[L], h[L], x, flag = ts.step(C[L], R[L], G[L], Tk, Pk, Dk, h[L])
        xc, hc[L], _, _ = kc(nc[L] / nc[L].sum(1, keepdim=True), Tk, Pk, Dk, hc[L])
        nc[L] = xc / (xc @ ts.MW)[:, None]
        # scoring (live paths that have started)
        on = s0_t[L] <= k
        lx = torch.log10(x[:, gi].clamp(min=1e-300)).maximum(fl)
        lc = torch.log10(xc[:, gi].clamp(min=1e-300)).maximum(fl)
        dd, g = (lx - lc).abs().max(1)
        fin = torch.isfinite(x).all(1)
        dd = torch.where(fin, dd, torch.inf)
        mixed = on & (k >= first_t[L])
        pre = on & (k < first_t[L])
        up = mixed & (dd > A["wc"][L])
        A["kc"][L] = torch.where(up, k, A["kc"][L])
        A["gc"][L] = torch.where(up, g, A["gc"][L])
        A["wc"][L] = torch.where(up, dd, A["wc"][L])
        A["pre"][L] = torch.where(pre, torch.maximum(A["pre"][L], dd), A["pre"][L])
        A["cmiss"][L] += (mixed & (dd > MISS)).double()
        A["clarge"][L] += (mixed & (dd > LARGE)).double()
        sp = ((flag >> 2) & 3) == 2
        A["spill"][L] += (on & sp).double()
        A["wspill"][L] = torch.where(on & sp, torch.maximum(A["wspill"][L], dd), A["wspill"][L])
        A["defect"][L] += (on & (((flag >> 4) & 1) == 1)).double()
        A["failed"][L] += (on & ((flag & 3) == 2)).double()
        A["nonfinite"][L] += (on & (fin.logical_not() | torch.isfinite(xc).all(1).logical_not())).double()
        n2 = x / (x @ ts.MW)[:, None]
        dr = torch.log10((n2 @ EL) / (nc[L] @ EL)).abs().nan_to_num(nan=0.0).max(1).values
        A["drift"][L] = torch.where(on, torch.maximum(A["drift"][L], dr), A["drift"][L])
        a, b = sb[k], sb[k + 1]
        if b > a:                                           # stored VULCAN states after call k
            c = SC[a:b]
            ev, gv = (lx[c] - ST[a:b]).abs().max(1)
            ev = torch.where(fin[c], ev, torch.inf)
            ec = (lc[c] - ST[a:b]).abs().max(1).values
            opn = ec > MISS
            mx = k >= first_t[c]
            sv = torch.where(opn, torch.where(dd[c] > OPEN_TOL, ev, torch.zeros_like(ev)), ev)
            upv = mx & (sv > A["wv"][c])
            A["kv"][c] = torch.where(upv, k, A["kv"][c])
            A["gv"][c] = torch.where(upv, gv, A["gv"][c])
            A["wv"][c] = torch.where(upv, sv, A["wv"][c])
            A["wcv"][c] = torch.maximum(A["wcv"][c], ec)
            A["smiss"][c] += (mx & (sv > MISS)).double()
            A["slarge"][c] += (mx & (sv > LARGE)).double()
            A["sopen"][c] += (mx & opn).double()
            A["stored"][c] += 1.0
        if (k + 1) % READ == 0 or k + 1 == ncall:
            w = torch.maximum(A["wc"], A["wv"])
            el = time.time() - t0
            print(f"{label}: call {k + 1:,} of {ncall:,} ({m} live paths), {el:.0f} s ({1e3 * el / (k + 1):.1f} ms per "
                  f"call); paths with a miss {int((w > MISS).sum())}, worst {float(w.max()):.3f} dex (vs the full "
                  f"mechanism {float(A['wc'].max()):.3f}, vs VULCAN {float(A['wv'].max()):.3f}); defects "
                  f"{int((A['pre'] > DEFECT).sum())}, worst before the first mix {float(A['pre'].max()):.2e}", flush=True)
    torch.cuda.synchronize()
    el = time.time() - t0
    out = {q: v.cpu().numpy() for q, v in A.items()}
    out.update(n=nend, first=first, s0=s0, wall_s=el, ms_per_call=1e3 * el / ncall, ncall=ncall,
               meta={q: np.array([c["meta"][q] for c in cells]) for q in cells[0]["meta"]},
               T=np.stack([c["T"][:c["n"]].min() for c in cells]))
    return out


def report(o, label, counts):
    """Misses first: the paths with a miss / large miss / defect, by group; validity; the worst paths. Returns a summary
    dict."""
    w = np.maximum(o["wc"], o["wv"])
    mixed = o["first"] < o["n"]
    meta = o["meta"]
    miss, large, defect = mixed & (w > MISS), mixed & (w > LARGE), o["pre"] > DEFECT
    S = dict(label=label, paths=len(w), mixing_paths=int(mixed.sum()), miss=int(miss.sum()), large=int(large.sum()),
             defect=int(defect.sum()), worst=float(w[mixed].max()) if mixed.any() else 0.0,
             worst_vs_full=float(o["wc"].max()), worst_vs_vulcan=float(o["wv"].max()),
             worst_before_first_mix=float(o["pre"].max()), full_vs_vulcan_worst=float(o["wcv"].max()),
             open_states=int(o["sopen"].sum()), cellcalls_miss=int(o["cmiss"].sum()),
             cellcalls_large=int(o["clarge"].sum()), states_miss=int(o["smiss"].sum()),
             states_large=int(o["slarge"].sum()), stored=int(o["stored"].sum()), spill_cellcalls=int(o["spill"].sum()),
             spill_worst_vs_full=float(o["wspill"].max()), defect_cellcalls=int(o["defect"].sum()),
             failed_cellcalls=int(o["failed"].sum()), nonfinite=int(o["nonfinite"].sum()),
             element_drift_dex=float(o["drift"].max()), wall_s=o["wall_s"], ms_per_call=o["ms_per_call"],
             ncall=o["ncall"], counts=counts, groups={}, worst_paths=[])
    print(f"\n{label} MISSES FIRST: {S['paths']} paths ({S['mixing_paths']} mixing): with a miss {S['miss']}, large "
          f"{S['large']}, defect (unmixed part > {DEFECT} dex off the full mechanism) {S['defect']}; worst {S['worst']:.3f} "
          f"dex (vs the full mechanism {S['worst_vs_full']:.3f}, vs VULCAN {S['worst_vs_vulcan']:.3f}; before the first "
          f"mix {S['worst_before_first_mix']:.2e}); cell-calls > 0.1 / 1 dex off the full mechanism {S['cellcalls_miss']}"
          f" / {S['cellcalls_large']}; stored states {S['stored']:,}: miss {S['states_miss']} large {S['states_large']}, "
          f"open (the full mechanism > 0.1 dex off VULCAN) {S['open_states']} (its worst {S['full_vs_vulcan_worst']:.3f})")
    print(f"  validity: reconstruction defects {S['defect_cellcalls']} cell-calls, chemistry still failed "
          f"{S['failed_cellcalls']}, non-finite {S['nonfinite']}; element drift vs the full mechanism "
          f"{S['element_drift_dex']:.1e} dex; spill path {S['spill_cellcalls']} cell-calls, worst difference there "
          f"{S['spill_worst_vs_full']:.3f} dex; layer counts {counts}; {S['ms_per_call']:.1f} ms per call, "
          f"{S['wall_s']:.0f} s")
    lab = np.char.add(np.char.add(meta["src"].astype(str), " "), meta["kind"].astype(str))
    for key, vals in [("panel", lab)] + [(q, meta[q].astype(str)) for q in ("stratum", "start", "comp")]:
        parts = []
        for v in dict.fromkeys(vals):
            s = vals == v
            ws = w[s & mixed]
            parts.append(f"{v} {int(s.sum())} (miss {int(miss[s].sum())}, large {int(large[s].sum())}, defect "
                         f"{int(defect[s].sum())}, worst {ws.max() if len(ws) else 0.0:.3f})")
            S["groups"][f"{key} {v}"] = dict(paths=int(s.sum()), miss=int(miss[s].sum()), large=int(large[s].sum()),
                                             defect=int(defect[s].sum()), worst=float(ws.max()) if len(ws) else 0.0)
        print(f"  by {key}: " + "; ".join(parts))
    print("  worst paths:")
    for i in np.argsort(-np.where(mixed, w, o["pre"]))[:8]:
        vs = "full" if o["wc"][i] >= o["wv"][i] else "VULCAN"
        k, g = (o["kc"][i], o["gc"][i]) if vs == "full" else (o["kv"][i], o["gv"][i])
        d = dict(id=int(meta["id"][i]), src=str(meta["src"][i]), kind=str(meta["kind"][i]),
                 stratum=str(meta["stratum"][i]), start=str(meta["start"][i]), comp=str(meta["comp"][i]),
                 dex=float(w[i]), vs=vs, gas=GASES[int(g)], call=int(k), first_mix=int(min(o["first"][i], o["n"][i])),
                 before_first_mix=float(o["pre"][i]), calls=int(o["n"][i]), Tmin=float(o["T"][i]),
                 spill=int(o["spill"][i]))
        S["worst_paths"].append(d)
        print(f"    {d['id']} {d['src']} {d['stratum']} {d['start']} {d['comp']}: {d['dex']:.3f} dex vs {vs} in "
              f"{d['gas']} at call {d['call']} (first mix {d['first_mix']}, {d['calls']} calls, T min {d['Tmin']:.0f} K); "
              f"before the first mix {d['before_first_mix']:.2e}; spill cell-calls {d['spill']}")
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", required=True, help="the design's JSON file (or a name in maxichem/designs/)")
    ap.add_argument("--panels", default=PANELS, help=f"(b): folders under the truth data (default {PANELS})")
    ap.add_argument("--days", type=float, default=100.0, help="(b): model time per path (default 100 days)")
    ap.add_argument("--save", help="write the summaries to this JSON file")
    a = ap.parse_args()
    paths.need(paths.QUAL / "QC")
    t_all = time.time()
    dev = torch.device("cuda")
    sp = [str(q) for q in np.load(sorted((paths.QUAL / "QC").glob("*.npz"))[0])["species"]]
    gi = [sp.index(g) for g in GASES]
    net = KV.VulcanNetwork(f"{paths.VULCAN_THERMO}/NCHO_thermo_network.txt", f"{paths.VULCAN_THERMO}/NASA9", species=sp)
    ts = KT.TracerSolver(net, a.design, dev)
    kc = KC.CudaSolver(net, dev)
    print(f"design {ts.name}: {ts.K} carried species + {ts.M} rows = {ts.K + ts.M} tracers, {ts.H} hidden; "
          f"{torch.cuda.get_device_name(dev)}; ready in {time.time() - t_all:.0f} s", flush=True)
    out = dict(design=ts.name, file=str(a.design), results=[])
    # (a) the kill gate
    files = [next(f for f in paths.QUAL.glob(f"*/{i}.npz")) for i in KILL]
    cells = load(files, sp, gi, ncap=KILL_CALLS)
    for i, n in (KILL_QL, KILL_QLC):
        cells += load([next(paths.QUAL.glob(f"*/{i}.npz"))], sp, gi, ncap=n)
    ts.total.zero_()
    o = gate(cells, ts, kc, net, dev, "(a) kill gate")
    S = report(o, "(a) kill gate", dict(zip(KT.COUNTS, ts.total.tolist())))
    killed = S["miss"] + S["defect"]
    S["verdict"] = "KILLED" if killed else "PASSED"
    print(f"(a) kill gate: {S['verdict']} ({S['miss']} paths with a miss, {S['defect']} with a defect; "
          f"{time.time() - t_all:.0f} s since start)", flush=True)
    out["results"].append(S)
    if not killed:
        files = [f for p in a.panels.split(",") for f in sorted((paths.DATA / "truth" / p).glob("*.npz"))]
        t1 = time.time()
        cells = load(files, sp, gi, days=a.days)
        print(f"(b) {len(cells)} paths of {a.panels} to {a.days:g} days, loaded in {time.time() - t1:.0f} s", flush=True)
        ts.total.zero_()
        o = gate(cells, ts, kc, net, dev, "(b) panels")
        out["results"].append(report(o, f"(b) panels {a.panels}", dict(zip(KT.COUNTS, ts.total.tolist()))))
    ts.check()
    kc.check()
    out["wall_s"] = time.time() - t_all
    print(f"wall time {out['wall_s']:.0f} s", flush=True)
    if a.save:
        Path(a.save).write_text(json.dumps(out, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else v))
        print(f"saved {a.save}")


if __name__ == "__main__":
    main()
