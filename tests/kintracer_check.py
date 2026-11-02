"""The reduced-tracer layer (maxichem/kintracer.py) against its stored reference on one GPU; prints PASS / FAIL per case.

Reference (paths.TRACER_REF): the same layer written in torch (the projection's first iterations replayed from CUDA
graphs, the extra ones on the host in numpy, up to 300 in blocks of 10, then for a cell still above 1e-10 up to 400
steps on the dual objective from two starts, in numpy), run once for the designs N26 and N30 beside the full mechanism
(all 53 species advected) on exactly these inputs; every entry carries the digest of its inputs (the network, the
design's split and rows, the inputs) and a mismatch is refused (tests/libref.py).
  bed   a ring of GCM-like cells (5 planets x dt 25 / 100 / 500 s x 12 levels x 24 longitudes = 4,320 cells) from a
        spun-up state: zonal upwind advection and vertical exchange of every tracer before each call (linear in the
        specific moles; coefficients stored), T and P changing every call; 50 calls;
  qc    3 qual paths (4240050031 QCs, 4240000130 QC, 4240000338 QC) with their recorded mixes (the partner gas mixed by
        mass, f_m = f M_y / ((1 - f) M_x + f M_y), M_x the full mechanism's), 200 calls.
(a) Parity per design and case: the layer's call in specific moles (TracerSolver.step) replaying the reference's mixing.
    The layer's arithmetic differs from the reference's at rounding level, and a rounding difference can tip the
    chemistry's step control (an accepted / rejected step, a species at -atol) into another branch, after which the two
    differ at the integration tolerance (up to ~1e-3 relative). A cell is touched from the call on which the reference
    took the host iterations or the integration branched (accepted steps differ, or the next step size by more than 1e-6
    relative), and so is every cell whose gas it reached through the mixing. Pass: on untouched cells the 12 scored gases
    within 1e-8 dex (floors 1e-10 H2O CO CO2 CH4 NH3 HCN C2H2, 1e-8 H OH O, 1e-6 H2 N2) at every call, and the end state
    (all species, to 1e-8 relative + 1e-20; rows) within 1e-8; the extra iterations taken by the cells the reference
    sent to the host (at most 1e-3 of the untouched cell-calls differ: a residual at the 1e-10 threshold); on touched
    cells each cell's worst error against the full mechanism (same calls, 12 gases, floored) at most the reference's
    plus 1e-3 dex. Every call: the elements of the state before the chemistry equal those of the tracers (the carried
    species' atoms plus the element rows) to 1e-12 relative, and no reconstruction defect.
(b) The spill path: 200 cells whose budgets no mixture of the hidden species can meet (N atoms with almost no C, H, O)
    take the extra iterations, are counted as not converged and flagged, and the elements stay exact (1e-12).
(c) The GCM-facing call: the tracers of split(x) are those of x; one call through the wrapper equals the specific-mole
    call to 1e-12 relative, the mean molar mass is 1 / sum n, cp and the heating rate equal a numpy evaluation of the
    NASA-9 coefficients (1e-12 relative; the heating rate relative to the scale of its terms), heating off returns None;
    a photolysis argument raises NotImplementedError.
(d) Batch size and order: 4,320 mixed bed cells in one batch, in two, permuted, and in batches of 100, 3 and 1 cells
    give bitwise the same tracers, memory, step sizes, states and flags.
(e) A cell below 300 K is counted by the chemistry and check() raises.
(f) Far from the memory (N30): 216 bed cells keep their own memory while their hidden budgets become 1e-3 of their own
    plus 1e-6 mol/g of CH3OH's atoms and 1e-12 mol/g of N2H4's, so the consistent answer is almost all CH3OH, many orders
    above the memory (the projection without the dual-objective steps leaves about half of them on the spill path).
    Every cell converges (status 1, nothing counted as not converged; the dual-objective steps taken), each element
    row's remainder spilled into its atom is at most 1e-8 of its budget, the elements are exact (1e-12), no defect, and
    the state equals the reference's to 1e-6 relative on the species above 1e-12 of the cell's moles (both meet the 1e-8
    convergence test; a spill or a stalled projection differs by orders of magnitude); the cells permuted and one cell
    alone give bitwise the same states, status and flags.
    python tests/kintracer_check.py [VULCAN thermo folder] (about 1 min). Exit 0 = every case PASS."""
import argparse
import sys
import time

import numpy as np
import torch

from maxichem import kintracer as KT  # noqa: E402
from maxichem import kinvulcan as KV  # noqa: E402
import libref as LR  # noqa: E402
from maxichem import paths  # noqa: E402

GASES = ["H2O", "CO", "CO2", "CH4", "NH3", "HCN", "C2H2", "H", "OH", "O", "H2", "N2"]
FLOORS = np.log10([1e-10] * 7 + [1e-8] * 3 + [1e-6] * 2)
TOL, TRUTH_SLACK, ELTOL, NEED_SHARE = 1e-8, 1e-3, 1e-12, 1e-3
NBED, NQC = 50, 200
QCIDS = (4240050031, 4240000130, 4240000338)
dev = torch.device("cuda")
results = []


def verdict(name, ok, text):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {text}", flush=True)


def tt(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float64, device=dev)


def fl(lx):
    """Floored log10 of the 12 gases (numpy, last axis)."""
    return np.maximum(lx, FLOORS)


def layer(ts, C, R, G):
    """The state before the chemistry (one prepare()): relative element error against the tracers [B], defects [B]."""
    m, status, bad = ts.prepare(C, R, G)
    C53 = torch.zeros_like(m).index_copy(1, ts.ic, C)
    want = C53 @ ts.EL + R @ ts.RE
    return ((m @ ts.EL - want).abs() / want.abs().clamp(min=1e-300)).amax(1), bad, status


class BedMix:
    """The ring's mixing (stored coefficients): zonal upwind advection, then vertical exchange, per (planet, dt) block."""

    def __init__(self, geom, mixc):
        self.g = [int(q) for q in geom]
        Pn, D, L, I = self.g
        self.f, self.adn, self.aup = (tt(c).reshape(1, 1, D, L, 1, 1) for c in mixc)

    def __call__(self, Q):
        Pn, D, L, I = self.g
        q = Q.view(-1, Pn, D, L, I, Q.shape[1])
        q = q + self.f * (torch.roll(q, 1, dims=4) - q)
        dn = torch.zeros_like(q)
        dn[:, :, :, :-1] = q[:, :, :, 1:] - q[:, :, :, :-1]
        up = torch.zeros_like(q)
        up[:, :, :, 1:] = q[:, :, :, :-1] - q[:, :, :, 1:]
        q = q + self.adn * dn + self.aup * up
        return q.reshape(Q.shape)


def parity(name, ts, E, ncall, n0, h0, forcing, mix, spread, gi):
    """Replay one reference case through ts.step: forcing(k) -> T, P, dt; mix(k, C, R) -> C, R before call k;
    spread(touched) carries the touched mask through one call's mixing."""
    C = n0[:, ts.ic].contiguous()
    R = torch.where(ts.hid, n0, torch.zeros_like(n0)) @ ts.A
    G = n0[:, ts.ih].contiguous()
    h = h0.clone()
    B = C.shape[0]
    touched, branched = np.zeros(B, bool), np.zeros(B, bool)
    worst_lib, worst_ref = np.zeros(B), np.zeros(B)
    d_clean = d_touch = el_err = 0.0
    need_bad = defects = ncl = nun = 0
    for k in range(ncall):
        C, R = mix(k, C, R)
        before = spread(touched)                               # touched before this call's projection
        er, bad, _ = layer(ts, C, R, G)
        el_err = max(el_err, float(er.max()))
        defects += int(bad.sum())
        C, R, G, h, x, flag = ts.step(C, R, G, *forcing(k), h)
        hn, st = h.cpu().numpy(), ts.kc.steps.cpu().numpy()
        br = (st != E["steps"][k]) | (np.abs(hn - E["h"][k]) > 1e-6 * np.abs(E["h"][k]))   # the integration branched
        branched |= br
        touched = before | E["left"][k] | br
        lx = torch.log10(x[:, gi].clamp(min=1e-300)).cpu().numpy()
        d = np.abs(fl(lx) - fl(E["x12"][k])).max(1)
        clean = np.logical_not(touched)
        ncl += int(clean.sum())
        if clean.any():
            d_clean = max(d_clean, float(d[clean].max()))
        if touched.any():
            d_touch = max(d_touch, float(d[touched].max()))
        need = ((flag.cpu().numpy() >> 2) & 3) >= 1                # extra iterations taken (converged or not)
        un = np.logical_not(before)
        nun += int(un.sum())
        need_bad += int((need[un] != E["left"][k][un]).sum())
        t12 = fl(E["t12"][k])
        worst_lib = np.maximum(worst_lib, np.abs(fl(lx) - t12).max(1))
        worst_ref = np.maximum(worst_ref, np.abs(fl(E["x12"][k]) - t12).max(1))
    clean = np.logical_not(touched)
    ne = E["n_end"]
    xr = ne / ne.sum(1, keepdims=True)
    rel = np.abs(x.cpu().numpy() - xr) / (np.abs(xr) + 1e-20)
    rb = np.abs(R.cpu().numpy() - E["bud_end"]) / np.maximum(np.abs(E["bud_end"]), 1e-300)
    rel_end = float(rel[clean].max()) if clean.any() else 0.0
    rb_end = float(rb[clean].max()) if clean.any() else 0.0
    slack = float((worst_lib - worst_ref)[touched].max()) if touched.any() else 0.0
    tot = dict(zip(KT.COUNTS, ts.total.tolist()))
    ok = (d_clean <= TOL and rel_end <= TOL and rb_end <= TOL and need_bad <= max(5, NEED_SHARE * nun)
          and slack <= TRUTH_SLACK and el_err <= ELTOL and defects == 0)
    verdict(name, ok, f"{ncall} calls x {B} cells; untouched cell-calls {ncl}: worst 12-gas difference {d_clean:.2e} dex, "
            f"end state {rel_end:.1e} rel. (+1e-20), rows {rb_end:.1e} rel.; extra iterations differing from the "
            f"reference's host iterations {need_bad} of {nun} cell-calls; branched cells {int(branched.sum())}, touched "
            f"{int(touched.sum())}: difference {d_touch:.2e} dex, worst error vs the full mechanism minus the reference's "
            f"{slack:+.2e} dex (slack {TRUTH_SLACK:g}); elements before the chemistry {el_err:.1e} rel.; reconstruction "
            f"defects {defects}; the layer's counts so far: extra iterations {tot['extra']}, not converged "
            f"{tot['unconverged']} (reference: host iterations {int(E['left'].sum())}, not converged {int(E['fail'].sum())})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vulcan_thermo", nargs="?",
                    help="folder with NCHO_thermo_network.txt and NASA9/ (default: $VULCAN_THERMO)")
    a = ap.parse_args()
    a.vulcan_thermo = a.vulcan_thermo or str(paths.VULCAN_THERMO)
    paths.need(paths.QUAL / "QC", paths.TRACER_REF)
    t0 = time.time()
    sp = [str(s) for s in np.load(sorted((paths.QUAL / "QC").glob("*.npz"))[0])["species"]]
    nf, nasa = f"{a.vulcan_thermo}/NCHO_thermo_network.txt", f"{a.vulcan_thermo}/NASA9"
    net = KV.VulcanNetwork(nf, nasa, species=sp)
    sig = LR.netsig(nf, nasa, sp)
    gi = [sp.index(g) for g in GASES]
    REF = paths.TRACER_REF
    bed = LR.arrays("bed", REF)
    geom, mixc, fb = bed["geom"], bed["mixc"], bed["forcing"]
    bm = BedMix(geom, mixc)
    qz = [np.load(next(paths.QUAL.glob(f"*/{i}.npz"))) for i in QCIDS]
    qin = [np.asarray(z[q]) for z in qz for q in ("T", "Pbar", "dt", "x0", "mix_at", "mix_f", "mix_x")]
    # the QC paths' mixes in the reference's order: by call, stable over the paths
    mp = np.concatenate([np.full(len(z["mix_at"]), p) for p, z in enumerate(qz)])
    at = np.concatenate([z["mix_at"] for z in qz]).astype(int)
    o = np.argsort(at, kind="stable")
    y = np.concatenate([z["mix_x"] for z in qz])[o]
    MP, MF, MY = torch.tensor(mp[o], device=dev), tt(np.concatenate([z["mix_f"] for z in qz])[o]), tt(
        y / y.sum(1, keepdims=True))
    off = np.searchsorted(at[o], np.arange(NQC + 1))
    T = tt(np.stack([z["T"] for z in qz]))
    P = tt(np.stack([z["Pbar"] for z in qz]) * 1e6)
    DT = tt(np.stack([z["dt"] for z in qz]))
    x0 = np.stack([z["x0"] for z in qz])
    x0 = tt(x0 / x0.sum(1, keepdims=True))
    solvers = {}
    for nm in ("N26", "N30"):
        ts = solvers[nm] = KT.TracerSolver(net, nm, dev)
        _keep_steps(ts.kc)
        car, hid, AT = ts.car.cpu().numpy(), ts.hid.cpu().numpy(), ts.AT.cpu().numpy()
        E = LR.entry(f"bed/{nm}", sig, car, hid, AT, bed["n0"], bed["h0"], fb, geom, mixc, str(NBED), path=REF)

        def mix_b(k, C, R, ts=ts):
            Q = bm(torch.cat([C, R], 1))
            return Q[:, :ts.K].contiguous(), Q[:, ts.K:].contiguous()

        def spread_b(m):
            return (bm(torch.as_tensor(m, dtype=torch.float64, device=dev)[:, None])[:, 0] > 0).cpu().numpy()

        parity(f"(a) bed {nm}", ts, E, NBED, tt(bed["n0"]), tt(bed["h0"]),
               lambda k: tuple(tt(fb[k][i]) for i in range(3)), mix_b, spread_b, gi)
        # the 3 QC paths, with the full mechanism beside them (its M_x sets each mix's mass fraction)
        E = LR.entry(f"qc/{nm}", sig, car, hid, AT, *qin, str(NQC), path=REF)
        n0 = x0 / (x0 @ ts.MW)[:, None]
        st = dict(n=n0.clone(), h=torch.full((3,), 1e-10, dtype=torch.float64, device=dev))

        def mix_q(k, C, R, ts=ts, st=st):
            nc = st["n"]
            if k > 0:                                          # the full mechanism's call k - 1
                xo, st["h"], _, _ = ts.kc(nc / nc.sum(1, keepdim=True), T[:, k - 1], P[:, k - 1], DT[:, k - 1], st["h"])
                nc = xo / (xo @ ts.MW)[:, None]
            a, b = off[k], off[k + 1]
            if b > a:
                p, f, y = MP[a:b], MF[a:b], MY[a:b]
                yn = y / (y @ ts.MW)[:, None]
                Mx, My = 1.0 / nc[p].sum(1), y @ ts.MW
                fm = (f * My / ((1.0 - f) * Mx + f * My))[:, None]
                nc, C, R = nc.clone(), C.clone(), R.clone()
                nc[p] = (1.0 - fm) * nc[p] + fm * yn
                C[p] = (1.0 - fm) * C[p] + fm * yn[:, ts.ic]
                R[p] = (1.0 - fm) * R[p] + fm * (torch.where(ts.hid, yn, torch.zeros_like(yn)) @ ts.A)
            st["n"] = nc
            return C, R

        parity(f"(a) qc {nm}", ts, E, NQC, n0, torch.full((3,), 1e-10, dtype=torch.float64, device=dev),
               lambda k: (T[:, k], P[:, k], DT[:, k]), mix_q, lambda m: m, gi)
    # ---- (b) the spill path
    ts = solvers["N26"]
    nb = tt(bed["n0"][:200])
    C, G = nb[:, ts.ic].contiguous(), nb[:, ts.ih].contiguous()
    R = torch.where(ts.hid, nb, torch.zeros_like(nb)) @ ts.A
    rows = [ts.species[i] if i >= 0 else None for i in ts.spill]      # the element of each element row
    R[:, rows.index("N")] += 1e-4                                     # N atoms the hidden pool cannot hold ...
    for e in ("C", "H", "O"):
        R[:, rows.index(e)] = 1e-20                                   # ... with almost no C, H, O beside them
    ts.count.zero_()
    er, bad, status = layer(ts, C, R, G)
    cnt = dict(zip(KT.COUNTS, ts.count.tolist()))
    el = float(er.max())
    ok = (cnt["unconverged"] == 200 and cnt["extra"] == 200 and bool((status == 2).all()) and el <= ELTOL
          and not bool(bad.any()))
    verdict("(b) spill path", ok, f"200 infeasible cells: not converged {cnt['unconverged']}, flagged "
            f"{int((status == 2).sum())}, extra iterations {cnt['extra']}; elements {el:.1e} rel.; defects {int(bad.sum())}")
    # ---- (c) the GCM-facing call
    ts = solvers["N30"]
    nb = tt(bed["n0"])
    xb = nb / nb.sum(1, keepdim=True)
    q, mem = ts.split(xb)
    rq = lambda a, b: float(((a - b).abs() / b.abs().clamp(min=1e-300)).max())  # noqa: E731
    C, R, G = nb[:, ts.ic], torch.where(ts.hid, nb, torch.zeros_like(nb)) @ ts.A, nb[:, ts.ih]
    r_split = max(rq(q[:, :ts.K] / ts.mwc, C), rq(q[:, ts.K:] / ts.unit, R), rq(mem / ts.mwh, G))
    fo = tuple(tt(fb[0][i]) for i in range(3))
    h0 = tt(bed["h0"])
    q2, mem2, h2, mu, cp, Q, flag, counts = ts(q, mem, *fo, h0, heating=True)
    q2b, _, _, _, _, Qb, _, _ = ts(q, mem, *fo, h0)
    C1 = q[:, :ts.K] / ts.mwc
    R1 = q[:, ts.K:] / ts.unit
    G1 = mem / ts.mwh
    n1 = ts.prepare(C1, R1, G1)[0]
    C3, R3, G3, h3, x3, flag3 = ts.step(C1, R1, G1, *fo, h0)
    r_call = max(rq(q2[:, :ts.K] / ts.mwc, C3), rq(q2[:, ts.K:] / ts.unit, R3), rq(mem2 / ts.mwh, G3))
    n3 = x3 / (x3 @ ts.MW)[:, None]
    r_mu = rq(mu, 1.0 / n3.sum(1))
    same = bool(torch.equal(h2, h3) and torch.equal(flag, flag3) and torch.equal(q2, q2b) and Qb is None)
    # cp and the heating rate from the NASA-9 coefficients, evaluated here in numpy (erg g^-1 K^-1, erg g^-1 s^-1)
    Tn, dtn = fo[0].cpu().numpy(), fo[2].cpu().numpy()
    lo = (Tn < 1000.0)[:, None, None]
    a = np.where(lo, net.lo[None], net.hi[None])
    t = Tn[:, None]
    cpr = sum(a[..., j] * t ** (j - 2) for j in range(7))
    hrt = (-a[..., 0] / t ** 2 + a[..., 1] * np.log(t) / t + sum(a[..., j] * t ** (j - 2) / (j - 1) for j in range(2, 7))
           + a[..., 7] / t)
    RG = KV.KB * 6.02214076e23
    n1n, n3n = n1.cpu().numpy(), (x3 / mu[:, None]).cpu().numpy()   # the layer's own n after the call
    cp_np = RG * (n3n * cpr).sum(1)
    Q_np = -RG * Tn * (hrt * (n3n - n1n)).sum(1) / dtn
    r_cp = float(np.abs(cp.cpu().numpy() / cp_np - 1).max())
    hs = RG * Tn * np.abs(hrt * (n3n - n1n)).sum(1) / dtn               # scale of the sum (its terms cancel)
    r_Q = float((np.abs(Q.cpu().numpy() - Q_np) / hs).max())
    try:
        ts(q, mem, *fo, h0, photo=torch.zeros(q.shape[0], 1, dtype=torch.float64, device=dev))
        raised = False
    except NotImplementedError:
        raised = True
    ok = (r_split <= 1e-12 and r_call <= 1e-12 and r_mu <= 1e-12 and same and raised and r_cp <= 1e-12
          and r_Q <= 1e-12)
    verdict("(c) GCM-facing call", ok,
            f"split {r_split:.1e} rel.; wrapper vs specific moles {r_call:.1e} rel., step sizes, flags and heating off "
            f"as expected {same}; mean molar mass vs 1 / sum n {r_mu:.1e} rel.; cp vs numpy NASA-9 {r_cp:.1e} rel. "
            f"({cp_np.min():.3e}-{cp_np.max():.3e} erg/g/K), heating rate {r_Q:.1e} of its terms' scale; photolysis "
            f"argument raises {raised}; counts {dict(zip(KT.COUNTS, counts.tolist()))}")
    # ---- (d) batch size and order: the same cells alone, split, permuted and in a small batch, bitwise equal
    ts = solvers["N26"]
    nb = tt(bed["n0"])
    Q0 = bm(torch.cat([nb[:, ts.ic], torch.where(ts.hid, nb, torch.zeros_like(nb)) @ ts.A], 1))
    C, R, G = Q0[:, :ts.K].contiguous(), Q0[:, ts.K:].contiguous(), nb[:, ts.ih].contiguous()
    B = C.shape[0]

    def run(ix):
        return ts.step(C[ix], R[ix], G[ix], fo[0][ix], fo[1][ix], fo[2][ix], h0[ix])

    full_out = run(torch.arange(B, device=dev))
    nx = int(ts.count[0])
    ext = torch.nonzero(((full_out[5] >> 2) & 3) >= 1).squeeze(1)
    perm = torch.randperm(B, generator=torch.Generator().manual_seed(0)).to(dev)
    cases = {"split 2,000 + 2,320": [torch.arange(0, 2000, device=dev), torch.arange(2000, B, device=dev)],
             "permuted": [perm], "3 cells (2 with extra iterations)": [torch.cat([ext[:2], ext[:1] * 0 + 7])],
             "100 cells": [torch.arange(1000, 1100, device=dev)], "1 cell": [ext[:1]]}
    bad = []
    for nm, parts in cases.items():
        for ix in parts:
            if not all(torch.equal(a, b[ix]) for a, b in zip(run(ix), full_out)):
                bad.append(nm)
    verdict("(d) batch size and order", not bad and nx > 0,
            f"{B} mixed bed cells ({nx} with extra iterations) vs " + ", ".join(cases) + f": not bitwise {bad or 'none'}")
    # ---- (e) 300 K
    ts = solvers["N26"]
    q, mem = ts.split(xb[:4])
    ts(q, mem, tt([290.0, 1000.0, 1500.0, 2000.0]), fo[1][:4], fo[2][:4], h0[:4])
    cold = int(ts.count[KT.COUNTS.index("cold")])
    try:
        ts.check()
        raised = False
    except ValueError:
        raised = True
    verdict("(e) 300 K", cold == 1 and raised, f"cells below 300 K counted {cold}; check() raises {raised}")
    # ---- (f) far from the memory: a CH3OH-dominated hidden budget
    ts = solvers["N30"]
    car, hid, AT = ts.car.cpu().numpy(), ts.hid.cpu().numpy(), ts.AT.cpu().numpy()
    nb = bed["n0"][::20]
    Cf, Gf = nb[:, car], nb[:, hid]
    Rf = (1e-3 * (np.where(hid[None], nb, 0.0) @ AT.T) + 1e-6 * AT[:, ts.species.index("CH3OH")][None]
          + 1e-12 * AT[:, ts.species.index("N2H4")][None])
    F = LR.entry("far/N30", sig, car, hid, AT, Cf, Rf, Gf, path=REF)
    ts.count.zero_()
    mt, status, bad = ts.prepare(tt(Cf), tt(Rf), tt(Gf))
    cnt = dict(zip(KT.COUNTS, ts.count.tolist()))
    want = tt(Cf) @ ts.EL[ts.ic] + tt(Rf) @ ts.RE
    er = ((mt @ ts.EL - want).abs() / want.abs().clamp(min=1e-300)).amax(1)
    m = mt.cpu().numpy()
    held = np.where(hid[None], m, 0.0) @ AT.T
    spill = float((np.abs(Rf - held) / Rf).max())
    m[:, ts.species.index("He")] = 0.0
    ref = F["n"]
    big = np.maximum(np.abs(m), np.abs(ref)) > 1e-12 * ref.sum(1, keepdims=True)
    dif = float(np.where(big, np.abs(m - ref) / np.maximum(np.abs(ref), 1e-300), 0.0).max())
    B, stn = Cf.shape[0], status.cpu().numpy()
    perm = torch.randperm(B, generator=torch.Generator().manual_seed(1)).to(dev)
    same = all(torch.equal(a[perm], b) for a, b in zip((mt, status, bad), ts.prepare(tt(Cf)[perm], tt(Rf)[perm],
                                                                                    tt(Gf)[perm])))
    same = same and all(torch.equal(a[5:6], b) for a, b in zip((mt, status, bad), ts.prepare(tt(Cf[5:6]), tt(Rf[5:6]),
                                                                                           tt(Gf[5:6]))))
    ok = (bool(F["conv"].all()) and bool((stn == 1).all()) and cnt["unconverged"] == 0 and cnt["dual"] > 0
          and spill <= 1e-8 and float(er.max()) <= ELTOL and not bool(bad.any()) and dif <= 1e-6 and same)
    verdict("(f) far from the memory", ok, f"{B} cells with a CH3OH-dominated hidden budget: status 1 for "
            f"{int((stn == 1).sum())}, not converged {cnt['unconverged']}, dual-objective steps {cnt['dual']} (reference: "
            f"converged {int(F['conv'].sum())}, dual-objective steps {int(F['p3'].sum())}); spilled share of each element "
            f"row {spill:.1e}; state vs the reference {dif:.1e} rel.; elements {float(er.max()):.1e} rel.; defects "
            f"{int(bad.sum())}; permuted and one-cell batches bitwise {same}")
    print(f"{sum(results)} of {len(results)} cases PASS ({time.time() - t0:.0f} s)", flush=True)
    sys.exit(0 if all(results) else 1)


def _keep_steps(kc):
    """Keep each call's accepted steps on the solver (kc.steps), for the parity case."""
    launch = kc.launch

    def spy(*a, **k):
        out = launch(*a, **k)
        kc.steps = out[3]
        return out
    kc.launch = spy


if __name__ == "__main__":
    main()
