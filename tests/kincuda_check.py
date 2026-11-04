"""The whole-call CUDA kernel (maxichem/kincuda.py) against the stored reference on one GPU; prints PASS / FAIL per case.

Reference = the call computed by batched fp64 Rodas3 in torch with the kernel's numerics (rates from T and P every call,
exact Jacobian, LU without row exchanges, rtol 1e-3, atol kincuda.ATOL, the step control of maxichem/kincuda.py), stored for
exactly these inputs (tests/libref.py, paths.LIBREF; a check of the inputs' digest refuses others). Each solver
advances its own state from the same inputs.
(a) Parity. Pass: floored difference on the 12 scored gases (floors 1e-10 H2O CO CO2 CH4 NH3 HCN C2H2, 1e-8 H OH O,
    1e-6 H2 N2) <= 1e-6 dex on all but one cell-call in 100,000 (at least one per case) and <= 1e-5 dex on every
    cell-call (the reference's batched arithmetic rounds differently), accepted-step counts identical on all but
    <= 1e-4 of the cell-calls, no non-finite state, no unfinished call, no cell recomputed by the guard.
    The network is VULCAN-JAX's NCHO_photo_network.txt (thermal part; photolysis in the photolysis cases). The
    recorded starts are 53-species states mapped onto its 69 species by name (the others start at 0).
    - yardstick cold calls: 4,096 recorded starts at their T, P, dt 1 and 500 s, starting step 1e-10 s;
    - scrambled starts: the same kind of starts, each species times 10^(sigma N(0, 1)), sigma 0 / 1 / 2 dex, at
      dt 0.01 / 1 / 50 / 500 s;
    - a 2,000-call chain: decision panel P's 256 paths x 4 offsets (1,024 cells), T, P and dt of the recorded forcing
      every call (mixing events not applied), step sizes carried from call to call;
    - photolysis (CudaSolver photo=True): the cold starts and the sigma-1 scrambled starts at dt 1 and 500 s with the
      J of 1-D-model layers stored in the reference (the reference adds the branches by its own parser), the cold
      starts at dt 500 s with J = 0 against the thermal reference, a lit cell forced to fail (max_steps 10) and
      recomputed on the device, a lit 300-call panel-P chain (1,024 cells), and bad J (negative, NaN, inf: counted,
      not stepped, returned as the normalized input, the other cells bitwise unchanged; a cell below 300 K with a bad J counted cold).
(b) Guard and recovery. A cell with a NaN input is counted, not recomputed, and left non-finite; a cell forced to fail
    (max_steps 10 for a cold start at dt 500 s among the reference's settled cells) and, with the rounding allowance
    DRIFT set to 0, the cells whose element shares move by more than the clamping allowance are recomputed on the
    device: within 1e-6 dex of the reference call with the recovery's tries (REC_STEPS x max_steps; the reference is
    not bitwise the kernel: its sums run in another order), counted as recovered or still failed, the other cells
    bitwise those of a solver that flags none; a cell below 300 K is counted and check() raises.
(c) Batch size and order: 128 cells and one cell alone bitwise the same cells inside 8,192, and 8,192 cells in a
    permuted order bitwise the same per cell (x, h, done, steps, flag).
    python tests/kincuda_check.py <VULCAN-JAX thermo folder: NCHO_photo_network.txt, NASA9/>
(about 1 min on an idle GPU). Exit 0 = every case PASS."""
import argparse
import sys
import time

import numpy as np
import torch

from maxichem import kincuda as KC  # noqa: E402
from maxichem import kinvulcan as KV  # noqa: E402
import libref as LR  # noqa: E402
from maxichem import paths  # noqa: E402

YARD = paths.YARD / "design"
PANEL = paths.DESIGNS / "decision/P"
paths.need(YARD, PANEL, paths.LIBREF)
GASES = ["H2O", "CO", "CO2", "CH4", "NH3", "HCN", "C2H2", "H", "OH", "O", "H2", "N2"]
FLOORS = [1e-10] * 7 + [1e-8] * 3 + [1e-6] * 2
TOL, MISMATCH = 1e-6, 1e-4
dev = torch.device("cuda")
results = []


def tt(a):
    return torch.tensor(np.asarray(a), dtype=torch.float64, device=dev)


def full(B, v):
    return torch.full((B,), v, dtype=torch.float64, device=dev)


def floored(a, b):
    """Per cell: worst |log10 max(a, S) - log10 max(b, S)| over the scored gases (columns in GASES order)."""
    fl = torch.log10(torch.tensor(FLOORS, dtype=torch.float64, device=dev))
    la = torch.maximum(torch.log10(a.clamp(min=1e-300)), fl)
    lb = torch.maximum(torch.log10(b.clamp(min=1e-300)), fl)
    return (la - lb).abs().max(1).values.nan_to_num(nan=np.inf)


def tensors(e):
    """A stored reference call as tensors: x on the scored gases, accepted steps, finished, finite."""
    return dict(x=tt(e["x"]), steps=tt(e["steps"]), done=torch.tensor(e["done"], device=dev),
                finite=torch.tensor(e["finite"], device=dev))


def verdict(name, ok, text):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {text}", flush=True)


class Tally:
    """Parity counts over the calls of one case."""

    def __init__(self, frac=1e-5, cap=1e-5):
        """frac: the share of cell-calls allowed above TOL (at least one per case), none above cap (a reference whose
        step sequence branches differently, or whose rounding differs, can differ by slightly more than TOL)."""
        self.worst, self.calls, self.mis, self.nf, self.unf, self.redo = 0.0, 0, 0, [0, 0], [0, 0], 0
        self.frac, self.cap, self.over = frac, cap, 0

    def add(self, gi, ref, mod, kc):
        xb, _, db, nb = mod
        e = floored(ref["x"], xb[:, gi])
        self.worst = max(self.worst, float(e.max()))
        self.over += int((e > TOL).sum())
        self.calls += xb.shape[0]
        self.mis += int((ref["steps"] != nb).sum())
        for i, (fin, d) in enumerate(((ref["finite"], ref["done"]), (torch.isfinite(xb).all(1), db))):
            self.nf[i] += int(torch.logical_not(fin).sum())
            self.unf[i] += int(torch.logical_not(d).sum())
        self.redo += kc.fallback["recomputed"]

    def verdict(self, name):
        ok = (self.worst <= self.cap and self.over <= max(1.0, self.frac * self.calls) and self.mis <= MISMATCH * self.calls
              and self.nf == [0, 0] and self.unf == [0, 0] and self.redo == 0)
        verdict(name, ok, f"worst {self.worst:.2e} dex ({self.over} cell-calls above {TOL:g}); step-count mismatches "
                          f"{self.mis} of {self.calls} cell-calls; "
                          f"non-finite ref/kernel {self.nf[0]}/{self.nf[1]}; unfinished {self.unf[0]}/{self.unf[1]}; "
                          f"recomputed by the guard {self.redo}")


SP53 = [str(s) for s in np.load(YARD / "units.npz")["species53"]]
SP = []                                                  # the network's species (main)


def mapped(x53):
    """53-species states onto the network's species by name (the others 0), renormalized."""
    x = np.zeros((len(x53), len(SP)))
    for j, q in enumerate(SP):
        if q in SP53:
            x[:, j] = x53[:, SP53.index(q)]
    return x / x.sum(1, keepdims=True)


def yardstick(B, seed, sigma):
    """B recorded starts (x, T, P) mapped onto the network, each species scaled by 10^(sigma N(0, 1)), renormalized."""
    u, r = np.load(YARD / "units.npz"), np.load(YARD / "rows.npz")
    row_of_unit = {int(q): i for i, q in enumerate(r["unit"])}
    rng = np.random.default_rng(seed)
    units = rng.choice(len(u["t0"]), B, replace=True)
    idx = np.array([row_of_unit[int(q)] for q in units])
    x = mapped(u["x53"][units])
    if sigma:
        x = x * 10.0 ** (sigma * rng.standard_normal(x.shape))
        x = x / x.sum(1, keepdims=True)
    return tt(x), tt(r["T"][idx]), tt(r["P_barye"][idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vulcan_thermo", help="VULCAN-JAX's thermo folder (NCHO_photo_network.txt and NASA9/)")
    a = ap.parse_args()
    nf, nasa = f"{a.vulcan_thermo}/NCHO_photo_network.txt", f"{a.vulcan_thermo}/NASA9"
    net = KV.VulcanNetwork(nf, nasa)
    sp = list(net.species)
    SP[:] = sp
    assert float(LR.load()["atol"]) == KC.ATOL, "the reference was made at another atol"
    assert [str(q) for q in LR.load()["species"]] == sp, "the reference's species order differs from the reader's"
    sig = LR.netsig(nf, nasa, sp)
    gi = [sp.index(g) for g in GASES]
    t0 = time.time()
    kc = KC.CudaSolver(net, dev)
    print(f"kernel built and compiled in {time.time() - t0:.1f} s; {kc.wpb} warps per block, {kc.smem} B shared "
          f"memory, {kc.max_blocks} resident blocks", flush=True)

    # (a) parity
    B = 4096
    x0, T, P = yardstick(B, 0, 0.0)
    for dt in (1.0, 500.0):
        t = Tally()
        args = (x0, T, P, full(B, dt), full(B, 1e-10))
        t.add(gi, tensors(LR.entry(f"call/cold{dt:g}", sig, *args)), kc(*args), kc)
        t.verdict(f"yardstick cold calls, dt {dt:g} s")
    for sigma in (0.0, 1.0, 2.0):
        x0, T, P = yardstick(B, int(10 + sigma), sigma)
        for dt in (0.01, 1.0, 50.0, 500.0):
            t = Tally()
            args = (x0, T, P, full(B, dt), full(B, 1e-10))
            t.add(gi, tensors(LR.entry(f"call/scr{sigma:g}_{dt:g}", sig, *args)), kc(*args), kc)
            t.verdict(f"scrambled starts sigma {sigma:g} dex, dt {dt:g} s")
    r, s, F = (np.load(PANEL / f"{f}.npz") for f in ("rows", "starts", "forcing"))
    assert [str(q) for q in s["species53"]] == SP53
    B, NC = 1024, 2000
    path, of = np.arange(B) % 256, (np.arange(B) // 256) * 1500
    Tc, Pc, dc = (np.stack([np.asarray(q, np.float64)[path, of + k] for k in range(NC)])
                  for q in (F["T"], F["Pbar"] * 1e6, F["dt"]))
    xb = tt(mapped(s["x53"][r["unit"][path]]))
    hb = full(B, 1e-10)
    R = tensors(LR.entry("chain", sig, xb, hb, Tc, Pc, dc))
    t, tb = Tally(frac=1e-5, cap=1e-5), 0.0
    for k in range(NC):
        torch.cuda.synchronize(); c0 = time.perf_counter()
        rb = kc(xb, tt(Tc[k]), tt(Pc[k]), tt(dc[k]), hb)
        torch.cuda.synchronize(); tb += time.perf_counter() - c0
        t.add(gi, {q: v[k] for q, v in R.items()}, rb, kc)
        xb, hb = rb[:2]
    t.verdict(f"panel P chain, {NC} calls x {B} cells (the kernel on its own state; ms per call kernel "
              f"{tb / NC * 1e3:.2f}, GPU may be shared)")

    # photolysis
    kp = KC.CudaSolver(net, dev, photo=True)
    J = tt(LR.arrays("photo")["J"])
    B = J.shape[0]
    for name, seed, sigma in (("cold", 0, 0.0), ("scr1", 11, 1.0)):
        x0, T, P = yardstick(B, seed, sigma)
        for dt in (1.0, 500.0):
            t = Tally()
            args = (x0, T, P, full(B, dt), full(B, 1e-10))
            t.add(gi, tensors(LR.entry(f"photo/{name}{dt:g}", sig, *args, J)), kp(*args, J=J), kp)
            t.verdict(f"photolysis, {name} starts, dt {dt:g} s")
    x0, T, P = yardstick(B, 0, 0.0)
    args = (x0, T, P, full(B, 500.0), full(B, 1e-10))
    t = Tally()
    t.add(gi, tensors(LR.entry("call/cold500", sig, *args)), kp(*args, J=torch.zeros_like(J)), kp)
    t.verdict("photolysis with J = 0 against the thermal reference, cold starts, dt 500 s")
    i = torch.tensor([7], device=dev)                    # a lit cell forced to fail and recomputed on the device
    y, _, d, n, flag = kp.launch(x0[i], T[i], P[i], full(1, 500.0), full(1, 1e-10), max_steps=10, J=J[i])
    fb = dict(kp.fallback)
    R = tensors(LR.entry("photo/forced", sig, x0[i], T[i], P[i], full(1, 500.0), full(1, 1e-10), J[i],
                         np.array(10 * KC.REC_STEPS)))
    e = float(floored(y[:, gi], R["x"]).max())
    ok = (fb["unfinished"] == 1 and fb["recomputed"] == 1 and e <= TOL and bool(torch.equal(n, R["steps"]))
          and bool(torch.equal(d, R["done"])))
    verdict("photolysis: forced failure (max_steps 10) recomputed", ok,
            f"counts {fb}; {e:.1e} dex from the reference call with {10 * KC.REC_STEPS} tries, finished {bool(d[0])}")
    Bc, NC = 1024, 300                                   # a lit 300-call chain (panel P's paths and forcing)
    r, s, F = (np.load(PANEL / f"{f}.npz") for f in ("rows", "starts", "forcing"))
    path, of = np.arange(Bc) % 256, (np.arange(Bc) // 256) * 1500
    Tc, Pc, dc = (np.stack([np.asarray(q, np.float64)[path, of + k] for k in range(NC)])
                  for q in (F["T"], F["Pbar"] * 1e6, F["dt"]))
    xb, hb = tt(mapped(s["x53"][r["unit"][path]])), full(Bc, 1e-10)
    Jc = J[:Bc]                                           # rows i % L of the layers, as the reference
    R = tensors(LR.entry("photo/chain", sig, xb, hb, Tc, Pc, dc, Jc))
    t = Tally()
    for k in range(NC):
        rb = kp(xb, tt(Tc[k]), tt(Pc[k]), tt(dc[k]), hb, J=Jc)
        t.add(gi, {q: v[k] for q, v in R.items()}, rb, kp)
        xb, hb = rb[:2]
    t.verdict(f"photolysis: lit chain, {NC} calls x {Bc} cells")
    xs = x0[:64].clone()                                  # bad photolysis rates: counted, not stepped (x normalized)
    Jb = J[:64].clone()
    Jb[3, 0], Jb[5, 1], Jb[9, 2] = -1.0, float("nan"), float("inf")
    Tb = T[:64].clone()
    Tb[11] = 250.0
    Jb[11, 0] = float("nan")                             # a cold cell with a bad J: counted cold
    y, _, d, _ = kp(xs, Tb, P[:64], full(64, 1.0), full(64, 1e-10), J=Jb)
    fb = dict(kp.fallback)
    good = torch.ones(64, dtype=torch.bool, device=dev)
    good[[3, 5, 9, 11]] = False
    yr, _, dr, _ = kp(xs[good], Tb[good], P[:64][good], full(int(good.sum()), 1.0), full(int(good.sum()), 1e-10),
                      J=Jb[good])
    ok = (fb["bad_input"] == 3 and fb["cold"] == 1 and not bool(d[[3, 5, 9, 11]].any())
          and bool(torch.allclose(y[[3, 5, 9]], xs[[3, 5, 9]] / xs[[3, 5, 9]].sum(1, keepdim=True), rtol=1e-14, atol=0))
          and bool(torch.equal(y[good], yr))
          and bool(torch.equal(d[good], dr)))
    verdict("photolysis: bad J counted, cold precedence", ok,
            f"counts {fb}; rows 3, 5, 9 (J -1, NaN, inf) not finished and returned as the normalized input, row 11 (250 K, NaN J) cold; "
            f"the other 60 cells bitwise those of a batch without them")
    try:
        kp.check()
    except ValueError:
        pass                                             # the cold cell above: check() raises, as intended

    # (b) guard
    B = 64
    x0, T, P = yardstick(B, 1, 0.0)
    G = LR.entry("guard", sig, x0, T, P)
    xs, hs = tt(G["xs"]), tt(G["hs"])                                     # the reference's settled states (100 s)
    keep = torch.arange(B, device=dev) != 3
    x = xs.clone()
    x[3, 5] = float("nan")
    y, _, d, _ = kc(x, T, P, full(B, 1.0), hs)
    fb = dict(kc.fallback)
    ya = tensors(LR.arrays("guard/nan"))["x"][keep]                     # the reference from xs, hs at dt 1 s
    e = float(floored(y[keep][:, gi], ya).max())
    ok = (fb["bad_input"] == 1 and fb["recomputed"] == 0 and not bool(torch.isfinite(y[3]).all()) and not bool(d[3])
          and e <= TOL and bool(d[keep].all()))
    verdict("guard: NaN input", ok, f"counts {fb}; row 3 non-finite and unfinished, the other {B - 1} cells within "
                                    f"{e:.1e} dex of the reference")

    x, h, dt = xs.clone(), hs.clone(), full(B, 1.0)
    x[7], h[7], dt[7] = x0[7], 1e-10, 500.0                             # cell 7: cold start at 500 s
    y, _, d, n, flag = kc.launch(x, T, P, dt, h, max_steps=10)
    fb = dict(kc.fallback)
    i = torch.tensor([7], device=dev)
    R = tensors(LR.entry("guard/forced", sig, x[i], T[i], P[i], dt[i], h[i], np.array(10 * KC.REC_STEPS)))
    yr, dr, nr = R["x"], R["done"], R["steps"]
    e = float(floored(y[i][:, gi], yr).max())
    ok = (fb["unfinished"] == 1 and fb["recomputed"] == 1 and fb["recovered"] + fb["still_failed"] == 1 and e <= TOL
          and bool(torch.equal(n[i], nr)) and bool(torch.equal(d[i], dr)) and int(flag[7]) == (1 if bool(dr) else 2)
          and bool(d[torch.arange(B, device=dev) != 7].all()))
    verdict("guard: forced failure (max_steps 10)", ok,
            f"counts {fb}; cell 7 recomputed with {10 * KC.REC_STEPS} tries ({e:.1e} dex from the reference call "
            f"with those tries, {int(nr)} accepted steps, finished {bool(dr)}, flag {int(flag[7])})")

    kz = KC.CudaSolver(net, dev, drift=0.0)          # every rounding change of an element share is flagged (at atol
    x0, T, P = yardstick(4096, 2, 1.0)               # 1e-18: the default's clamping allowance covers rounding drift)
    args = (x0, T, P, full(4096, 500.0), full(4096, 1e-10))
    y, _, _, _, flag = kz.launch(*args, atol=1e-18)
    fb = dict(kz.fallback)
    yk, _, _, _, flag0 = kc.launch(*args, atol=1e-18)                   # the same call, nothing flagged
    i, o = (flag > 0).nonzero().squeeze(1), flag == 0
    yr = tensors(LR.entry("guard/drift", sig, *args, np.array(4000 * KC.REC_STEPS)))["x"][i]
    e = float(floored(y[i][:, gi], yr).max())
    ok = (fb["drift"] > 0 and fb["recomputed"] == fb["drift"] and fb["recovered"] + fb["still_failed"] == fb["drift"]
          and e <= TOL and int(flag0.sum()) == 0 and bool(torch.equal(y[o], yk[o])))
    verdict("guard: element drift (allowance 0)", ok,
            f"counts {fb}; the flagged cells recomputed on the device ({e:.1e} dex from the reference call; still "
            f"failing an allowance of 0 is expected), the others bitwise the kernel's result")

    T2 = T.clone()
    T2[0] = 250.0
    y, _, d, _ = kc(*((x0, T2) + args[2:]))
    try:
        kc.check()
        verdict("guard: T < 300 K raises in check()", False, "no error")
    except ValueError as e:
        verdict("guard: T < 300 K raises in check()", kc.fallback["cold"] == 1 and not bool(d[0]),
                f"cell 0 not stepped; ValueError: {e}")

    # (c) batch size and order
    x0, T, P = yardstick(8192, 7, 1.0)
    a = (x0, T, P, full(8192, 50.0), full(8192, 1e-10))
    big, sel = kc.launch(*a), torch.arange(0, 8192, 64, device=dev)
    perm = torch.randperm(8192, generator=torch.Generator().manual_seed(3)).to(dev)
    shuf = kc.launch(*(q[perm] for q in a))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(8192, device=dev)
    alone = all(torch.equal(p[sel], q) for p, q in zip(big, kc.launch(*(q[sel] for q in a))))
    one = all(torch.equal(p[sel[:1]], q) for p, q in zip(big, kc.launch(*(q[sel[:1]] for q in a))))
    order = all(torch.equal(p, q[inv]) for p, q in zip(big, shuf))
    verdict("batch size and order", alone and one and order,
            f"128 cells alone bitwise inside 8,192 {alone}; one cell alone bitwise {one}; 8,192 cells permuted "
            f"bitwise {order}")

    print(f"{sum(results)} of {len(results)} cases PASS", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
