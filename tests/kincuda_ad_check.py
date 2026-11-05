"""The derivative kernels (maxichem/kincuda_ad.py, ADSolver) on one GPU, on a VULCAN-JAX network with photolysis, the
vm11 law and the collision limit; prints PASS / FAIL per case.
States: truth boxes made with an earlier version of the NCHO network data (paths.AD_BOXES), mapped by species name
(species the network lacks dropped, renormalized; sulfur networks H2S = 1.41e-5 x H2), their T, P and J (branches
matched by absorber and branch; dark boxes J = 0; on SNCHO_photo_network_2025 the branches the boxes lack take the J
of layer i % L of paths.JREF_S, a 1-D WASP-39b model on that network); lm = 0.1 N(0, 1) per reaction unless named
lm = 0.
(a) primal: ADSolver jvp / vjp outputs bitwise CudaSolver(photo=True).launch: settled gas dt 100 s, box starts dt 500 s,
    cold boxes (T < 650 K, where the collision limit acts) dt 100 s.
(b) adaptive tangents against central differences of the primal (CudaSolver's at lm = 0, the derivative kernels' own
    with lm), per input class x, T, P, J (lit cells, relative), lm, rc; one random direction each. Per cell and step
    e in 1e-4..1e-7: err_e = max over species (x > 1e-20) |d ln x_fd - d ln x_ad| / S, S = max |d ln x| of the larger
    of the two; a plateau where the steps agree (same accepted-step count both sides, two consecutive e within 1e-2 of
    S); the best err over the plateau. Judged: cells with S > 1e-6 (from either side). Pass: >= 90 % of them with a
    plateau, p90 <= 1e-4, p99 <= 1e-2.
(b') J in dark cells, along dJ >= 0 (all branches |N(0, 1)|, or one branch per cell), 12 cases (case, one branch or all,
    lm 0 or not); error in x / max |dx|; judged where max |dx / dJ| of either side > 1e-2; three steps, the largest
    scaled per cell so that e max |dx_ad| = 1e-8 and capped, then smaller by ten twice; the plateau and pass rules of
    (b). Near 0, at the default atol: the tangent at J = 1e-8 dJ against central differences (e <= 1e-9); pooled, every
    judged cell counts (>= 90 % with a plateau), and each case with 20 judged cells has a plateau in >= 80 %. At the
    default atol a tangent at exactly J = 0 holds only for rates far below any physical one, too small for differences
    to resolve. At exactly 0, atol 1e-16: one-sided differences, Richardson 2 D(e) - D(2e), e <= 1e-6, with no coverage
    bar (a tiny J often changes the accepted-step count, and such a cell has no plateau).
(c) dot products <c, J v> vs <J^T c, v> per cell, both contracts, per class, c_i = N(0, 1) / x_i on species above
    1e-20 (0 below: trace species near 0 carry rounding of the clamp at 0); scale = the larger sum of absolute terms;
    judged where scale > 1e-10 sum |c x| (counts of the others printed). Pass: p99 <= 1e-6 (lm: 1e-4: a fast balanced
    reaction's multiplier changes nothing in exact arithmetic, its computed sensitivity is rounding of its one-way flux;
    the lm tangent itself is judged in (b)) and max <= 1e-3, no non-finite value.
(d) recovered cells (max_steps 3): primal and flags; dot products as (c) with every class; the rc cotangent of a batch
    of the recovered cells alone; finite derivatives on flags 0 / 1, NaN on flag 2 (state and h).
(e) the rc batch dot product (relative 1e-7: a sum over cells, reactions and columns), its photolysis rows zero;
    several directions in one call bitwise; determinism.
Exit status 1 when any case fails.
    python tests/kincuda_ad_check.py [network stem or file in VULCAN_THERMO, default NCHO_photo]"""

import sys

import numpy as np
import torch

from maxichem import kincuda as KC
from maxichem import kincuda_ad as KA
from maxichem import kinvulcan as KV
from maxichem import paths

paths.need(paths.AD_BOXES)
TH = paths.VULCAN_THERMO
dev = torch.device("cuda")
tt = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)  # noqa: E731
NAME = sys.argv[1] if len(sys.argv) > 1 else "NCHO_photo"
FILE = NAME if NAME.endswith(".txt") else f"{NAME}_network.txt"
net = KV.VulcanNetwork(f"{TH}/{FILE}", f"{TH}/NASA9")
sp = net.species
kc = KC.CudaSolver(net, dev, photo=True)
ad = KA.ADSolver(net, dev, photo=True)
print(f"{NAME}: {len(sp)} species, {ad.nr} rows ({ad.nj} photolysis), rate table width {ad.rcw}", flush=True)
z = np.load(paths.AD_BOXES)
sp0, lines = [str(q) for q in z["species"]], [str(q) for q in z["lines"]]
col = [
    lines.index(f"{q['absorber']},{q['branch']}") if f"{q['absorber']},{q['branch']}" in lines else -1
    for q in net.photo
]
JREF = None  # J of the branches the boxes lack (columns of col < 0)
if FILE == "SNCHO_photo_network_2025.txt":
    paths.need(paths.JREF_S)
    jr = np.load(paths.JREF_S)
    keys = [str(q) for q in jr["lines"]]
    JREF = np.stack([jr["J"][:, keys.index(f"{q['absorber']},{q['branch']}")] for q in net.photo], 1)


def states(sel, k):
    X = z["X"][sel, k]
    x = np.stack([X[:, sp0.index(q)] if q in sp0 else np.zeros(len(sel)) for q in sp], 1)
    if "H2S" in sp:
        x[:, sp.index("H2S")] = 1.41e-5 * x[:, sp.index("H2")]
    x = x / x.sum(1, keepdims=True)
    J = (
        np.stack(
            [
                z["J"][sel, c] if c >= 0 else (JREF[sel % len(JREF), j] if JREF is not None else np.zeros(len(sel)))
                for j, c in enumerate(col)
            ],
            1,
        )
        * z["lit"][sel, None]
    )
    return tt(x), tt(z["T"][sel]), tt(z["P"][sel]), tt(J)


rng = np.random.default_rng(3)
n = len(z["T"])
k4 = int(np.argmin(np.abs(z["stops"] - 1e4)))
cold = np.flatnonzero(z["T"] < 650)
CASES = {
    "settled dt 100": (*states(rng.choice(n, 1024, replace=False), k4), 100.0),
    "starts dt 500": (*states(rng.choice(n, 1024, replace=False), 0), 500.0),
    "cold dt 100": (*states(rng.choice(cold, min(1024, len(cold)), replace=False), k4), 100.0),
}
results = []


def verdict(name, ok, text):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {text}", flush=True)


def pct(a):
    a = np.asarray(a, float)
    if not len(a):
        return "none judged"
    return (
        f"median {np.median(a):.1e} p90 {np.quantile(a, 0.9):.1e} p99 {np.quantile(a, 0.99):.1e} max {a.max():.1e} "
        f"(n {len(a)})"
    )


def zt(x, D=None):
    B = x.shape[0]
    lead = () if D is None else (D,)
    zb = torch.zeros(lead + (B,), dtype=torch.float64, device=dev)
    return torch.zeros(lead + tuple(x.shape), dtype=torch.float64, device=dev), zb, zb.clone(), zb.clone(), zb.clone()


def gen(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return lambda *s: torch.randn(*s, generator=g, dtype=torch.float64, device=dev)


def primal(x, T, P, d, h, J, lm=None, rc=None, atol=KC.ATOL):
    """The call's (x, steps): CudaSolver's at lm = 0 and the network's rc, else the derivative kernels' own primal."""
    if lm is None and rc is None:
        o = kc.launch(x, T, P, d, h, J=J, atol=atol)
    else:
        o = ad.jvp(x, T, P, d, h, *zt(x), lm=lm, rc=rc, J=J, atol=atol)
    return o[0], o[3]


def plateau(fds, ad_t, big, k1, ks):
    """Per cell: the best error over a plateau of the differences (list over steps) against ad_t, and whether a plateau
    exists; S = max |.| of the larger side (cells with S judged by the caller)."""
    B = ad_t.shape[0]
    mx = lambda a: torch.where(big, a.abs(), torch.zeros_like(a)).amax(1)  # noqa: E731
    S = torch.maximum(mx(ad_t), torch.stack([mx(f) for f in fds]).amax(0))
    best = torch.full((B,), np.inf, dtype=torch.float64, device=dev)
    for q in range(len(fds) - 1):
        same = (ks[q] == k1) & (ks[q + 1] == k1)
        agree = mx(fds[q] - fds[q + 1]) <= 1e-2 * S
        for f in (fds[q], fds[q + 1]):
            err = mx(f - ad_t) / S.clamp_min(1e-300)
            best = torch.where(same & agree, torch.minimum(best, err), best)
    return best, S


# ------------------------------------------------------------------------------------------------------------- (a)
for name, (x, T, P, J, dt) in CASES.items():
    h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
    ref = kc.launch(x, T, P, d, h, J=J)[:4]
    j = ad.jvp(x, T, P, d, h, *zt(x), J=J)[:4]
    v = ad.vjp(x, T, P, d, h, torch.zeros_like(x), J=J)[:4]
    same = all(torch.equal(p, q) and torch.equal(p, w) for p, q, w in zip(ref, j, v))
    verdict(
        f"primal {name}",
        same,
        f"{x.shape[0]} cells ({int((J > 0).any(1).sum())} lit): jvp and vjp bitwise CudaSolver's: {same}",
    )


# ------------------------------------------------------------------------------------------------------------- (b)
def fd_case(x, T, P, J, dt, cls, seed, with_lm):
    B = x.shape[0]
    rn = gen(seed)
    h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
    dx, dT, dP, ddt, dh = zt(x)
    lm = 0.1 * rn(B, ad.nr) if with_lm else None
    dlm, dJ, drc = torch.zeros(B, ad.nr, dtype=torch.float64, device=dev), torch.zeros_like(J), None
    lit = (J > 0).any(1)
    if cls == "x":
        dx = x * rn(*x.shape)
    elif cls == "T":
        dT = T * rn(B)
    elif cls == "P":
        dP = P * rn(B)
    elif cls == "lm":
        dlm = rn(B, ad.nr)
    elif cls == "rc":
        drc = rn(ad.nr, 9) * torch.tensor([1.0, 0.1, 30.0, 1.0, 0.1, 30.0, 1.0, 0.1, 30.0], device=dev)
    elif cls == "J":
        dJ = J * rn(*J.shape)
    x1, _, _, k1, tx, _ = ad.jvp(x, T, P, d, h, dx, dT, dP, ddt, dh, lm=lm, dlm=dlm, drc=drc, dJ=dJ, J=J, adapt=True)
    adl = tx / x1
    big = x1 > 1e-20
    fds, ks = [], []
    for e in (1e-4, 1e-5, 1e-6, 1e-7):
        out = []
        for sg in (1.0, -1.0):
            lmp = None if lm is None and cls != "lm" else (lm if lm is not None else 0.0 * dlm) + sg * e * dlm
            rcp = None
            if cls == "rc":
                rcp = ad.rc0.clone()
                rcp[:, :9] += sg * e * drc
            out.append(primal(x + sg * e * dx, T + sg * e * dT, P + sg * e * dP, d, h, J + sg * e * dJ, lmp, rcp))
        (xp, kp), (xm, km) = out
        fds.append((torch.log(xp.clamp_min(1e-300)) - torch.log(xm.clamp_min(1e-300))) / (2 * e))
        ks.append(torch.where(kp == km, kp, torch.full_like(kp, -1.0)))
    best, S = plateau(fds, adl, big, k1, ks)
    judged = S > 1e-6
    if cls == "J":
        judged &= lit
    return best[judged].cpu().numpy(), int(judged.sum())


def fd_verdict(label, b, nj, cover=0.9):
    fin = b[np.isfinite(b)]
    cov = len(fin) / max(nj, 1)
    ok = len(fin) >= 20 and cov >= cover and np.quantile(fin, 0.9) <= 1e-4 and np.quantile(fin, 0.99) <= 1e-2
    verdict(label, ok, f"judged {nj}, with a plateau {len(fin)} ({100 * cov:.0f} %); {pct(fin)}")


for name, (x, T, P, J, dt) in CASES.items():
    for with_lm in (False, True):
        for cls in ("x", "T", "P", "J", "lm", "rc"):
            b, nj = fd_case(x, T, P, J, dt, cls, 11, with_lm)
            fd_verdict(f"FD {name} {cls}{' (lm != 0)' if with_lm else ''}", b, nj)


# ------------------------------------------------------------------------------------------------------------ (b')
def dark_fd(x, T, P, J, dt, single, with_lm, seed, J0, atol):
    B = x.shape[0]
    rn = gen(seed)
    lit = (J > 0).any(1)
    dark = torch.logical_not(lit)
    h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
    lm = 0.1 * rn(B, ad.nr) if with_lm else None
    dJ = rn(*J.shape).abs()
    if single:
        keep = torch.randint(0, ad.nj, (B,), generator=torch.Generator().manual_seed(seed)).to(dev)
        dJ = torch.zeros_like(dJ).scatter_(1, keep[:, None], 1.0)
    dJ = torch.where(dark[:, None], dJ, torch.zeros_like(dJ))
    Jb = J + J0 * dJ
    x0, _, _, k0, tx, _ = ad.jvp(x, T, P, d, h, *zt(x), lm=lm, dJ=dJ, J=Jb, atol=atol)
    mt = tx.abs().amax(1).clamp_min(1e-300)
    big = torch.ones_like(x0, dtype=torch.bool)
    e = (1e-8 / mt).clamp_max(0.1 * J0 if J0 > 0 else 1e-6)[:, None]  # capped once, then smaller by ten twice
    fds, ks = [], []
    for _ in range(3):
        if J0 > 0:
            xa, ka = primal(x, T, P, d, h, Jb + e * dJ, lm, atol=atol)
            xb, kb = primal(x, T, P, d, h, Jb - e * dJ, lm, atol=atol)
            fds.append((xa - xb) / (2 * e))
        else:
            xa, ka = primal(x, T, P, d, h, Jb + e * dJ, lm, atol=atol)
            xb, kb = primal(x, T, P, d, h, Jb + 2 * e * dJ, lm, atol=atol)
            fds.append(2 * (xa - x0) / e - (xb - x0) / (2 * e))
        ks.append(torch.where((ka == kb), ka, torch.full_like(ka, -1.0)))
        e = e / 10
    best, S = plateau(fds, tx, big, k0, ks)
    judged = dark & (S > 1e-2)  # resolvable: J 1e-6 s^-1 moves x by > 1e-8 within dt
    return best[judged].cpu().numpy(), int(judged.sum())


pool, njudged, low = [], 0, []
for name, (x, T, P, J, dt) in CASES.items():
    for single in (False, True):
        for with_lm in (False, True):
            b, nj = dark_fd(x, T, P, J, dt, single, with_lm, 13, 1e-8, KC.ATOL)
            fin = b[np.isfinite(b)]
            pool.append(fin)
            njudged += nj
            lab = f"{name} {'one branch' if single else 'all branches'}{' (lm != 0)' if with_lm else ''}"
            if nj >= 20 and len(fin) < 0.8 * nj:
                low.append(f"{lab}: {len(fin)} of {nj}")
            print(f"INFO  FD J near 0 {lab}: judged {nj}, with a plateau {len(fin)}; {pct(fin)}", flush=True)
fd_verdict("FD J near 0 (J0 1e-8 s^-1), the 12 cases pooled", np.concatenate(pool), njudged)
verdict("FD J near 0, every case with 20 judged cells has a plateau in >= 80 %", not low, "; ".join(low) or "all do")
pool = []
for name, (x, T, P, J, dt) in CASES.items():
    for single in (False, True):
        for with_lm in (False, True):
            b, nj = dark_fd(x, T, P, J, dt, single, with_lm, 13, 0.0, 1e-16)
            fin = b[np.isfinite(b)]
            pool.append(fin)
            print(
                f"INFO  FD J at exactly 0, atol 1e-16, {name} {'one branch' if single else 'all branches'}"
                f"{' (lm != 0)' if with_lm else ''}: judged {nj}, with a plateau {len(fin)}; {pct(fin)}",
                flush=True,
            )
fd_verdict("FD J at exactly 0, atol 1e-16, the 12 cases pooled", np.concatenate(pool), sum(len(q) for q in pool), 0.0)


# ------------------------------------------------------------------------------------------------------------- (c)
def dot_terms(x, T, P, J, dt, cls, adapt, seed, max_steps=4000, sel=None):
    B = x.shape[0]
    rn = gen(seed)
    h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
    dx, dT, dP, ddt, dh = zt(x)
    lm = 0.1 * rn(B, ad.nr)
    dlm, dJ = torch.zeros_like(lm), torch.zeros_like(J)
    if cls == "x":
        dx = x * rn(*x.shape)
    elif cls == "T":
        dT = T * rn(B)
    elif cls == "P":
        dP = P * rn(B)
    elif cls == "dt":
        ddt = d * rn(B)
    elif cls == "h":
        dh = h * rn(B)
    elif cls == "lm":
        dlm = rn(B, ad.nr)
    elif cls == "J":
        dJ = J.clamp_min(1e-8) * rn(*J.shape)
    x1, h1, _, _, tx, th = ad.jvp(
        x, T, P, d, h, dx, dT, dP, ddt, dh, lm=lm, dlm=dlm, dJ=dJ, J=J, adapt=adapt, max_steps=max_steps
    )
    flj = ad.flag.clone()
    c = torch.where(x1 > 1e-20, rn(*x.shape) / x1.clamp_min(1e-300), 0.0)
    ch = rn(B) / h1 if adapt else None
    gg = ad.vjp(x, T, P, d, h, c, gh=ch, lm=lm, J=J, adapt=adapt, max_steps=max_steps)[4]
    assert torch.equal(flj, ad.flag)
    lt = c * tx
    if adapt:
        lt = torch.cat([lt, (ch * th)[:, None]], 1)
    rt = {
        "x": gg["x"] * dx,
        "T": (gg["T"] * dT)[:, None],
        "P": (gg["P"] * dP)[:, None],
        "dt": (gg["dt"] * ddt)[:, None],
        "h": (gg["h"] * dh)[:, None],
        "lm": gg["lm"] * dlm,
        "J": gg["J"] * dJ,
    }[cls]
    sc = torch.maximum(lt.abs().sum(1), rt.abs().sum(1))
    e = (lt.sum(1) - rt.sum(1)).abs() / sc.clamp_min(1e-300)
    judged = sc > 1e-10 * (c * x1).abs().sum(1)
    if sel is not None:
        judged &= sel(flj)
    nonfin = int((torch.logical_not(torch.isfinite(e)) & (flj != 2)).sum())
    return (
        e[judged & torch.isfinite(e)].cpu().numpy(),
        int(judged.sum()),
        int((flj != 2).sum()) - int(judged.sum()),
        nonfin,
        flj,
        tx,
        gg,
        th,
    )


def dot_verdict(label, e, nj, nex, nonfin, p99=1e-6):
    ok = nonfin == 0 and len(e) > 0 and np.quantile(e, 0.99) <= p99 and e.max() <= 1e-3
    verdict(label, ok, f"judged {nj}, not judged (both sides below rounding) {nex}, non-finite {nonfin}; {pct(e)}")


for name, (x, T, P, J, dt) in CASES.items():
    for adapt in (False, True):
        for cls in ("x", "T", "P", "dt", "h", "lm", "J"):
            if cls in ("dt", "h") and not adapt:
                continue
            e, nj, nex, nonfin = dot_terms(x, T, P, J, dt, cls, adapt, 21)[:4]
            dot_verdict(
                f"dot {name} {'adaptive' if adapt else 'frozen'} {cls}",
                e,
                nj,
                nex,
                nonfin,
                1e-4 if cls == "lm" else 1e-6,
            )


# ------------------------------------------------------------------------------------------------------------- (d)
x, T, P, J, dt = CASES["starts dt 500"]
B = x.shape[0]
h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
ref = kc.launch(x, T, P, d, h, J=J, max_steps=3)
j = ad.jvp(x, T, P, d, h, *zt(x), J=J, max_steps=3)
fl_j = ad.flag.clone()
v = ad.vjp(x, T, P, d, h, torch.zeros_like(x), J=J, max_steps=3)
same = all(torch.equal(p, q) and torch.equal(p, w) for p, q, w in zip(ref[:4], j[:4], v[:4]))
verdict(
    "recovered cells: primal and flags",
    same and torch.equal(ref[4], fl_j) and torch.equal(ref[4], ad.flag),
    f"max_steps 3: flags 0 / 1 / 2 = {[int((ref[4] == k).sum()) for k in range(3)]}, bitwise CudaSolver's: {same}",
)
for adapt in (False, True):
    for cls in ("x", "T", "P", "dt", "h", "lm", "J"):
        if cls in ("dt", "h") and not adapt:
            continue
        e, nj, nex, nonfin, flj, tx, gg, th = dot_terms(x, T, P, J, dt, cls, adapt, 31, 3, lambda f: f == 1)
        bad2 = flj == 2
        nan2 = bool(
            torch.isnan(tx[bad2]).all()
            and torch.isnan(gg["x"][bad2]).all()
            and torch.isnan(gg["T"][bad2]).all()
            and (not adapt or torch.isnan(th[bad2]).all())
        )
        fin = bool(
            torch.isfinite(tx[torch.logical_not(bad2)]).all() and torch.isfinite(gg["x"][torch.logical_not(bad2)]).all()
        )
        dot_verdict(
            f"recovered cells dot {'adaptive' if adapt else 'frozen'} {cls} (finite on flags 0 / 1 {fin}, NaN on "
            f"flag 2 {nan2})",
            e if fin and nan2 else np.array([np.inf]),
            nj,
            nex,
            nonfin,
        )
keep = torch.nonzero(ref[4] == 1).flatten()  # the recovered cells as their own batch: rc batch dot
xr, Tr, Pr, Jr = x[keep], T[keep], P[keep], J[keep]
for adapt in (False, True):
    rn = gen(43)
    hr, dr = torch.full_like(Tr, 1e-10), torch.full_like(Tr, dt)
    drc = rn(ad.nr, 9)
    x1, _, _, _, tx, _ = ad.jvp(xr, Tr, Pr, dr, hr, *zt(xr), drc=drc, J=Jr, adapt=adapt, max_steps=3)
    allrec = bool((ad.flag == 1).all())
    c = torch.where(x1 > 1e-20, rn(*xr.shape) / x1.clamp_min(1e-300), 0.0)
    gg = ad.vjp(xr, Tr, Pr, dr, hr, c, J=Jr, adapt=adapt, max_steps=3)[4]
    lhs, rhs = float((c * tx).sum()), float((gg["rc"] * drc).sum())
    e = abs(lhs - rhs) / max(float((c * tx).abs().sum()), float((gg["rc"] * drc).abs().sum()))
    verdict(
        f"recovered cells rc (batch) {'adaptive' if adapt else 'frozen'}",
        allrec and e <= 1e-7,
        f"{len(keep)} cells, all recovered again {allrec}; relative {e:.1e}",
    )

# ------------------------------------------------------------------------------------------------------------- (e)
for name, (x, T, P, J, dt) in CASES.items():
    for adapt in (False, True):
        rn = gen(41)
        h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
        drc = rn(ad.nr, 9)
        x1, _, _, _, tx, _ = ad.jvp(x, T, P, d, h, *zt(x), drc=drc, J=J, adapt=adapt)
        c = torch.where(x1 > 1e-20, rn(*x.shape) / x1.clamp_min(1e-300), 0.0)
        gg = ad.vjp(x, T, P, d, h, c, J=J, adapt=adapt)[4]
        lhs, rhs = float((c * tx).sum()), float((gg["rc"] * drc).sum())
        e = abs(lhs - rhs) / max(float((c * tx).abs().sum()), float((gg["rc"] * drc).abs().sum()))
        zero = bool((gg["rc"][ad.nr - ad.nj :] == 0).all())
        verdict(
            f"dot rc (batch) {name} {'adaptive' if adapt else 'frozen'}",
            e <= 1e-7 and zero,
            f"relative {e:.1e}; photolysis rows of the rc cotangent zero: {zero}",
        )
x, T, P, J, dt = CASES["settled dt 100"]
B = x.shape[0]
h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
rn = gen(51)
D = 3
tan = (x * rn(D, *x.shape), T * rn(D, B), P * rn(D, B), d * rn(D, B), h * rn(D, B))
dlm, dJ = rn(D, B, ad.nr), J.clamp_min(1e-8) * rn(D, *J.shape)
lm = 0.1 * rn(B, ad.nr)
allx = ad.jvp(x, T, P, d, h, *tan, lm=lm, dlm=dlm, dJ=dJ, J=J)
one = [ad.jvp(x, T, P, d, h, *(a[k] for a in tan), lm=lm, dlm=dlm[k], dJ=dJ[k], J=J) for k in range(D)]
same = all(torch.equal(allx[4][k], one[k][4]) and torch.equal(allx[5][k], one[k][5]) for k in range(D))
verdict("three directions in one call", same, f"bitwise equal to one call per direction: {same}")
c = rn(*x.shape)
a1 = ad.vjp(x, T, P, d, h, c, lm=lm, J=J)[4]
a2 = ad.vjp(x, T, P, d, h, c, lm=lm, J=J)[4]
perm = torch.randperm(B, generator=torch.Generator().manual_seed(1)).to(dev)
a3 = ad.vjp(x[perm], T[perm], P[perm], d[perm], h[perm], c[perm], lm=lm[perm], J=J[perm])[4]
det = all(torch.equal(a1[k], a2[k]) for k in a1)
pm = all(torch.equal(a1[k][perm], a3[k]) for k in a1 if k != "rc")
verdict("determinism", det and pm, f"same call twice bitwise: {det}; cells permuted, per-cell cotangents bitwise: {pm}")

print(f"\n{NAME}: {sum(results)} of {len(results)} passed", flush=True)
sys.exit(0 if all(results) else 1)
