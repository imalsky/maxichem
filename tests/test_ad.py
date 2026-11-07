"""The derivative kernels against finite differences of the call and against each other.

States: truth boxes made with an earlier version of the NCHO network data (paths.ad_boxes()), mapped by species name,
with their T, P and J. On SNCHO_photo_network_2025 the branches the boxes lack take the J of a 1-D WASP-39b model. The
cases are in docs/testing.md. Run with ``pytest tests/test_ad.py -s`` to see each case's numbers.
"""

import types
from typing import Any

import numpy as np
import pytest
import torch

from maxichem import _checks, kincuda, kincuda_ad, kinvulcan, paths

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# Tangents against central differences, per cell: the difference has its own truncation and rounding, so a limit bounds
# the best error over a plateau of step sizes. Measured on all three networks: p90 at most 2.6e-5, p99 at most 2.4e-3,
# worst 0.77.
FD_COVERAGE = 0.9  # the share of judged cells that must have a plateau; measured at least 0.94
FD_MIN_CELLS = 20
FD_P90 = 1e-4
FD_P99 = 1e-2
FD_MAX = 1.0  # the worst cell: a plateau can sit near a branch of the step count, so this limit is loose
FD_CASE_COVERAGE = 0.8  # a case with enough judged cells, for J near 0
# <c, J v> against <J^T c, v>: both sides are this code, so only rounding separates them. Measured: p99 at most 1.5e-7,
# worst 9.8e-5.
DOT_P99 = 1e-6
DOT_P99_LM = 1e-4  # a fast balanced reaction's multiplier changes nothing in exact arithmetic; this is its rounding
DOT_MAX = 1e-3
RC_BATCH = 1e-7  # relative: a sum over cells, reactions and columns


CELLS = 1024
COLD_LIMIT = 650.0  # K: the collision limit acts below this


def pct(a):
    """Format the quantiles of an error array."""
    a = np.asarray(a, float)
    if not len(a):
        return "none judged"
    return (
        f"median {np.median(a):.1e} p90 {np.quantile(a, 0.9):.1e} p99 {np.quantile(a, 0.99):.1e} max {a.max():.1e} "
        f"(n {len(a)})"
    )


def fd_verdict(verdict, label, b, nj, nonfin, flag2, cover=FD_COVERAGE):
    """Judge a case's finite-difference errors, and its counts of non-finite cells and of cells the guard flagged 2."""
    fin = b[np.isfinite(b)]
    cov = len(fin) / max(nj, 1)
    ok = (
        nonfin == 0
        and flag2 == 0
        and len(fin) >= FD_MIN_CELLS
        and cov >= cover
        and np.quantile(fin, 0.9) <= FD_P90
        and np.quantile(fin, 0.99) <= FD_P99
        and fin.max() <= FD_MAX
    )
    verdict(
        label,
        ok,
        f"judged {nj}, with a plateau {len(fin)} ({100 * cov:.0f} %), non-finite {nonfin}, flag 2 {flag2}; {pct(fin)}",
    )


def dot_verdict(verdict, label, e, nj, nex, nonfin, p99=DOT_P99):
    """Judge a case's dot-product errors."""
    ok = nonfin == 0 and len(e) > 0 and np.quantile(e, 0.99) <= p99 and e.max() <= DOT_MAX
    verdict(label, ok, f"judged {nj}, not judged (both sides below rounding) {nex}, non-finite {nonfin}; {pct(e)}")


class DerivativeChecks:
    """A network with its solvers, states and the comparisons the tests run."""

    def __init__(self, name, device):
        thermo = paths.vulcan_thermo()
        paths.need(paths.ad_boxes())
        file = name if name.endswith(".txt") else f"{name}_network.txt"
        self.device = device
        self.net = kinvulcan.VulcanNetwork(thermo / file, thermo / "NASA9")
        self.kc = kincuda.CudaSolver(self.net, device, photo=True)
        self.ad = kincuda_ad.ADSolver(self.net, device, photo=True)
        self.boxes = np.load(paths.ad_boxes())
        self.box_species = [str(q) for q in self.boxes["species"]]
        lines = [str(q) for q in self.boxes["lines"]]
        self.columns = [
            lines.index(f"{q['absorber']},{q['branch']}") if f"{q['absorber']},{q['branch']}" in lines else -1
            for q in self.net.photo
        ]
        self.reference_J = None  # J of the branches the boxes lack
        if file == "SNCHO_photo_network_2025.txt":
            paths.need(paths.jref_sulfur())
            stored = np.load(paths.jref_sulfur())
            keys = [str(q) for q in stored["lines"]]
            self.reference_J = np.stack(
                [stored["J"][:, keys.index(f"{q['absorber']},{q['branch']}")] for q in self.net.photo], 1
            )
        rng = np.random.default_rng(3)
        n = len(self.boxes["T"])
        settled = int(np.argmin(np.abs(self.boxes["stops"] - 1e4)))
        cold = np.flatnonzero(self.boxes["T"] < COLD_LIMIT)
        self.cases = {
            "settled dt 100": (*self.states(rng.choice(n, CELLS, replace=False), settled), 100.0),
            "starts dt 500": (*self.states(rng.choice(n, CELLS, replace=False), 0), 500.0),
            "cold dt 100": (*self.states(rng.choice(cold, min(CELLS, len(cold)), replace=False), settled), 100.0),
        }

    def close(self):
        """Unload the kernels."""
        self.kc.close()
        self.ad.close()

    def states(self, sel, k):
        """Return the boxes' states at stop k as tensors (x, T, P, J), mapped onto the network."""
        z, species = self.boxes, self.net.species
        box = z["X"][sel, k]
        x = np.stack(
            [box[:, self.box_species.index(q)] if q in self.box_species else np.zeros(len(sel)) for q in species], 1
        )
        if "H2S" in species:
            x[:, species.index("H2S")] = _checks.H2S_PER_H2_SOLAR * x[:, species.index("H2")]
        x = x / x.sum(1, keepdims=True)
        rows = []
        for j, c in enumerate(self.columns):
            if c >= 0:
                rows.append(z["J"][sel, c])
            elif self.reference_J is not None:
                rows.append(self.reference_J[sel % len(self.reference_J), j])
            else:
                rows.append(np.zeros(len(sel)))
        J = np.stack(rows, 1) * z["lit"][sel, None]
        return tuple(self.tensor(a) for a in (x, z["T"][sel], z["P"][sel], J))

    def tensor(self, a):
        """Return an array as a float64 tensor on the GPU."""
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=self.device)

    def zt(self, x, D=None):
        """Return zero tangents (dx, dT, dP, ddt, dh) for the state x, with an optional leading axis of D directions."""
        B = x.shape[0]
        lead = () if D is None else (D,)
        zb = torch.zeros((*lead, B), dtype=torch.float64, device=self.device)
        return (
            torch.zeros(lead + tuple(x.shape), dtype=torch.float64, device=self.device),
            zb,
            zb.clone(),
            zb.clone(),
            zb.clone(),
        )

    def gen(self, seed):
        """Return a function that draws float64 normal tensors from a generator seeded with seed."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        return lambda *s: torch.randn(*s, generator=g, dtype=torch.float64, device=self.device)

    def primal(self, x, T, P, d, h, J, lm=None, rc=None, atol=kincuda.ATOL):
        """Return the call's (x, steps): CudaSolver's at lm = 0 and the network's rc, else the derivative kernels'."""
        if lm is None and rc is None:
            o = self.kc.launch(x, T, P, d, h, J=J, atol=atol)
        else:
            o = self.ad.jvp(x, T, P, d, h, *self.zt(x), lm=lm, rc=rc, J=J, atol=atol)
        return o[0], o[3]

    def last_flag(self):
        """Return a copy of the last call's per-cell flag."""
        assert self.ad.flag is not None
        return self.ad.flag.clone()

    @staticmethod
    def nonfinite_cells(flag, tx, th, S):
        """Count the cells with a non-finite tangent or difference (S holds both); only flag 2 may have them."""
        finite = torch.isfinite(tx).all(1) & torch.isfinite(th) & torch.isfinite(S)
        return int(((flag != 2) & torch.logical_not(finite)).sum())

    def plateau(self, fds, ad_t, big, k1, ks):
        """Return per cell the best error over a plateau of the differences against ad_t, and its scale S.

        S = max |.| of the larger side (cells with S judged by the caller); no plateau leaves the error infinite.
        """
        B = ad_t.shape[0]

        def mx(a):
            return torch.where(big, a.abs(), torch.zeros_like(a)).amax(1)

        S = torch.maximum(mx(ad_t), torch.stack([mx(f) for f in fds]).amax(0))
        best = torch.full((B,), np.inf, dtype=torch.float64, device=self.device)
        for q in range(len(fds) - 1):
            same = (ks[q] == k1) & (ks[q + 1] == k1)
            agree = mx(fds[q] - fds[q + 1]) <= 1e-2 * S
            for f in (fds[q], fds[q + 1]):
                err = mx(f - ad_t) / S.clamp_min(1e-300)
                best = torch.where(same & agree, torch.minimum(best, err), best)
        return best, S

    def fd_case(self, x, T, P, J, dt, cls, seed, with_lm):
        """Compare one input class's tangent with central differences: judged errors, count, non-finite, flag 2."""
        B = x.shape[0]
        rn = self.gen(seed)
        h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
        dx, dT, dP, ddt, dh = self.zt(x)
        lm = 0.1 * rn(B, self.ad.nr) if with_lm else None
        dlm, dJ = torch.zeros(B, self.ad.nr, dtype=torch.float64, device=self.device), torch.zeros_like(J)
        drc: Any = None
        lit = (J > 0).any(1)
        if cls == "x":
            dx = x * rn(*x.shape)
        elif cls == "T":
            dT = T * rn(B)
        elif cls == "P":
            dP = P * rn(B)
        elif cls == "lm":
            dlm = rn(B, self.ad.nr)
        elif cls == "rc":
            drc = rn(self.ad.nr, 9) * torch.tensor([1.0, 0.1, 30.0, 1.0, 0.1, 30.0, 1.0, 0.1, 30.0], device=self.device)
        elif cls == "J":
            dJ = J * rn(*J.shape)
        x1, _, _, k1, tx, th = self.ad.jvp(
            x, T, P, d, h, dx, dT, dP, ddt, dh, lm=lm, dlm=dlm, drc=drc, dJ=dJ, J=J, adapt=True
        )
        flag = self.last_flag()  # the differences' own calls replace it
        adl = tx / x1
        big = x1 > 1e-20
        fds, ks = [], []
        for e in (1e-4, 1e-5, 1e-6, 1e-7):
            out = []
            for sg in (1.0, -1.0):
                lmp = None if lm is None and cls != "lm" else (lm if lm is not None else 0.0 * dlm) + sg * e * dlm
                rcp = None
                if cls == "rc":
                    rcp = self.ad.rc0.clone()
                    rcp[:, :9] += sg * e * drc
                out.append(
                    self.primal(x + sg * e * dx, T + sg * e * dT, P + sg * e * dP, d, h, J + sg * e * dJ, lmp, rcp)
                )
            (xp, kp), (xm, km) = out
            fds.append((torch.log(xp.clamp_min(1e-300)) - torch.log(xm.clamp_min(1e-300))) / (2 * e))
            ks.append(torch.where(kp == km, kp, torch.full_like(kp, -1.0)))
        best, S = self.plateau(fds, adl, big, k1, ks)
        judged = S > 1e-6
        if cls == "J":
            judged &= lit
        return (
            best[judged].cpu().numpy(),
            int(judged.sum()),
            self.nonfinite_cells(flag, tx, th, S),
            int((flag == 2).sum()),
        )

    def dark_fd(self, x, T, P, J, dt, single, with_lm, seed, j0, atol):
        """Compare the J tangent in dark cells with differences: judged errors, their count, non-finite, flag 2."""
        B = x.shape[0]
        rn = self.gen(seed)
        lit = (J > 0).any(1)
        dark = torch.logical_not(lit)
        h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
        lm = 0.1 * rn(B, self.ad.nr) if with_lm else None
        dJ = rn(*J.shape).abs()
        if single:
            keep = torch.randint(0, self.ad.nj, (B,), generator=torch.Generator().manual_seed(seed)).to(self.device)
            dJ = torch.zeros_like(dJ).scatter_(1, keep[:, None], 1.0)
        dJ = torch.where(dark[:, None], dJ, torch.zeros_like(dJ))
        j_base = J + j0 * dJ
        x0, _, _, k0, tx, th = self.ad.jvp(x, T, P, d, h, *self.zt(x), lm=lm, dJ=dJ, J=j_base, atol=atol)
        flag = self.last_flag()  # the differences' own calls replace it
        mt = tx.abs().amax(1).clamp_min(1e-300)
        big = torch.ones_like(x0, dtype=torch.bool)
        e = (1e-8 / mt).clamp_max(0.1 * j0 if j0 > 0 else 1e-6)[:, None]  # capped once, then smaller by ten twice
        fds, ks = [], []
        for _ in range(3):
            if j0 > 0:
                xa, ka = self.primal(x, T, P, d, h, j_base + e * dJ, lm, atol=atol)
                xb, kb = self.primal(x, T, P, d, h, j_base - e * dJ, lm, atol=atol)
                fds.append((xa - xb) / (2 * e))
            else:
                xa, ka = self.primal(x, T, P, d, h, j_base + e * dJ, lm, atol=atol)
                xb, kb = self.primal(x, T, P, d, h, j_base + 2 * e * dJ, lm, atol=atol)
                fds.append(2 * (xa - x0) / e - (xb - x0) / (2 * e))
            ks.append(torch.where((ka == kb), ka, torch.full_like(ka, -1.0)))
            e = e / 10
        best, S = self.plateau(fds, tx, big, k0, ks)
        judged = dark & (S > 1e-2)  # resolvable: J 1e-6 s^-1 moves x by > 1e-8 within dt
        return (
            best[judged].cpu().numpy(),
            int(judged.sum()),
            self.nonfinite_cells(flag, tx, th, S),
            int((flag == 2).sum()),
        )

    def dot_terms(self, x, T, P, J, dt, cls, adapt, seed, max_steps=4000, sel=None):
        """Return the dot-product errors of one input class, with the call's flags, tangents and cotangents."""
        B = x.shape[0]
        rn = self.gen(seed)
        h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
        dx, dT, dP, ddt, dh = self.zt(x)
        lm = 0.1 * rn(B, self.ad.nr)
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
            dlm = rn(B, self.ad.nr)
        elif cls == "J":
            dJ = J.clamp_min(1e-8) * rn(*J.shape)
        x1, h1, _, _, tx, th = self.ad.jvp(
            x, T, P, d, h, dx, dT, dP, ddt, dh, lm=lm, dlm=dlm, dJ=dJ, J=J, adapt=adapt, max_steps=max_steps
        )
        flj = self.last_flag()
        c = torch.where(x1 > 1e-20, rn(*x.shape) / x1.clamp_min(1e-300), 0.0)
        ch = rn(B) / h1 if adapt else None
        gg = self.ad.vjp(x, T, P, d, h, c, gh=ch, lm=lm, J=J, adapt=adapt, max_steps=max_steps)[4]
        assert torch.equal(flj, self.ad.flag)
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
        return types.SimpleNamespace(
            e=e[judged & torch.isfinite(e)].cpu().numpy(),
            judged=int(judged.sum()),
            not_judged=int((flj != 2).sum()) - int(judged.sum()),
            nonfinite=nonfin,
            flag=flj,
            tx=tx,
            cotangents=gg,
            th=th,
        )


@pytest.fixture(scope="session", params=["NCHO_photo", "SNCHO_photo_network_2025.txt", "CHO_photo"])
def checks(request, gpu):
    """Return a network with its solvers and states."""
    case = DerivativeChecks(request.param, gpu)
    yield case
    case.close()


def test_primal_is_the_call(checks, verdict):
    for name, (x, T, P, J, dt) in checks.cases.items():
        h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
        ref = checks.kc.launch(x, T, P, d, h, J=J)[:4]
        j = checks.ad.jvp(x, T, P, d, h, *checks.zt(x), J=J)[:4]
        v = checks.ad.vjp(x, T, P, d, h, torch.zeros_like(x), J=J)[:4]
        same = all(torch.equal(p, q) and torch.equal(p, w) for p, q, w in zip(ref, j, v, strict=True))
        verdict(
            f"primal {name}",
            same,
            f"{x.shape[0]} cells ({int((J > 0).any(1).sum())} lit): jvp and vjp bitwise CudaSolver's: {same}",
        )


def test_tangents_against_differences(checks, verdict):
    for name, (x, T, P, J, dt) in checks.cases.items():
        for with_lm in (False, True):
            for cls in ("x", "T", "P", "J", "lm", "rc"):
                b, nj, nonfin, flag2 = checks.fd_case(x, T, P, J, dt, cls, 11, with_lm)
                fd_verdict(verdict, f"FD {name} {cls}{' (lm != 0)' if with_lm else ''}", b, nj, nonfin, flag2)


def test_dark_tangents_near_zero(checks, verdict):
    pool, njudged, nonfinite, flagged, low = [], 0, 0, 0, []
    for name, (x, T, P, J, dt) in checks.cases.items():
        for single in (False, True):
            for with_lm in (False, True):
                b, nj, nonfin, flag2 = checks.dark_fd(x, T, P, J, dt, single, with_lm, 13, 1e-8, kincuda.ATOL)
                fin = b[np.isfinite(b)]
                pool.append(fin)
                njudged += nj
                nonfinite += nonfin
                flagged += flag2
                lab = f"{name} {'one branch' if single else 'all branches'}{' (lm != 0)' if with_lm else ''}"
                if nj >= FD_MIN_CELLS and len(fin) < FD_CASE_COVERAGE * nj:
                    low.append(f"{lab}: {len(fin)} of {nj}")
                print(f"INFO  FD J near 0 {lab}: judged {nj}, with a plateau {len(fin)}; {pct(fin)}", flush=True)
    fd_verdict(
        verdict, "FD J near 0 (J0 1e-8 s^-1), the 12 cases pooled", np.concatenate(pool), njudged, nonfinite, flagged
    )
    verdict(
        "FD J near 0, every case with 20 judged cells has a plateau in >= 80 %", not low, "; ".join(low) or "all do"
    )


def test_dark_tangents_at_zero(checks, verdict):
    pool, judged_at_zero, nonfinite, flagged = [], 0, 0, 0
    for name, (x, T, P, J, dt) in checks.cases.items():
        for single in (False, True):
            for with_lm in (False, True):
                b, nj, nonfin, flag2 = checks.dark_fd(x, T, P, J, dt, single, with_lm, 13, 0.0, 1e-16)
                fin = b[np.isfinite(b)]
                pool.append(fin)
                judged_at_zero += nj
                nonfinite += nonfin
                flagged += flag2
                print(
                    f"INFO  FD J at exactly 0, atol 1e-16, {name} {'one branch' if single else 'all branches'}"
                    f"{' (lm != 0)' if with_lm else ''}: judged {nj}, with a plateau {len(fin)}; {pct(fin)}",
                    flush=True,
                )
    fd_verdict(
        verdict,
        "FD J at exactly 0, atol 1e-16, the 12 cases pooled",
        np.concatenate(pool),
        judged_at_zero,
        nonfinite,
        flagged,
        0.0,
    )


def test_dot_products(checks, verdict):
    for name, (x, T, P, J, dt) in checks.cases.items():
        for adapt in (False, True):
            for cls in ("x", "T", "P", "dt", "h", "lm", "J"):
                if cls in ("dt", "h") and not adapt:
                    continue
                t = checks.dot_terms(x, T, P, J, dt, cls, adapt, 21)
                dot_verdict(
                    verdict,
                    f"dot {name} {'adaptive' if adapt else 'frozen'} {cls}",
                    t.e,
                    t.judged,
                    t.not_judged,
                    t.nonfinite,
                    DOT_P99_LM if cls == "lm" else DOT_P99,
                )


def test_recovered_primal_and_flags(checks, verdict):
    x, T, P, J, dt = checks.cases["starts dt 500"]
    h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
    ref = checks.kc.launch(x, T, P, d, h, J=J, max_steps=3)
    j = checks.ad.jvp(x, T, P, d, h, *checks.zt(x), J=J, max_steps=3)
    fl_j = checks.ad.flag.clone()
    v = checks.ad.vjp(x, T, P, d, h, torch.zeros_like(x), J=J, max_steps=3)
    same = all(torch.equal(p, q) and torch.equal(p, w) for p, q, w in zip(ref[:4], j[:4], v[:4], strict=True))
    verdict(
        "recovered cells: primal and flags",
        same and torch.equal(ref[4], fl_j) and torch.equal(ref[4], checks.ad.flag),
        f"max_steps 3: flags 0 / 1 / 2 = {[int((ref[4] == k).sum()) for k in range(3)]}, bitwise CudaSolver's: {same}",
    )


def test_recovered_dot_products(checks, verdict):
    x, T, P, J, dt = checks.cases["starts dt 500"]
    for adapt in (False, True):
        for cls in ("x", "T", "P", "dt", "h", "lm", "J"):
            if cls in ("dt", "h") and not adapt:
                continue
            t = checks.dot_terms(x, T, P, J, dt, cls, adapt, 31, 3, lambda f: f == 1)
            bad2 = t.flag == 2
            nan2 = bool(
                torch.isnan(t.tx[bad2]).all()
                and torch.isnan(t.cotangents["x"][bad2]).all()
                and torch.isnan(t.cotangents["T"][bad2]).all()
                and (not adapt or torch.isnan(t.th[bad2]).all())
            )
            fin = bool(
                torch.isfinite(t.tx[torch.logical_not(bad2)]).all()
                and torch.isfinite(t.cotangents["x"][torch.logical_not(bad2)]).all()
            )
            dot_verdict(
                verdict,
                f"recovered cells dot {'adaptive' if adapt else 'frozen'} {cls} (finite on flags 0 / 1 {fin}, NaN on "
                f"flag 2 {nan2})",
                t.e if fin and nan2 else np.array([np.inf]),
                t.judged,
                t.not_judged,
                t.nonfinite,
            )


def test_recovered_rate_table(checks, verdict):
    x, T, P, J, dt = checks.cases["starts dt 500"]
    h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
    ref = checks.kc.launch(x, T, P, d, h, J=J, max_steps=3)
    keep = torch.nonzero(ref[4] == 1).flatten()  # the recovered cells as their own batch: rc batch dot
    xr, t_r, p_r, j_r = x[keep], T[keep], P[keep], J[keep]
    for adapt in (False, True):
        rn = checks.gen(43)
        hr, dr = torch.full_like(t_r, 1e-10), torch.full_like(t_r, dt)
        drc = rn(checks.ad.nr, 9)
        x1, _, _, _, tx, _ = checks.ad.jvp(
            xr, t_r, p_r, dr, hr, *checks.zt(xr), drc=drc, J=j_r, adapt=adapt, max_steps=3
        )
        allrec = bool((checks.ad.flag == 1).all())
        c = torch.where(x1 > 1e-20, rn(*xr.shape) / x1.clamp_min(1e-300), 0.0)
        gg = checks.ad.vjp(xr, t_r, p_r, dr, hr, c, J=j_r, adapt=adapt, max_steps=3)[4]
        lhs, rhs = float((c * tx).sum()), float((gg["rc"] * drc).sum())
        e = abs(lhs - rhs) / max(float((c * tx).abs().sum()), float((gg["rc"] * drc).abs().sum()))
        verdict(
            f"recovered cells rc (batch) {'adaptive' if adapt else 'frozen'}",
            allrec and e <= RC_BATCH,
            f"{len(keep)} cells, all recovered again {allrec}; relative {e:.1e}",
        )


def test_rate_table_dot_products(checks, verdict):
    for name, (x, T, P, J, dt) in checks.cases.items():
        for adapt in (False, True):
            rn = checks.gen(41)
            h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
            drc = rn(checks.ad.nr, 9)
            x1, _, _, _, tx, _ = checks.ad.jvp(x, T, P, d, h, *checks.zt(x), drc=drc, J=J, adapt=adapt)
            c = torch.where(x1 > 1e-20, rn(*x.shape) / x1.clamp_min(1e-300), 0.0)
            gg = checks.ad.vjp(x, T, P, d, h, c, J=J, adapt=adapt)[4]
            lhs, rhs = float((c * tx).sum()), float((gg["rc"] * drc).sum())
            e = abs(lhs - rhs) / max(float((c * tx).abs().sum()), float((gg["rc"] * drc).abs().sum()))
            zero = bool((gg["rc"][checks.ad.nr - checks.ad.nj :] == 0).all())
            verdict(
                f"dot rc (batch) {name} {'adaptive' if adapt else 'frozen'}",
                e <= RC_BATCH and zero,
                f"relative {e:.1e}; photolysis rows of the rc cotangent zero: {zero}",
            )


def _directions(checks):
    """Return a settled case with three random tangent directions and a cotangent."""
    x, T, P, J, dt = checks.cases["settled dt 100"]
    B = x.shape[0]
    h, d = torch.full_like(T, 1e-10), torch.full_like(T, dt)
    rn = checks.gen(51)
    D = 3
    tan = (x * rn(D, *x.shape), T * rn(D, B), P * rn(D, B), d * rn(D, B), h * rn(D, B))
    dlm, dJ = rn(D, B, checks.ad.nr), J.clamp_min(1e-8) * rn(D, *J.shape)
    lm = 0.1 * rn(B, checks.ad.nr)
    c = rn(*x.shape)
    tangents_h = (*tan[:4], d * rn(D, B))
    return types.SimpleNamespace(
        x=x, T=T, P=P, J=J, d=d, h=h, tan=tan, tangents_with_h=tangents_h, dlm=dlm, dJ=dJ, lm=lm, c=c, D=D, B=B
    )


def test_directions_share_one_call(checks, verdict):
    s = _directions(checks)
    allx = checks.ad.jvp(s.x, s.T, s.P, s.d, s.h, *s.tan, lm=s.lm, dlm=s.dlm, dJ=s.dJ, J=s.J)
    one = [
        checks.ad.jvp(s.x, s.T, s.P, s.d, s.h, *(a[k] for a in s.tan), lm=s.lm, dlm=s.dlm[k], dJ=s.dJ[k], J=s.J)
        for k in range(s.D)
    ]
    same = all(torch.equal(allx[4][k], one[k][4]) and torch.equal(allx[5][k], one[k][5]) for k in range(s.D))
    verdict("three directions in one call", same, f"bitwise equal to one call per direction: {same}")


def test_determinism(checks, verdict):
    s = _directions(checks)
    a1 = checks.ad.vjp(s.x, s.T, s.P, s.d, s.h, s.c, lm=s.lm, J=s.J)[4]
    a2 = checks.ad.vjp(s.x, s.T, s.P, s.d, s.h, s.c, lm=s.lm, J=s.J)[4]
    perm = torch.randperm(s.B, generator=torch.Generator().manual_seed(1)).to(checks.device)
    a3 = checks.ad.vjp(s.x[perm], s.T[perm], s.P[perm], s.d[perm], s.h[perm], s.c[perm], lm=s.lm[perm], J=s.J[perm])[4]
    det = all(torch.equal(a1[k], a2[k]) for k in a1)
    pm = all(torch.equal(a1[k][perm], a3[k]) for k in a1 if k != "rc")
    verdict(
        "determinism", det and pm, f"same call twice bitwise: {det}; cells permuted, per-cell cotangents bitwise: {pm}"
    )


def test_non_finite_step_acts_as_twice_dt(checks, verdict):
    # A non-finite starting step acts as dt with no sensitivity to it: min(h, dt) = dt at h = 2 dt too.
    s = _directions(checks)
    j2 = checks.ad.jvp(s.x, s.T, s.P, s.d, 2.0 * s.d, *s.tangents_with_h, lm=s.lm, dlm=s.dlm, dJ=s.dJ, J=s.J)
    v2 = checks.ad.vjp(s.x, s.T, s.P, s.d, 2.0 * s.d, s.c, lm=s.lm, J=s.J)
    nonfin = {}
    for q in (float("nan"), float("-inf")):
        jn = checks.ad.jvp(
            s.x, s.T, s.P, s.d, torch.full_like(s.T, q), *s.tangents_with_h, lm=s.lm, dlm=s.dlm, dJ=s.dJ, J=s.J
        )
        vn = checks.ad.vjp(s.x, s.T, s.P, s.d, torch.full_like(s.T, q), s.c, lm=s.lm, J=s.J)
        nonfin[str(q)] = (
            all(torch.equal(a, b) for a, b in zip(jn, j2, strict=True))
            and all(torch.equal(a, b) for a, b in zip(vn[:4], v2[:4], strict=True))
            and all(torch.equal(vn[4][k], v2[4][k]) for k in vn[4])
            and bool((vn[4]["h"] == 0).all())
            and checks.ad.fallback["nonfinite_h"] == s.x.shape[0]
        )
    verdict(
        "non-finite starting step",
        all(nonfin.values()),
        f"h = NaN / -inf bitwise the h = 2 dt call (tangents, cotangents; h cotangent zero): {nonfin}",
    )


def test_nan_log_multiplier_takes_no_tries(checks, verdict):
    s = _directions(checks)
    lm_bad = s.lm.clone()
    lm_bad[0, 0] = float("nan")
    jb = checks.ad.jvp(s.x, s.T, s.P, s.d, 2.0 * s.d, *s.tangents_with_h, lm=lm_bad, dlm=s.dlm, dJ=s.dJ, J=s.J)
    cj = checks.ad.fallback
    vb = checks.ad.vjp(s.x, s.T, s.P, s.d, 2.0 * s.d, s.c, lm=lm_bad, J=s.J)
    cv = checks.ad.fallback
    j2 = checks.ad.jvp(s.x, s.T, s.P, s.d, 2.0 * s.d, *s.tangents_with_h, lm=s.lm, dlm=s.dlm, dJ=s.dJ, J=s.J)
    v2 = checks.ad.vjp(s.x, s.T, s.P, s.d, 2.0 * s.d, s.c, lm=s.lm, J=s.J)
    ok = (
        int(jb[3][0]) == 0
        and int(vb[3][0]) == 0
        and not bool(jb[2][0])
        and not bool(vb[2][0])
        and cj["still_failed"] == 1
        and cv["still_failed"] == 1
        and all(torch.equal(a[1:], b[1:]) for a, b in zip(jb[:4], j2[:4], strict=True))
        and all(torch.equal(a[1:], b[1:]) for a, b in zip(vb[:4], v2[:4], strict=True))
    )
    verdict(
        "NaN log multiplier",
        ok,
        f"jvp {cj}; vjp {cv}; cell 0 takes no step and fails, the others bitwise the valid batch's",
    )


def test_missing_j_is_refused(checks):
    s = _directions(checks)
    with pytest.raises(ValueError, match="required"):
        checks.ad.jvp(s.x, s.T, s.P, s.d, 2.0 * s.d, *s.tangents_with_h)
    with pytest.raises(ValueError, match="required"):
        checks.ad.vjp(s.x, s.T, s.P, s.d, 2.0 * s.d, s.c)


@pytest.mark.parametrize(
    ("argument", "bad"),
    [
        ("lm", lambda a: a[:-1]),
        ("rc", lambda a: a[:-1]),
        ("dx", lambda a: a.float()),
        ("dT", lambda a: a[:-1]),
        ("dlm", lambda a: a[:-1]),
        ("dJ", lambda a: a.cpu()),
    ],
)
def test_wrong_tangent_inputs_name_the_argument(checks, argument, bad):
    s = _directions(checks)
    good = dict(zip(("dx", "dT", "dP", "ddt", "dh"), s.tan, strict=True)) | {
        "dlm": s.dlm,
        "dJ": s.dJ,
        "lm": s.lm,
        "rc": checks.ad.rc0,
    }
    good[argument] = bad(good[argument])
    with pytest.raises(ValueError, match=rf"^{argument}: float64 "):
        checks.ad.jvp(s.x, s.T, s.P, s.d, s.h, J=s.J, **good)


def test_zero_dimensional_tangent_is_refused_by_name(checks):
    s = _directions(checks)
    with pytest.raises(ValueError, match=r"^dx: \[B, S\] or \[D, B, S\]"):
        checks.ad.jvp(s.x, s.T, s.P, s.d, s.h, torch.tensor(0.0, dtype=torch.float64), *s.tan[1:], J=s.J)


@pytest.mark.parametrize("argument", ["gx", "gh"])
def test_wrong_cotangents_name_the_argument(checks, argument):
    s = _directions(checks)
    gh = torch.zeros_like(s.T)
    with pytest.raises(ValueError, match=rf"^{argument}: float64 "):
        checks.ad.vjp(
            s.x,
            s.T,
            s.P,
            s.d,
            s.h,
            s.c[:-1] if argument == "gx" else s.c,
            gh=gh[:-1] if argument == "gh" else gh,
            lm=s.lm,
            J=s.J,
        )


def test_negative_x_is_bad_input(checks, verdict):
    s = _directions(checks)
    j2 = checks.ad.jvp(s.x, s.T, s.P, s.d, 2.0 * s.d, *s.tangents_with_h, lm=s.lm, dlm=s.dlm, dJ=s.dJ, J=s.J)
    v2 = checks.ad.vjp(s.x, s.T, s.P, s.d, 2.0 * s.d, s.c, lm=s.lm, J=s.J)
    xneg = s.x.clone()
    xneg[0, 0] = -1e-3
    jb = checks.ad.jvp(xneg, s.T, s.P, s.d, 2.0 * s.d, *s.tangents_with_h, lm=s.lm, dlm=s.dlm, dJ=s.dJ, J=s.J)
    cj = checks.ad.fallback
    vb = checks.ad.vjp(xneg, s.T, s.P, s.d, 2.0 * s.d, s.c, lm=s.lm, J=s.J)
    cv = checks.ad.fallback
    negok = (
        cj["bad_input"] == 1
        and cv["bad_input"] == 1
        and not bool(jb[2][0])
        and not bool(vb[2][0])
        and all(torch.equal(a[1:], b[1:]) for a, b in zip(jb[:4], j2[:4], strict=True))
        and all(torch.equal(a[1:], b[1:]) for a, b in zip(vb[:4], v2[:4], strict=True))
        and torch.equal(jb[4][:, 1:], j2[4][:, 1:])  # tangents [D, B, S], [D, B]
        and torch.equal(jb[5][:, 1:], j2[5][:, 1:])
        and all(torch.equal(vb[4][k][1:], v2[4][k][1:]) for k in vb[4] if k != "rc")  # rc sums over the cells
    )
    verdict("negative x", negok, f"jvp {cj}; vjp {cv}; cell 0 not done, the others bitwise the valid batch's")
