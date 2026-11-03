"""The reduced-tracer call: the GCM advects fewer tracers than the network has species, and each cell keeps the others as
a memory of its own.

A design (a YAML file; the package's designs/ folder holds G31, N30sN2H, G30b, N30 and N26 for VULCAN's NCHO network) splits the network's species, He aside,
into carried species, each advected as its own tracer, and hidden species, advected only through M rows over them:
    name: ...
    about: ...
    carried: [species]        (names quoted: YAML reads a bare NO as false)
    hidden: [species]
    rows: [{element: "C"}, ..., {sum: {"C2H4": 1, "NH3": 1}, mass: 22.54}]
An element row holds the atoms of one element in the hidden species (weights: the atom counts); a sum row a fixed
non-negative molar sum of named hidden species. Every element that a hidden species holds needs its element row. The GCM
advects the K carried species and the M rows as mass mixing ratios: a carried species' mass per mass of gas, an element
row's mass of that element per mass of gas, a sum row's molar sum (mol per gram) times its stated mass (g/mol; a unit
only: a sum row adds no mass, its atoms are in the element rows). He is no tracer: it is the mass remainder, exact
because the carried species and the element rows hold every other atom. Each cell keeps the amounts of its hidden
species from its last call (the memory, not advected).

One call, per cell (inside, specific moles n_i = q_i / m_i, mol per gram of gas; mixing is linear in them):
 1. Projection of the memory onto the cell's row budgets b (after the GCM's mixing): maximum entropy relative to the
    memory, n_h = exp(lg_h + sum_r lam_r a_rh) on the H hidden species with lg = log(memory + 1e-12 max_r b_r / H),
    solving sum_h a_rh n_h = b_r on the dual. Newton steps: Jacobi-scaled, at most 30 in lam (natural-log units), each
    followed by a line search on the budget-scaled residual sum_r ((sum_h a_rh n_h - b_r) / b_r)^2 (Armijo, 1e-4; the
    longest step length that passes). A row with a zero budget removes the species that hold it; a row nobody holds is
    left to the fix. Up to IT1 iterations with the step lengths 1, 0.5, 0.25, 0.1, stopping once the largest relative
    budget residual is at most RTOL with finite densities, then one more iteration; a cell whose residual is still
    above RTOL after the IT1 iterations takes up to IT2 more, in blocks of BLK, with the step lengths 2^-k (k < 30),
    stopping after the first block that ends below RTOL. A cell still above RTOL (the residual measure can stall when
    the consistent answer is far from the memory, e.g. a hidden budget near one species' atom ratio) then minimises
    the convex dual objective sum_h n_h - sum_r b_r lam_r: up to P3IT damped Newton steps (Jacobi-scaled, no step cap;
    the longest step length 2^-k, k < 40, that passes the Armijo test (1e-4) on the objective, else the longest that
    lowers the budget-scaled residual, else stop), once from the multipliers reached and once from zero; the end
    point with the lowest residual (including the one before these steps) is kept.
 2. The exact fix: each hidden species is scaled by the smallest held / budget ratio of its rows (<= 1), and each
    element row's remainder (budget minus what the hidden species hold, >= 0) goes into that element's atomic species,
    so every element is exact to rounding whatever the projection reached. A cell whose largest relative budget residual
    before the fix exceeds CONV (absolute residuals below 1e-28 mol/g ignored: fp64 cannot resolve a budget that far
    below the cell's others) has not converged: it is counted and flagged, and it goes on with the fix of its last
    iterate (the spill path).
 3. The full state (carried and hidden species, He from the mass remainder), x = n / sum n, and the chemistry call
    (kincuda.CudaSolver: its guard and device-side recovery).
 4. New tracers from the result (n = x / sum_i x_i m_i): the carried species, the row sums of the hidden species, and
    the hidden species as the new memory; the mean molar mass, cp and (opt-in) the heating rate.
Steps 1-2 and 4 are CUDA kernels generated per design (NVRTC, as kincuda.py). For a design with hidden species whose
row weights on them are integers 0-12 (at most 6 rows), steps 1-2 run first in a kernel with one thread per cell: the
design's weights are literals in the code, the densities are formed as p_h prod_r exp(lam_r)^a_rh (p_h = exp(lg_h),
M exponentials per evaluation) with the row sums and the Newton matrix accumulated alongside, through the IT1
iterations, the polishing iteration, the fix and the outputs. A cell goes on to the one-warp-per-cell kernel, which redoes it from
its inputs, when an input is not finite, when an evaluation could leave the exponential's range (sum_r |lam_r|
max_h a_rh > 600, or log max_h p_h plus that sum > 690) or gives a non-finite density, when the cell needs more than
the IT1 iterations, or when its residual after the polishing iteration is above RTOL; these cells are counted
(handoff). Other designs (no hidden species, more than 6 rows, other weights) run the one-warp-per-cell kernel for
every cell. Step 4 is one warp per cell. Each cell's arithmetic depends only on its own inputs: results do not depend on
the batch size or order, and nothing is read back to the host during a call. The chemistry conserves mass, so the
tracers and the mean molar mass come back consistent.

TracerSolver(net, design, device, drift=None):
    (q, mem, T, P, dt, h, *, photo=None, heating=False) -> (q, mem, h, mu, cp, Q, flag, counts): the GCM-facing call.
        q [B, K + M] the advected tracers (mass mixing ratios: carried species in the design's order, then its rows),
        mem [B, H] the hidden species (mass mixing ratios, the design's order), T [B] K, P [B] dyn cm^-2, dt [B] s,
        h [B] the step sizes carried from the last call; float64 CUDA tensors. mem and h are the cell's own state
        (not advected; a restart stores them with the tracers). A run starts from split(x) and h = 1e-10 s; a zero
        memory is also a defined start (the projection's prior is then the floor alone, uniform over the hidden
        species). photo: per-cell photolysis rates [B, n_branch] in s^-1, one column per photolysis line of the network
        file in its order: not supported yet (anything but None raises NotImplementedError). Returns the new tracers,
        memory and step sizes; mu [B] the mean molar mass (g/mol); cp [B] the gas's specific heat at constant pressure
        at T after the call (erg g^-1 K^-1; NASA-9, low range below 1000 K); Q [B] with heating=True, else None: the
        chemistry's heating rate -sum_i H_i(T) (n_i after - n_i before the chemistry) / dt (erg g^-1 s^-1; the state
        before is the projected one, so the projection adds no heat); flag [B] int32 (bits 0-1: the chemistry guard's
        flag, 0 passed, 1 recovered, 2 still failed; bits 2-3: the projection, 0 converged in the first IT1 iterations,
        1 converged later, 2 not converged (spill path); bit 4: reconstruction defect, an advected row not reproduced
        (relative 1e-10, absolute 1e-30), a negative He remainder or a non-finite or negative state); counts (device
        int64 [len(COUNTS)], this call's: cells that took extra iterations, that took the dual-objective steps, that did
        not converge, with a defect, with a negative He remainder, handed to the warp kernel, then the chemistry's).
    split(x) -> (q, mem): tracers and memory of full states x [B, NS] (mole fractions), to start a run.
    step(C, R, G, T, P, dt, h) -> (C, R, G, h, x, flag): the same call in specific moles (carried species, rows,
        memory), with x [B, NS] the mole fractions after the call.
    prepare(C, R, G) -> (n, proj, bad): the full state before the chemistry (specific moles [B, NS]), the projection's
        status [B] and the reconstruction-defect mask [B] (adds to count).
    count / total: this call's counts / the counts since construction (device int64; COUNTS); fallback: this call's
    counts as a dict (one small copy); check(): the totals as a dict, raises if any cell was below the chemistry's
    300 K (kincuda). One call at a time on torch's current stream."""
import ctypes
import string
from pathlib import Path

import numpy as np
import torch
import yaml

from . import kincuda as KC
from . import kinvulcan as KV

F64 = torch.float64
DESIGNS = Path(__file__).resolve().parent / "designs"
AMASS = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "He": 4.002602, "S": 32.06}   # g/mol
RTOL, CONV, FLOOR, DCAP = 1e-10, 1e-8, 1e-12, 30.0   # DCAP: largest Newton step in lam before the line search
IT1, IT2, BLK, P3IT = 8, 60, 10, 400
ROWTOL, ROWABS = 1e-10, 1e-30                        # reconstruction: advected rows reproduced to this
COUNTS = ("extra", "dual", "unconverged", "defect", "he_negative", "handoff") + KC.COUNTS
RGAS = KV.KB * 6.02214076e23                         # erg mol^-1 K^-1
WARPS = 4                                            # warps (cells in flight) per block


def _fast_src(A, hidx, cidx, NS, MW, mrow, spill, relem, EL, IHE):
    """CUDA source of the one-thread-per-cell kernel prepfast (module docstring) for a design, or None (the warp kernel
    alone) if the design has no hidden species, more than 6 rows, or a weight on a hidden species that is not an integer
    in 0-12. A [M, H] the rows' weights on the hidden species, hidx / cidx the species indices of the hidden / carried
    species, EL [NS, E] element counts."""
    A = np.asarray(A, float)
    if not (A.size and A.shape[0] <= 6 and np.all(A == np.round(A)) and A.min() >= 0 and A.max() <= 12):
        return None
    A = A.astype(int)
    M, H = A.shape
    K, NE = len(cidx), EL.shape[1]
    f = lambda v: repr(float(v))  # noqa: E731
    pmax = [int(A[k].max()) for k in range(M)]

    def dens(name, rows, hess, store):
        o = [f"__device__ __forceinline__ bool {name}(const double* p, double lpm, const double* lam, double* rs, "
             "double* hm, double* n, bool* fin) {",
             "  double g = 0.0;"]
        o += [f"  g += fabs(lam[{k}]) * {f(pmax[k])};" for k in range(M)]
        o += ["  if (!(g <= 600.0) || !(lpm + g <= 690.0)) return false;"]
        for k in range(M):
            if pmax[k] > 0:
                o.append(f"  const double P{k}_1 = exp(lam[{k}]);")
                o += [f"  const double P{k}_{q} = P{k}_{q - 1} * P{k}_1;" for q in range(2, pmax[k] + 1)]
        if rows:
            o += [f"  rs[{k}] = 0.0;" for k in range(M)]
        if hess:
            o += [f"  hm[{e * M + g_}] = 0.0;" for e in range(M) for g_ in range(e, M)]
        o += ["  bool ok = true;", "  double v;"]
        for h in range(H):
            fac = "".join(f" * P{k}_{A[k, h]}" for k in range(M) if A[k, h] > 0)
            o.append(f"  v = p[{h}]{fac};")
            o.append("  ok = ok && isfinite(v);")
            if store:
                o.append(f"  n[{h}] = v;")
            if rows:
                o += [f"  rs[{k}] += {'v' if A[k, h] == 1 else f(A[k, h]) + ' * v'};" for k in range(M) if A[k, h] > 0]
            if hess:
                o += [f"  hm[{e * M + g_}] += {'v' if A[e, h] * A[g_, h] == 1 else f(A[e, h] * A[g_, h]) + ' * v'};"
                      for e in range(M) for g_ in range(e, M) if A[e, h] * A[g_, h] > 0]
        o += ["  *fin = ok;", "  return ok;", "}"]
        return "\n".join(o)

    o = [dens("fd_rsh", True, True, False), dens("fd_rs", True, False, False), dens("fd_n", False, False, True), r"""
// the Jacobi-scaled Newton direction (as ndir) from the accumulated matrix hm (upper triangle, unmasked)
__device__ __forceinline__ void tdir(const double* hm, const double* act, const double* r, double* d) {
  double Hm[M][M], dg[M], S[M][M], y[M];
  for (int e = 0; e < M; ++e)
    for (int f = e; f < M; ++f) {
      const double v = hm[e * M + f] * (act[e] * act[f]) + (e == f ? 1.0 - act[e] : 0.0);
      Hm[e][f] = v;
      Hm[f][e] = v;
    }
  for (int e = 0; e < M; ++e) dg[e] = 1.0 / sqrt(Hm[e][e] > 1e-300 ? Hm[e][e] : 1e-300);
  for (int e = 0; e < M; ++e) {
    for (int f = 0; f < M; ++f) S[e][f] = Hm[e][f] * dg[e] * dg[f] + (e == f ? 1e-14 : 0.0);
    y[e] = dg[e] * r[e];
  }
  for (int p = 0; p < M; ++p) {
    int piv = p;
    for (int i = p + 1; i < M; ++i) if (fabs(S[i][p]) > fabs(S[piv][p])) piv = i;
    if (piv != p) {
      for (int j = 0; j < M; ++j) { const double t = S[p][j]; S[p][j] = S[piv][j]; S[piv][j] = t; }
      const double t = y[p]; y[p] = y[piv]; y[piv] = t;
    }
    for (int i = p + 1; i < M; ++i) {
      const double f = S[i][p] / S[p][p];
      for (int j = p + 1; j < M; ++j) S[i][j] -= f * S[p][j];
      y[i] -= f * y[p];
    }
  }
  for (int i = M - 1; i >= 0; --i) {
    double v = y[i];
    for (int j = i + 1; j < M; ++j) v -= S[i][j] * y[j];
    y[i] = v / S[i][i];
  }
  for (int e = 0; e < M; ++e) {
    d[e] = dg[e] * y[e];
    if (!isfinite(d[e])) d[e] = 0.0;
  }
}

// one phase-1 Newton iteration (as newton(): step cap, line search over LS1, Armijo); stop: the early exit; false:
// the product form's range left (the cell goes to the warp kernel)
__device__ __forceinline__ bool fiter(const double* p, double lpm, double* lam, const double* b, const double* act,
                                      const double* s, int stop, bool* done) {
  double rs[M], hm[M * M], r[M], d[M];
  bool fin;
  *done = false;
  if (!fd_rsh(p, lpm, lam, rs, hm, 0, &fin)) return false;
  for (int k = 0; k < M; ++k) r[k] = (rs[k] - b[k]) * act[k];
  if (stop) {
    double res = 0.0;
    for (int k = 0; k < M; ++k) {
      const double v = fabs(r[k]) / s[k];
      res = (isnan(v) || v > res) ? v : res;
    }
    if (fin && !isnan(res) && res <= $RTOL) { *done = true; return true; }
  }
  tdir(hm, act, r, d);
  double dmax = 0.0;
  for (int e = 0; e < M; ++e) dmax = fmax(dmax, fabs(d[e]));
  const double cf = $DCAP / (dmax > $DCAP ? dmax : $DCAP);
  double f0 = 0.0;
  for (int k = 0; k < M; ++k) {
    d[k] *= cf;
    f0 += (r[k] / s[k]) * (r[k] / s[k]);
  }
  double t = 0.0;
  for (int l = 0; l < 4; ++l) {
    const double L = LS1[l];
    double lt[M], rt[M];
    for (int k = 0; k < M; ++k) lt[k] = lam[k] - L * d[k];
    bool tf;
    if (!fd_rs(p, lpm, lt, rt, 0, 0, &tf)) return false;
    double ft = 0.0;
    for (int k = 0; k < M; ++k) {
      const double v = (rt[k] - b[k]) * act[k] / s[k];
      ft += v * v;
    }
    if (ft <= (1.0 - 1e-4 * L * cf) * f0) { t = L; break; }
  }
  for (int k = 0; k < M; ++k) lam[k] -= t * d[k];
  return true;
}

// resid() of the product form (INF if not finite); false: range left
__device__ __forceinline__ bool fres(const double* p, double lpm, const double* lam, const double* b, const double* act,
                                     const double* s, double* res) {
  double rs[M];
  bool fin;
  if (!fd_rs(p, lpm, lam, rs, 0, 0, &fin)) return false;
  double v0 = 0.0;
  for (int k = 0; k < M; ++k) {
    const double v = fabs((rs[k] - b[k]) * act[k]) / s[k];
    v0 = (isnan(v) || v > v0) ? v : v0;
  }
  *res = (fin && !isnan(v0)) ? v0 : INF;
  return true;
}

extern "C" __global__ void prepfast(const double* __restrict__ C, const double* __restrict__ R,
    const double* __restrict__ G, int B, double* __restrict__ X, double* __restrict__ N, int* __restrict__ status,
    int* __restrict__ badv, u64* __restrict__ counts, int* __restrict__ idx, int* __restrict__ nidx) {
  for (int cell = blockIdx.x * blockDim.x + threadIdx.x; cell < B; cell += gridDim.x * blockDim.x) {
    const double* Cc = C + (size_t)cell * NC;
    const double* Rc = R + (size_t)cell * M;
    const double* Gc = G + (size_t)cell * NH;
    double b[M], act[M], s[M], p[NH], c[NC], n[NH];
    bool ok = true;
    double rm = 0.0, scale = 0.0, sm = 0.0;"""]
    o += [f"    b[{k}] = Rc[{k}]; ok = ok && isfinite(b[{k}]); rm += b[{k}] * {f(mrow[k])}; "
          f"scale = fmax(scale, fabs(b[{k}]));" for k in range(M)]
    o += [f"    c[{i}] = Cc[{i}]; ok = ok && isfinite(c[{i}]); sm += c[{i}] * {f(MW[cidx[i]])};" for i in range(K)]
    o += [f"    const double he = (1.0 - sm - rm) / {f(MW[IHE])};",
          "    scale = fmax(scale, 1e-30);",
          f"    const double fl = $FLOOR * scale / (double)NH;"]
    o += [f"    p[{h}] = Gc[{h}]; ok = ok && isfinite(p[{h}]); p[{h}] = fmax(p[{h}], 0.0) + fl;" for h in range(H)]
    o += ["    bool dlg = !ok;", "    double lam[M], res = INF;", "    for (int k = 0; k < M; ++k) lam[k] = 0.0;",
          "    if (!dlg) {"]
    for k in range(M):          # rows with holders and no budget remove their holders
        hold = [h for h in range(H) if A[k, h] > 0]
        o.append(f"      const bool z{k} = {'true' if hold else 'false'} && b[{k}] <= 0.0;")
    for h in range(H):
        zs = " || ".join(f"z{k}" for k in range(M) if A[k, h] > 0) or "false"
        o.append(f"      if ({zs}) p[{h}] = 0.0;")
    for k in range(M):
        hold = " || ".join(f"p[{h}] > 0.0" for h in range(H) if A[k, h] > 0) or "false"
        o.append(f"      act[{k}] = (({hold}) && b[{k}] > 0.0) ? 1.0 : 0.0; s[{k}] = act[{k}] > 0.0 ? b[{k}] : 1.0;")
    o.append(r"""      double pm = 0.0;
      for (int h = 0; h < NH; ++h) pm = fmax(pm, p[h]);
      const double lpm = log(pm);                                 // log of the largest prior (-inf if none is left)
      bool done = false;
      for (int it = 0; it < $IT1 && !done; ++it)
        if (!fiter(p, lpm, lam, b, act, s, 1, &done)) { dlg = true; break; }
      if (!dlg && !fres(p, lpm, lam, b, act, s, &res)) dlg = true;
      dlg = dlg || res > $RTOL;                                   // needs phase 2: the warp kernel
      if (!dlg && !fiter(p, lpm, lam, b, act, s, 0, &done)) dlg = true;   // the polishing iteration
      if (!dlg && !fres(p, lpm, lam, b, act, s, &res)) dlg = true;
      dlg = dlg || res > $RTOL;
      bool fin = true;
      if (!dlg && !fd_n(p, lpm, lam, 0, 0, n, &fin)) dlg = true;
      dlg = dlg || !fin;
    }
    if (dlg) {                                                    // the warp kernel redoes this cell
      idx[atomicAdd(nidx, 1)] = cell;
      atomicAdd(counts + 5, 1ULL);
      continue;
    }
    // the exact fix (as prepare)
    double qk[M], held, rem[M];""")
    for k in range(M):
        terms = " + ".join(f"{'' if A[k, h] == 1 else f(A[k, h]) + ' * '}n[{h}]" for h in range(H) if A[k, h] > 0) or "0.0"
        o.append(f"    held = {terms};")
        o.append(f"    {{ const double v = held > b[{k}] ? b[{k}] / fmax(held, 1e-300) : 1.0; "
                 f"qk[{k}] = fmin(fmax(v, 0.0), 1.0); }}")
    for h in range(H):
        o.append("    { double sc = 1.0;" + "".join(f" sc = fmin(sc, qk[{k}]);" for k in range(M) if A[k, h] > 0)
                 + f" n[{h}] *= sc; }}")
    for k in range(M):
        terms = " + ".join(f"{'' if A[k, h] == 1 else f(A[k, h]) + ' * '}n[{h}]" for h in range(H) if A[k, h] > 0) or "0.0"
        o.append(f"    rem[{k}] = fmax(b[{k}] - ({terms}), 0.0);")
    # the full state, in species order
    hpos = {j: h for h, j in enumerate(hidx)}
    cpos = {j: i for i, j in enumerate(cidx)}
    o.append("    double m[NS];")
    for j in range(NS):
        v = f"c[{cpos[j]}]" if j in cpos else f"n[{hpos[j]}]" if j in hpos else "0.0"
        v += "".join(f" + rem[{k}]" for k in range(M) if spill[k] == j)
        o.append(f"    m[{j}] = {'fmax(he, 0.0)' if j == IHE else v};")
    o.append("    bool nf = false, neg = false, off = false;")
    o.append("    double tot = 0.0, w, g;")
    o.append("    for (int j = 0; j < NS; ++j) { nf = nf || !isfinite(m[j]); neg = neg || m[j] < 0.0; tot += m[j]; }")
    for e in range(NE):
        wt = " + ".join(f"{'' if EL[j, e] == 1 else f(EL[j, e]) + ' * '}c[{cpos[j]}]" for j in cidx if EL[j, e] != 0)
        wt = (wt or "0.0") + "".join(f" + b[{k}]" for k in range(M) if relem[k] == e)
        gt = " + ".join(f"{'' if EL[j, e] == 1 else f(EL[j, e]) + ' * '}m[{j}]" for j in range(NS) if EL[j, e] != 0) or "0.0"
        o.append(f"    w = {wt}; g = {gt}; off = off || fabs(g - w) > $ROWTOL * fabs(w) + $ROWABS;")
    for k in range(M):
        if spill[k] < 0:
            terms = " + ".join(f"{'' if A[k, h] == 1 else f(A[k, h]) + ' * '}m[{hidx[h]}]" for h in range(H)
                               if A[k, h] > 0) or "0.0"
            o.append(f"    off = off || fabs(({terms}) - b[{k}]) > $ROWTOL * fabs(b[{k}]) + $ROWABS;")
    o.append(r"""    const bool bad = off || he < 0.0 || nf || neg;
    for (int j = 0; j < NS; ++j) {
      X[(size_t)cell * NS + j] = m[j] / tot;
      N[(size_t)cell * NS + j] = m[j];
    }
    status[cell] = 0;
    badv[cell] = bad;
    if (bad) atomicAdd(counts + 3, 1ULL);
    if (he < 0.0) atomicAdd(counts + 4, 1ULL);
  }
}
""")
    return "\n".join(o)


def molar_masses(species):
    """Molar masses (g/mol) of species names, from AMASS and the element counts of the names."""
    return [sum(AMASS[e] * c for e, c in KV.formula(s).items()) for s in species]


_SRC = r"""
#define NS $NS
#define M $M
#define NSL $NSL
#define NC $NC
#define NH $NH
#define NE $NE
#define IHE $IHE
#define FULL 0xffffffffu
typedef unsigned long long u64;
#define INF __longlong_as_double(0x7ff0000000000000ULL)
__constant__ double LS1[4] = {1.0, 0.5, 0.25, 0.1};

__device__ __forceinline__ double wsum(double v) {   // the warp's sum, lane 0's value in every lane
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return __shfl_sync(FULL, v, 0);
}

struct Cell {                       // lane l holds species j = l + 32 q, q < NSL
  double lg[NSL], a[M][NSL];        // log prior (-inf off the pool), row weights
  double b[M], act[M], s[M];        // budgets, active rows (1 / 0), residual scale
};

// n_j = exp(min(lg_j + sum_k lam_k a_kj, 700)) (a NaN stays NaN)
__device__ __forceinline__ void dens(const Cell& c, const double* lam, double* n) {
#pragma unroll
  for (int q = 0; q < NSL; ++q) {
    double e = 0.0;
#pragma unroll
    for (int k = 0; k < M; ++k) e += lam[k] * c.a[k][q];
    e = c.lg[q] + e;
    n[q] = exp(e > 700.0 ? 700.0 : e);
  }
}

__device__ __forceinline__ double rowsum(const Cell& c, const double* n, int k) {
  double v = 0.0;
#pragma unroll
  for (int q = 0; q < NSL; ++q) v += n[q] * c.a[k][q];
  return wsum(v);
}

// largest relative budget residual over the active rows (inf if n is not finite or the residual is NaN); n out
__device__ double resid(const Cell& c, const double* lam, double* n) {
  dens(c, lam, n);
  bool fin = true;
#pragma unroll
  for (int q = 0; q < NSL; ++q) fin = fin && isfinite(n[q]);
  fin = __all_sync(FULL, fin);
  double res = 0.0;
  for (int k = 0; k < M; ++k) {
    const double v = fabs((rowsum(c, n, k) - c.b[k]) * c.act[k]) / c.s[k];
    res = (isnan(v) || v > res) ? v : res;
  }
  return (fin && !isnan(res)) ? res : INF;
}

// the Jacobi-scaled Newton direction d (lam -= d) at the densities n with the residuals r (non-finite entries -> 0)
__device__ __forceinline__ void ndir(const Cell& c, const double* n, const double* r, double* d) {
  double Hm[M][M], dg[M], S[M][M], y[M];
  for (int e = 0; e < M; ++e)
    for (int f = e; f < M; ++f) {
      double v = 0.0;
#pragma unroll
      for (int q = 0; q < NSL; ++q) v += n[q] * c.a[e][q] * c.a[f][q];
      v = wsum(v) * (c.act[e] * c.act[f]) + (e == f ? 1.0 - c.act[e] : 0.0);
      Hm[e][f] = v;
      Hm[f][e] = v;
    }
  for (int e = 0; e < M; ++e) dg[e] = 1.0 / sqrt(Hm[e][e] > 1e-300 ? Hm[e][e] : 1e-300);
  for (int e = 0; e < M; ++e) {
    for (int f = 0; f < M; ++f) S[e][f] = Hm[e][f] * dg[e] * dg[f] + (e == f ? 1e-14 : 0.0);
    y[e] = dg[e] * r[e];
  }
  for (int p = 0; p < M; ++p) {           // Gaussian elimination with partial pivoting
    int piv = p;
    for (int i = p + 1; i < M; ++i) if (fabs(S[i][p]) > fabs(S[piv][p])) piv = i;
    if (piv != p) {
      for (int j = 0; j < M; ++j) { const double t = S[p][j]; S[p][j] = S[piv][j]; S[piv][j] = t; }
      const double t = y[p]; y[p] = y[piv]; y[piv] = t;
    }
    for (int i = p + 1; i < M; ++i) {
      const double f = S[i][p] / S[p][p];
      for (int j = p + 1; j < M; ++j) S[i][j] -= f * S[p][j];
      y[i] -= f * y[p];
    }
  }
  for (int i = M - 1; i >= 0; --i) {
    double v = y[i];
    for (int j = i + 1; j < M; ++j) v -= S[i][j] * y[j];
    y[i] = v / S[i][i];
  }
  for (int e = 0; e < M; ++e) {
    d[e] = dg[e] * y[e];
    if (!isfinite(d[e])) d[e] = 0.0;
  }
}

// iters Newton iterations on the dual: Jacobi scaling, step cap DCAP, line search over the step lengths (phase 1:
// LS1; phase 2: 2^-l, l < 30), the longest that passes the Armijo test on the budget-scaled residual (none: no step);
// with stop, the iterations end once the iterate passes resid()'s test
__device__ void newton(const Cell& c, double* lam, int iters, int phase2, int stop) {
  double n[NSL];
  for (int it = 0; it < iters; ++it) {
    dens(c, lam, n);
    double r[M], d[M];
    for (int k = 0; k < M; ++k) r[k] = (rowsum(c, n, k) - c.b[k]) * c.act[k];
    if (stop) {                             // this iterate already at RTOL (resid()'s test): stop
      bool fin = true;
#pragma unroll
      for (int q = 0; q < NSL; ++q) fin = fin && isfinite(n[q]);
      fin = __all_sync(FULL, fin);
      double res = 0.0;
      for (int k = 0; k < M; ++k) {
        const double v = fabs(r[k]) / c.s[k];
        res = (isnan(v) || v > res) ? v : res;
      }
      if (fin && !isnan(res) && res <= $RTOL) break;
    }
    ndir(c, n, r, d);
    double dmax = 0.0;
    for (int e = 0; e < M; ++e) dmax = fmax(dmax, fabs(d[e]));
    const double cf = $DCAP / (dmax > $DCAP ? dmax : $DCAP);
    double f0 = 0.0;
    for (int k = 0; k < M; ++k) {
      d[k] *= cf;
      f0 += (r[k] / c.s[k]) * (r[k] / c.s[k]);
    }
    double t = 0.0;
    const int nl = phase2 ? 30 : 4;
    for (int l = 0; l < nl; ++l) {
      const double L = phase2 ? ldexp(1.0, -l) : LS1[l];
      double lt[M], nt[NSL];
      for (int k = 0; k < M; ++k) lt[k] = lam[k] - L * d[k];
      dens(c, lt, nt);
      double ft = 0.0;
      for (int k = 0; k < M; ++k) {
        const double v = (rowsum(c, nt, k) - c.b[k]) * c.act[k] / c.s[k];
        ft += v * v;
      }
      if (ft <= (1.0 - 1e-4 * L * cf) * f0) { t = L; break; }
    }
    for (int k = 0; k < M; ++k) lam[k] -= t * d[k];
  }
}

// up to P3IT damped Newton steps on the convex dual objective f(lam) = sum_h n_h - sum_r b_r lam_r (active rows): the
// Jacobi-scaled direction, no step cap; the longest step length 2^-l (l < 40) that passes the Armijo test (1e-4) on f,
// else the longest that lowers the budget-scaled residual, else stop; stops once every active row is within RTOL
// (+ 1e-28 mol/g). A failed or non-finite trial counts as no decrease.
__device__ __noinline__ void dual(const Cell& c, double* lam) {
  double n[NSL];
  for (int it = 0; it < $P3IT; ++it) {
    dens(c, lam, n);
    double r[M], d[M], sn = 0.0;
    bool done = true;
    for (int k = 0; k < M; ++k) {
      r[k] = (rowsum(c, n, k) - c.b[k]) * c.act[k];
      done = done && fabs(r[k]) <= $RTOL * c.s[k] + 1e-28;
    }
    if (done) break;
    ndir(c, n, r, d);
#pragma unroll
    for (int q = 0; q < NSL; ++q) sn += n[q];
    double f0 = wsum(sn), m0 = 0.0, gd = 0.0;
    for (int k = 0; k < M; ++k) {
      f0 -= c.act[k] * c.b[k] * lam[k];
      m0 += (r[k] / c.s[k]) * (r[k] / c.s[k]);
      gd += r[k] * d[k];
    }
    double t = 0.0;
    for (int pass = 0; pass < 2 && t == 0.0; ++pass)
      for (int l = 0; l < 40; ++l) {
        const double L = ldexp(1.0, -l);
        double lt[M], nt[NSL], ft = 0.0;
        for (int k = 0; k < M; ++k) lt[k] = lam[k] - L * d[k];
        dens(c, lt, nt);
        if (pass == 0) {
          double st = 0.0;
#pragma unroll
          for (int q = 0; q < NSL; ++q) st += nt[q];
          ft = wsum(st);
          for (int k = 0; k < M; ++k) ft -= c.act[k] * c.b[k] * lt[k];
          if (ft <= f0 - 1e-4 * L * gd) { t = L; break; }
        } else {
          for (int k = 0; k < M; ++k) {
            const double v = (rowsum(c, nt, k) - c.b[k]) * c.act[k] / c.s[k];
            ft += v * v;
          }
          if (ft < m0) { t = L; break; }
        }
      }
    if (t == 0.0) break;
    for (int k = 0; k < M; ++k) lam[k] -= t * d[k];
  }
}

// the projection, the exact fix and the full state: X [B, NS] mole fractions and N [B, NS] specific moles before the
// chemistry, status (0 / 1 / 2), bad (reconstruction defect); counts: extra, dual, unconverged, defect, he_negative
extern "C" __global__ void prepare(const double* __restrict__ C, const double* __restrict__ R,
    const double* __restrict__ G, int B, double* __restrict__ X, double* __restrict__ N, int* __restrict__ status,
    int* __restrict__ badv, u64* __restrict__ counts, const double* __restrict__ AT, const int* __restrict__ cpos,
    const int* __restrict__ hpos, const double* __restrict__ MW, const double* __restrict__ mrow,
    const int* __restrict__ spill, const int* __restrict__ relem, const double* __restrict__ EL,
    const int* __restrict__ idx, const int* __restrict__ nidx) {
  const int lane = threadIdx.x & 31;
  const int nw = (gridDim.x * blockDim.x) >> 5;
  const int nloop = nidx ? *nidx : B;
  Cell c;
#pragma unroll
  for (int q = 0; q < NSL; ++q) {
    const int j = lane + 32 * q;
    for (int k = 0; k < M; ++k) c.a[k][q] = j < NS ? AT[k * NS + j] : 0.0;
  }
  for (int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; w < nloop; w += nw) {
    const int cell = nidx ? idx[w] : w;
    double c53[NSL], m[NSL], n[NSL];
    double sm = 0.0, rm = 0.0, scale = 0.0;
#pragma unroll
    for (int q = 0; q < NSL; ++q) {
      const int j = lane + 32 * q;
      c53[q] = (j < NS && cpos[j] >= 0) ? C[(size_t)cell * NC + cpos[j]] : 0.0;
      m[q] = (j < NS && hpos[j] >= 0) ? G[(size_t)cell * NH + hpos[j]] : 0.0;     // memory, for the prior
      if (j < NS) sm += c53[q] * MW[j];
    }
    for (int k = 0; k < M; ++k) {
      c.b[k] = R[(size_t)cell * M + k];
      rm += c.b[k] * mrow[k];
      scale = fmax(scale, fabs(c.b[k]));
    }
    const double he = (1.0 - wsum(sm) - rm) / MW[IHE];             // He: the mass remainder
    scale = fmax(scale, 1e-30);
#pragma unroll
    for (int q = 0; q < NSL; ++q) {
      const int j = lane + 32 * q;
      c.lg[q] = (j < NS && hpos[j] >= 0) ? log(fmax(m[q], 0.0) + $FLOOR * scale / (double)NH) : -INF;
    }
    bool zero[M];                                                   // rows with holders and no budget
    for (int k = 0; k < M; ++k) {
      bool h = false;
#pragma unroll
      for (int q = 0; q < NSL; ++q) h = h || (isfinite(c.lg[q]) && c.a[k][q] > 0.0);
      zero[k] = __any_sync(FULL, h) && c.b[k] <= 0.0;
    }
#pragma unroll
    for (int q = 0; q < NSL; ++q) {
      bool kill = false;
      for (int k = 0; k < M; ++k) kill = kill || (zero[k] && c.a[k][q] > 0.0);
      if (kill) c.lg[q] = -INF;
    }
    for (int k = 0; k < M; ++k) {
      bool h = false;
#pragma unroll
      for (int q = 0; q < NSL; ++q) h = h || (isfinite(c.lg[q]) && c.a[k][q] > 0.0);
      const bool act = __any_sync(FULL, h) && c.b[k] > 0.0;
      c.act[k] = act ? 1.0 : 0.0;
      c.s[k] = act ? c.b[k] : 1.0;
    }
    double lam[M];
    for (int k = 0; k < M; ++k) lam[k] = 0.0;
    newton(c, lam, $IT1, 0, 1);
    double res = resid(c, lam, n);
    const bool need = res > $RTOL;
    if (need)
      for (int it = 0; it < $IT2; it += $BLK) {
        newton(c, lam, $BLK, 1, 0);
        res = resid(c, lam, n);
        if (!(res > $RTOL)) break;
      }
    if (!need) {                                                  // converged in phase 1: one more iteration
      newton(c, lam, 1, 0, 0);
      res = resid(c, lam, n);
    }
    const bool p3 = res > $RTOL;                                  // the dual objective, from lam and from 0
    if (p3) {
      double best[M], l[M], bres = res;
      for (int k = 0; k < M; ++k) best[k] = lam[k];
      for (int s0 = 0; s0 < 2; ++s0) {
        for (int k = 0; k < M; ++k) l[k] = s0 ? 0.0 : lam[k];
        dual(c, l);
        const double r = resid(c, l, n);
        if (r < bres) {
          bres = r;
          for (int k = 0; k < M; ++k) best[k] = l[k];
        }
      }
      for (int k = 0; k < M; ++k) lam[k] = best[k];
      res = resid(c, lam, n);
    }
    bool fin = true;
#pragma unroll
    for (int q = 0; q < NSL; ++q) fin = fin && isfinite(n[q]);
    fin = __all_sync(FULL, fin);
    bool small = true;
    for (int k = 0; k < M; ++k)
      small = small && fabs((rowsum(c, n, k) - c.b[k]) * c.act[k]) <= $CONV * c.s[k] + 1e-28;
    const bool conv = res <= $CONV || (small && fin);
    // the exact fix: scale each species by its rows' smallest held / budget ratio, spill the element remainders
#pragma unroll
    for (int q = 0; q < NSL; ++q) if (!isfinite(n[q])) n[q] = 0.0;
    double qk[M];
    for (int k = 0; k < M; ++k) {
      const double held = rowsum(c, n, k);
      const double v = held > c.b[k] ? c.b[k] / fmax(held, 1e-300) : 1.0;
      qk[k] = fmin(fmax(v, 0.0), 1.0);
    }
#pragma unroll
    for (int q = 0; q < NSL; ++q) {
      double sc = 1.0;
      for (int k = 0; k < M; ++k) if (c.a[k][q] > 0.0) sc = fmin(sc, qk[k]);
      n[q] *= sc;
    }
    double rem[M];
    for (int k = 0; k < M; ++k) rem[k] = fmax(c.b[k] - rowsum(c, n, k), 0.0);
    bool nf = false, neg = false;
    double tot = 0.0;
#pragma unroll
    for (int q = 0; q < NSL; ++q) {
      const int j = lane + 32 * q;
      m[q] = c53[q] + n[q];
      for (int k = 0; k < M; ++k) if (spill[k] == j) m[q] += rem[k];
      if (j == IHE) m[q] = fmax(he, 0.0);
      if (j < NS) {
        nf = nf || !isfinite(m[q]);
        neg = neg || m[q] < 0.0;
        tot += m[q];
      }
    }
    // reconstruction: each element of the tracers (carried species' atoms plus its element row) and each sum row
    bool off = false;
    for (int e = 0; e < NE; ++e) {
      double w = 0.0, g = 0.0;
#pragma unroll
      for (int q = 0; q < NSL; ++q) {
        const int j = lane + 32 * q;
        if (j < NS) { w += c53[q] * EL[j * NE + e]; g += m[q] * EL[j * NE + e]; }
      }
      w = wsum(w);
      g = wsum(g);
      for (int k = 0; k < M; ++k) if (relem[k] == e) w += c.b[k];
      off = off || fabs(g - w) > $ROWTOL * fabs(w) + $ROWABS;
    }
    for (int k = 0; k < M; ++k)
      if (spill[k] < 0) off = off || fabs(rowsum(c, m, k) - c.b[k]) > $ROWTOL * fabs(c.b[k]) + $ROWABS;
    const bool bad = off || he < 0.0 || __any_sync(FULL, nf) || __any_sync(FULL, neg);
    tot = wsum(tot);
#pragma unroll
    for (int q = 0; q < NSL; ++q) {
      const int j = lane + 32 * q;
      if (j < NS) {
        X[(size_t)cell * NS + j] = m[q] / tot;
        N[(size_t)cell * NS + j] = m[q];
      }
    }
    if (lane == 0) {
      status[cell] = conv ? (need ? 1 : 0) : 2;
      badv[cell] = bad;
      if (need) atomicAdd(counts + 0, 1ULL);
      if (p3) atomicAdd(counts + 1, 1ULL);
      if (!conv) atomicAdd(counts + 2, 1ULL);
      if (bad) atomicAdd(counts + 3, 1ULL);
      if (he < 0.0) atomicAdd(counts + 4, 1ULL);
    }
  }
}

// the new tracers from the call's result X2 [B, NS]: carried species C2 [B, NC], rows R2 [B, M], memory G2 [B, NH]
// (specific moles), mean molar mass MU, cp CP (erg/g/K) and, if heating, Q (erg/g/s) with N the state before the call
extern "C" __global__ void reform(const double* __restrict__ X2, const double* __restrict__ N,
    const double* __restrict__ T, const double* __restrict__ DT, int B, int heating, double* __restrict__ C2,
    double* __restrict__ R2, double* __restrict__ G2, double* __restrict__ MU, double* __restrict__ CP,
    double* __restrict__ Q, const double* __restrict__ AT, const int* __restrict__ cpos, const int* __restrict__ hpos,
    const double* __restrict__ MW, const double* __restrict__ LO, const double* __restrict__ HI) {
  const int lane = threadIdx.x & 31;
  const int nw = (gridDim.x * blockDim.x) >> 5;
  for (int cell = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; cell < B; cell += nw) {
    double x[NSL], mw = 0.0;
#pragma unroll
    for (int q = 0; q < NSL; ++q) {
      const int j = lane + 32 * q;
      x[q] = j < NS ? X2[(size_t)cell * NS + j] : 0.0;
      if (j < NS) mw += x[q] * MW[j];
    }
    mw = wsum(mw);
    const double t = T[cell];
    const double* a = t < 1000.0 ? LO : HI;
    double cp = 0.0, hq = 0.0, rk[M];
    for (int k = 0; k < M; ++k) rk[k] = 0.0;
#pragma unroll
    for (int q = 0; q < NSL; ++q) {
      const int j = lane + 32 * q;
      if (j >= NS) continue;
      const double n = x[q] / mw;
      if (cpos[j] >= 0) C2[(size_t)cell * NC + cpos[j]] = n;
      if (hpos[j] >= 0) {
        G2[(size_t)cell * NH + hpos[j]] = n;
        for (int k = 0; k < M; ++k) rk[k] += n * AT[k * NS + j];
      }
      const double* p = a + 9 * j;
      cp += n * (p[0] / (t * t) + p[1] / t + p[2] + p[3] * t + p[4] * t * t + p[5] * t * t * t + p[6] * t * t * t * t);
      if (heating)
        hq += (n - N[(size_t)cell * NS + j]) * (-p[0] / (t * t) + p[1] * log(t) / t + p[2] + p[3] * t / 2.0
              + p[4] * t * t / 3.0 + p[5] * t * t * t / 4.0 + p[6] * t * t * t * t / 5.0 + p[7] / t);
    }
    for (int k = 0; k < M; ++k) rk[k] = wsum(rk[k]);
    cp = wsum(cp);
    hq = wsum(hq);
    if (lane == 0) {
      MU[cell] = mw;
      CP[cell] = $RGAS * cp;
      if (heating) Q[cell] = -$RGAS * t * hq / DT[cell];
      for (int k = 0; k < M; ++k) R2[(size_t)cell * M + k] = rk[k];
    }
  }
}
"""


class TracerSolver:
    """The reduced-tracer call of the module docstring for one design (a YAML file, or a name in the package's designs/ folder)."""

    def __init__(self, net, design, device, drift=None):
        path = Path(design)
        if not path.suffix:
            path = DESIGNS / f"{design}.yaml"
        d = yaml.safe_load(path.read_text())
        self.dev = dev = torch.device(device)
        sp = [str(s) for s in net.species]
        NS = len(sp)
        self.name, self.species = d["name"], sp
        self.carried, self.hidden = list(d["carried"]), list(d["hidden"])
        if "He" not in sp:
            raise ValueError("the network has no He (the mass remainder)")
        if not all(isinstance(s, str) for s in self.carried + self.hidden):
            raise ValueError(f"{path}: carried and hidden must list species names (quoted: YAML reads a bare NO as false)")
        every = self.carried + self.hidden
        if len(set(every)) != len(every) or set(every) != set(sp) - {"He"}:
            raise ValueError(f"{path}: carried and hidden must split the network's species other than He")
        self.carried.sort(key=sp.index)
        self.hidden.sort(key=sp.index)
        atoms = np.asarray(net.atoms, float)
        el = list(net.elements)
        hidn = np.array([s in self.hidden for s in sp])
        AT, unit, mrow, spill = [], [], [], []
        for r in d["rows"]:
            if "element" in r:
                e = r["element"]
                if e not in el or e == "He" or e not in sp:
                    raise ValueError(f"{path}: element row {e}: not an element of the network with its atomic species")
                AT.append(atoms[:, el.index(e)])
                unit.append(AMASS[e]), mrow.append(AMASS[e]), spill.append(sp.index(e))
            else:
                w = np.zeros(NS)
                for q, v in r["sum"].items():
                    if q not in self.hidden or not float(v) > 0:
                        raise ValueError(f"{path}: sum row over {q}: a hidden species with a positive weight")
                    w[sp.index(q)] = float(v)
                if not float(r["mass"]) > 0:
                    raise ValueError(f"{path}: a sum row needs a positive mass (its tracer's unit)")
                AT.append(w), unit.append(float(r["mass"])), mrow.append(0.0), spill.append(-1)
        held = [e for e in el if e != "He" and atoms[hidn, el.index(e)].any()]
        rowel = [r["element"] for r in d["rows"] if "element" in r]
        if sorted(rowel) != sorted(held):
            raise ValueError(f"{path}: element rows {rowel}; the hidden species hold {held} (one row each)")
        M = len(AT)
        self.K, self.M, self.H = len(self.carried), M, len(self.hidden)
        els = [e for e in el if e != "He"]
        t = lambda a, dt=F64: torch.tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
        self.MW = t(molar_masses(sp))
        self.ic = torch.tensor([sp.index(s) for s in self.carried], device=dev)
        self.ih = torch.tensor([sp.index(s) for s in self.hidden], device=dev)
        self.car, self.hid = (torch.zeros(NS, dtype=torch.bool, device=dev) for _ in range(2))
        self.car[self.ic] = True
        self.hid[self.ih] = True
        self.AT = t(np.stack(AT))                                     # [M, NS] row weights (mol units)
        self.A = self.AT.T
        self.unit = t(unit)                                           # g/mol: each row's tracer unit
        self.EL = t(atoms[:, [el.index(e) for e in els]])            # [NS, E] element counts
        self.mwc, self.mwh = self.MW[self.ic], self.MW[self.ih]
        self.qunit = torch.cat([self.mwc, self.unit])
        cpos, hpos = np.full(NS, -1), np.full(NS, -1)
        cpos[[sp.index(s) for s in self.carried]] = np.arange(self.K)
        hpos[[sp.index(s) for s in self.hidden]] = np.arange(self.H)
        relem = [els.index(sp[i]) if i >= 0 else -1 for i in spill]
        RE = np.zeros((M, len(els)))
        for k, e in enumerate(relem):
            if e >= 0:
                RE[k, e] = 1.0
        self.RE = t(RE)                                               # [M, E] element row -> its element
        self.spill = spill                                            # each row's atomic species (-1: a sum row)
        i32 = torch.int32
        self._tab = [self.AT.contiguous(), t(cpos, i32), t(hpos, i32), self.MW, t(mrow), t(spill, i32), t(relem, i32),
                     self.EL.contiguous()]
        self._nasa = [t(net.lo).contiguous(), t(net.hi).contiguous()]   # NASA-9 a1..a7, b1, b2 per species
        hidx = [sp.index(s_) for s_ in self.hidden]
        fsrc = _fast_src(np.stack(AT)[:, hidx], hidx, [sp.index(s_) for s_ in self.carried], NS, molar_masses(sp),
                         mrow, spill, relem, atoms[:, [el.index(e) for e in els]], sp.index("He"))
        self.fast = fsrc is not None
        src = string.Template(_SRC + (fsrc or "")).substitute(
            NS=NS, M=M, NSL=-(-NS // 32), NC=max(self.K, 1), NH=max(self.H, 1), NE=len(els), IHE=sp.index("He"),
            FLOOR=repr(FLOOR), DCAP=repr(DCAP), RTOL=repr(RTOL), CONV=repr(CONV), IT1=IT1, IT2=IT2, BLK=BLK, P3IT=P3IT,
            ROWTOL=repr(ROWTOL), ROWABS=repr(ROWABS), RGAS=repr(RGAS))
        prop = torch.cuda.get_device_properties(dev)
        cu = KC._cuda()
        self._mod = ctypes.c_void_p()
        KC._check(cu.cuModuleLoadData(ctypes.byref(self._mod), KC.compile_cubin(src, f"{prop.major}{prop.minor}")),
                  "load")
        self._fn = {}
        for name in ("prepare", "reform") + (("prepfast",) if self.fast else ()):
            fn = ctypes.c_void_p()
            KC._check(cu.cuModuleGetFunction(ctypes.byref(fn), self._mod, name.encode()), "function")
            self._fn[name] = fn
        self._blocks = prop.multi_processor_count * 16                 # most blocks per launch (cells loop over warps)
        self.kc = KC.CudaSolver(net, dev, drift)
        self._nidx = torch.zeros(1, dtype=torch.int32, device=dev)
        self.count = torch.zeros(len(COUNTS), dtype=torch.int64, device=dev)
        self.total = torch.zeros(len(COUNTS), dtype=torch.int64, device=dev)

    def _launch(self, name, B, args):
        grid = max(1, min(-(-B // WARPS), self._blocks))
        KC._launch(self._fn[name], grid, 32 * WARPS, 0, torch.cuda.current_stream(self.dev).cuda_stream, args)

    def _f64(self, *a):
        for x in a:
            if not (x.is_cuda and x.dtype == F64):
                raise ValueError("float64 CUDA tensors")
        return [x.contiguous() for x in a]

    # -- the call -------------------------------------------------------------------------------------------------------
    def _prepare(self, C, R, G):
        C, R, G = self._f64(C, R, G)
        B, NS = C.shape[0], len(self.species)
        if C.shape[1] != self.K or R.shape[1] != self.M or G.shape[1] != self.H or R.shape[0] != B or G.shape[0] != B:
            raise ValueError(f"C [B, {self.K}], R [B, {self.M}], G [B, {self.H}] for design {self.name}")
        x = torch.empty(B, NS, dtype=F64, device=self.dev)
        n = torch.empty_like(x)
        status, bad = (torch.empty(B, dtype=torch.int32, device=self.dev) for _ in range(2))
        if self.fast:
            self._nidx.zero_()
            idx = torch.empty(B, dtype=torch.int32, device=self.dev)
            KC._launch(self._fn["prepfast"], max(1, -(-B // 128)), 128, 0,
                       torch.cuda.current_stream(self.dev).cuda_stream,
                       [C, R, G, ctypes.c_int(B), x, n, status, bad, self.count, idx, self._nidx])
            self._launch("prepare", B, [C, R, G, ctypes.c_int(B), x, n, status, bad, self.count, *self._tab, idx,
                                        self._nidx])
        else:
            self._launch("prepare", B, [C, R, G, ctypes.c_int(B), x, n, status, bad, self.count, *self._tab,
                                        ctypes.c_void_p(0), ctypes.c_void_p(0)])
        return x, n, status, bad

    def prepare(self, C, R, G):
        """The full state before the chemistry: (n [B, NS] specific moles, status [B], bad [B] bool)."""
        _, n, status, bad = self._prepare(C, R, G)
        return n, status, bad.bool()

    def _call(self, C, R, G, T, P, dt, h, heating=False):
        """prepare, the chemistry, the new tracers: (C, R, G, h, x, flag, mu, cp, Q)."""
        self.count.zero_()
        x, n, status, bad = self._prepare(C, R, G)
        T, P, dt, h = self._f64(T, P, dt, h)
        x2, h2, _, _, cflag = self.kc.launch(x, T, P, dt, h)
        self.kc.total += self.kc.count
        self.count[len(COUNTS) - len(KC.COUNTS):] += self.kc.count[1:9]
        self.total += self.count
        B = x.shape[0]
        e = lambda *s: torch.empty(*s, dtype=F64, device=self.dev)  # noqa: E731
        C2, R2, G2, mu, cp = e(B, self.K), e(B, self.M), e(B, self.H), e(B), e(B)
        Q = e(B) if heating else mu
        self._launch("reform", B, [x2, n, T, dt, ctypes.c_int(B), ctypes.c_int(int(heating)), C2, R2, G2, mu, cp, Q,
                                   self._tab[0], self._tab[1], self._tab[2], self.MW, *self._nasa])
        flag = cflag | (status << 2) | (bad << 4)
        return C2, R2, G2, h2, x2, flag, mu, cp, (Q if heating else None)

    def step(self, C, R, G, T, P, dt, h):
        """The call in specific moles: (C, R, G, h, x, flag); see the module docstring."""
        return self._call(C, R, G, T, P, dt, h)[:6]

    def __call__(self, q, mem, T, P, dt, h, *, photo=None, heating=False):
        """The GCM-facing call: (q, mem, h, mu, cp, Q, flag, counts); see the module docstring."""
        if photo is not None:
            raise NotImplementedError("per-cell photolysis rates are not supported yet (photo must be None)")
        if q.shape[1] != self.K + self.M or mem.shape[1] != self.H:
            raise ValueError(f"q [B, {self.K + self.M}] and mem [B, {self.H}] for design {self.name}")
        C, R, G, h2, _, flag, mu, cp, Q = self._call(q[:, :self.K] / self.mwc, q[:, self.K:] / self.unit,
                                                     mem / self.mwh, T, P, dt, h, heating)
        return torch.cat([C, R], 1) * self.qunit, G * self.mwh, h2, mu, cp, Q, flag, self.count.clone()

    def split(self, x):
        """Tracers q [B, K + M] and memory mem [B, H] (mass mixing ratios) of full states x [B, NS] (mole fractions)."""
        x = self._f64(x)[0]
        B = x.shape[0]
        e = lambda *s: torch.empty(*s, dtype=F64, device=self.dev)  # noqa: E731
        C, R, G, mu, cp = e(B, self.K), e(B, self.M), e(B, self.H), e(B), e(B)
        T = torch.full((B,), 1000.0, dtype=F64, device=self.dev)
        self._launch("reform", B, [x, x, T, T, ctypes.c_int(B), ctypes.c_int(0), C, R, G, mu, cp, mu, self._tab[0],
                                   self._tab[1], self._tab[2], self.MW, *self._nasa])
        return torch.cat([C, R], 1) * self.qunit, G * self.mwh

    @property
    def fallback(self):
        """This call's counts as a dict (one small copy)."""
        return dict(zip(COUNTS, self.count.tolist()))

    def check(self):
        """The counts since construction as a dict (one small copy); raises if any cell was below the chemistry's
        300 K."""
        self.kc.check()
        return dict(zip(COUNTS, self.total.tolist()))
