// The tangent kernels: the call with its directional derivatives.

// The tangent of F = (n_tot / nd) S v at Y, Fd (unless null), for the state tangent Yd and the rate tangents (dk:
// d kf, d kr; dlnd: d ln nd); leaves n, n1 = the concentrations and their tangent; sv, svd = S v and its tangent
// (va: a per-reaction work array). Returns n_tot, td = sum Yd.
__device__ __forceinline__ double rhs_dual(double* work, const double* Y, const double* Yd, double* Fd, double* va,
                                           const Rates& k, const double* dk, double dlnd, double& td, const Tab& tab,
                                           const Streams& st, int lane) {
  double *n = work + A_N0, *n1 = work + A_N1, *v = work + A_V, *part = work + A_PART, *sv = work + A_SVN,
         *svd = work + A_SV1N;
  const double *kf = k.kf, *kr = k.kr, nd = k.nd;
  double p = 0.0, pd = 0.0;
  for (int i = lane; i < NS; i += 32) {
    p += Y[i];
    pd += Yd[i];
  }
  const double ntot = wsum(p), xf = nd / ntot;
  td = wsum(pd);
  const double rt = td / ntot;
  for (int i = lane; i < NS; i += 32) {
    n[i] = Y[i] * xf;
    n1[i] = xf * (Yd[i] - Y[i] * rt) + n[i] * dlnd;
  }
  if (lane == 0) {
    n[NS] = 1.0;
    n1[NS] = 0.0;
  }
  __syncwarp();
  for (int r = lane; r < NR; r += 32) {
    const u8* q = tab.rx + 6 * r;
    const double a0 = n[q[0]], a1 = n[q[1]], a2 = n[q[2]], b0 = n[q[3]], b1 = n[q[4]], b2 = n[q[5]];
    const double P = a0 * a1 * a2, Q = b0 * b1 * b2;
    const double Pd = n1[q[0]] * a1 * a2 + a0 * n1[q[1]] * a2 + a0 * a1 * n1[q[2]];
    const double Qd = n1[q[3]] * b1 * b2 + b0 * n1[q[4]] * b2 + b0 * b1 * n1[q[5]];
    v[r] = kf[r] * P - kr[r] * Q;
    va[r] = dk[r] * P + kf[r] * Pd - (dk[NR + r] * Q + kr[r] * Qd);
  }
  __syncwarp();
  stoichiometry_times_two(v, va, sv, svd, part, tab, st.sst, st.sround, lane);
  if (Fd) {
    const double g = ntot / nd, gd = g * (rt - dlnd);
    for (int i = lane; i < NS; i += 32) Fd[i] = svd[i] * g + sv[i] * gd;
    __syncwarp();
  }
  return ntot;
}

// out = d/de [J(N) K] along the tangent (N + e Nd, k + e dk, ln nd + e dlnd): the second derivative of F, reaction
// by reaction, from hyper-dual products. N, Nd = the state and its tangent; n, n1 = the concentrations at N and their
// tangent; svN, sv1N = S v at N and its tangent; ntot, td = sum N, sum Nd; sv2, sv12 = work arrays.
__device__ __forceinline__ void jacobian_dot(double* work, const double* K, double* out, double ntot, double td,
                                             const Rates& k, const double* dk, double dlnd, double* va, const Tab& tab,
                                             const Streams& st, int lane) {
  double *n = work + A_N0, *n1 = work + A_N1, *n2 = work + A_N2, *n12 = work + A_N12, *v = work + A_V,
         *part = work + A_PART, *sv2 = work + A_T1, *sv12 = work + A_T2;
  const double *N = work + A_NN, *Nd = work + A_ND, *svN = work + A_SVN, *sv1N = work + A_SV1N;
  const double *kf = k.kf, *kr = k.kr, nd = k.nd;
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += K[i];
  const double t2 = wsum(p);
  for (int i = lane; i < NS; i += 32) {
    const double x = N[i] / ntot, x1 = (Nd[i] - x * td) / ntot, x2 = (K[i] - x * t2) / ntot;
    const double x12 = -(x1 * t2 + x2 * td) / ntot;
    n2[i] = nd * x2;
    n12[i] = nd * (x12 + x2 * dlnd);
  }
  if (lane == 0) {
    n2[NS] = 0.0;
    n12[NS] = 0.0;
  }
  __syncwarp();
  for (int r = lane; r < NR; r += 32) {
    const u8* q = tab.rx + 6 * r;
    double hp[2][4];
#pragma unroll
    for (int s = 0; s < 2; s++) {
      double p0 = 1.0, p1 = 0.0, p2 = 0.0, p12 = 0.0;
#pragma unroll
      for (int m = 0; m < 3; m++) {
        const int j = q[3 * s + m];
        const double a = n[j], a1 = n1[j], a2 = n2[j], a12 = n12[j];
        p12 = p12 * a + p1 * a2 + p2 * a1 + p0 * a12;
        p1 = p1 * a + p0 * a1;
        p2 = p2 * a + p0 * a2;
        p0 = p0 * a;
      }
      hp[s][0] = p0;
      hp[s][1] = p1;
      hp[s][2] = p2;
      hp[s][3] = p12;
    }
    v[r] = kf[r] * hp[0][2] - kr[r] * hp[1][2];
    va[r] = kf[r] * hp[0][3] + dk[r] * hp[0][2] - (kr[r] * hp[1][3] + dk[NR + r] * hp[1][2]);
  }
  __syncwarp();
  stoichiometry_times_two(v, va, sv2, sv12, part, tab, st.sst, st.sround, lane);
  const double g = ntot / nd, g1 = g * (td / ntot - dlnd), g2 = t2 / nd, g12 = -t2 * dlnd / nd;
  for (int i = lane; i < NS; i += 32) out[i] = sv12[i] * g + sv1N[i] * g2 + sv2[i] * g1 + svN[i] * g12;
  __syncwarp();
}

// The error's tangent of the attempt just taken, from Kd4 and the state tangents Nd, Yd: the RMS's, or on a first try
// (en = max(RMS, max-norm)) the selected norm's (half each at a tie; the max-norm's shared evenly among the species
// attaining it).
__device__ __forceinline__ double error_tangent(const double* N, const double* Nd, const double* Y, const double* Yd,
                                                const double* K4, const double* Kd4, double en, bool first, double rtol,
                                                double atol, int lane) {
  double e = 0.0, e2 = 0.0, q_max = -1.0, ld = 0.0, lc = 0.0;  // q_max, ld, lc: the lane's max |q|, its tangent, ties
  for (int i = lane; i < NS; i += 32) {
    const double aN = fabs(N[i]), aY = fabs(Y[i]), sc = atol + rtol * fmax(aN, aY);
    const double dN = sgn(N[i]) * Nd[i], dY = sgn(Y[i]) * Yd[i];
    const double dsc = rtol * (aN > aY ? dN : (aN < aY ? dY : 0.5 * (dN + dY)));
    const double q = K4[i] / sc;
    e += q * (Kd4[i] - q * dsc) / sc;
    if (first) {
      e2 += q * q;
      const double aq = fabs(q), dq = sgn(q) * (Kd4[i] - q * dsc) / sc;
      if (aq > q_max) {
        q_max = aq;
        ld = dq;
        lc = 1.0;
      } else if (aq == q_max) {
        ld += dq;
        lc += 1.0;
      }
    }
  }
  if (!first) return wsum(e) / (NS * en);
  const double er = sqrt(wsum(e2) / NS), emx = wmax(q_max);  // the attempt's RMS and max-norm, bitwise
  const double dm = wsum(q_max == emx ? ld : 0.0) / wsum(q_max == emx ? lc : 0.0);
  const double wr = er > emx ? 1.0 : (er < emx ? 0.0 : 0.5);
  return wr * (wsum(e) / (NS * er)) + (1.0 - wr) * dm;
}

// Tangent of the attempt a just taken (K1..K4, Y, the factors, u, a.c0 in place; size ha, first while no step of the
// call is accepted; DN: the recovery's dense factors) for the state tangent Nd, the step-size tangent had and the rate
// tangents: Yd = the new state's tangent; returns the error's tangent (adapt) or 0.
// size-limit-ok: 17 arguments: the attempt's factors and outcome, one direction's three tangents, the work arrays
template <bool DN>
__device__ __forceinline__ double tangent_step(double* work, const double (&c)[NT], const double* Wd, const Attempt& a,
                                               double ha, bool first, double had, int adapt, const Rates& k,
                                               const double* dk, double dlnd, double* va, const Tolerance& tol,
                                               const Tab& tab, const Streams& st, int lane, int myt) {
  double *N = work + A_NN, *Nd = work + A_ND, *Y = work + A_Y, *Yd = work + A_YD, *Fd = work + A_FD,
         *svN = work + A_SVN, *sv1N = work + A_SV1N, *Ys = work + A_YS, *Yds = work + A_YDS, *K = work + A_K1,
         *Kd = work + A_KD1;
  double *K1 = K, *K2 = K + NS, *K3 = K + 2 * NS, *K4 = K + 3 * NS;
  double *Kd1 = Kd, *Kd2 = Kd + NS, *Kd3 = Kd + 2 * NS, *Kd4 = Kd + 3 * NS;
  const double ih = 1.0 / ha, hh = had * ih * ih, c0 = a.c0;
  double td = 0.0, tdx = 0.0;
  const double ntot = rhs_dual(work, N, Nd, nullptr, va, k, dk, dlnd, td, tab, st, lane);
  for (int s = 0; s < 4; s++) {
    jacobian_dot(work, K + s * NS, Kd + s * NS, ntot, td, k, dk, dlnd, va, tab, st, lane);
    for (int i = lane; i < NS; i += 32) Kd[s * NS + i] += 2.0 * hh * K[s * NS + i];
  }
  {  // F's tangent at N (its space held n12 so far)
    const double g = ntot / k.nd, gd = g * (td / ntot - dlnd);
    for (int i = lane; i < NS; i += 32) Fd[i] = sv1N[i] * g + svN[i] * gd;
  }
  for (int i = lane; i < NS; i += 32) Kd1[i] += Fd[i];
  __syncwarp();
  solve_attempt<DN>(work, Wd, c, c0, Kd1, tab.smap, lane, myt);
  for (int i = lane; i < NS; i += 32) Kd2[i] += Fd[i] + RODAS_C21 * ih * Kd1[i] - RODAS_C21 * hh * K1[i];
  __syncwarp();
  solve_attempt<DN>(work, Wd, c, c0, Kd2, tab.smap, lane, myt);
  for (int i = lane; i < NS; i += 32) {
    Ys[i] = N[i] + 2.0 * K1[i];
    Yds[i] = Nd[i] + 2.0 * Kd1[i];
  }
  __syncwarp();
  rhs_dual(work, Ys, Yds, Fd, va, k, dk, dlnd, tdx, tab, st, lane);
  for (int i = lane; i < NS; i += 32) Kd3[i] += Fd[i] + ih * (Kd1[i] - Kd2[i]) - hh * (K1[i] - K2[i]);
  __syncwarp();
  solve_attempt<DN>(work, Wd, c, c0, Kd3, tab.smap, lane, myt);
  for (int i = lane; i < NS; i += 32) {
    Ys[i] += K3[i];
    Yds[i] += Kd3[i];
  }
  __syncwarp();
  rhs_dual(work, Ys, Yds, Fd, va, k, dk, dlnd, tdx, tab, st, lane);
  for (int i = lane; i < NS; i += 32)
    Kd4[i] += Fd[i] + ih * (Kd1[i] - Kd2[i] - RODAS_C43 * Kd3[i]) - hh * (K1[i] - K2[i] - RODAS_C43 * K3[i]);
  __syncwarp();
  solve_attempt<DN>(work, Wd, c, c0, Kd4, tab.smap, lane, myt);
  for (int i = lane; i < NS; i += 32) Yd[i] = Nd[i] + 2.0 * Kd1[i] + Kd3[i] + Kd4[i];
  __syncwarp();
  if (!adapt || !(a.en > 0.0)) return 0.0;
  return error_tangent(N, Nd, Y, Yd, K4, Kd4, a.en, first, tol.rtol, tol.atol, lane);
}

#define JVP_ARGS                                                                                                       \
  const double *__restrict__ x0, const double *__restrict__ Ta, const double *__restrict__ Pa,                         \
      const double *__restrict__ dta, const double *__restrict__ hin, const double *lm, const double *__restrict__ rc, \
      const double *__restrict__ dx0, const double *__restrict__ dTa, const double *__restrict__ dPa,                  \
      const double *__restrict__ ddta, const double *__restrict__ dhin, const double *dlm, const double *drc, int D,   \
      double *__restrict__ xout, double *__restrict__ hout, int *__restrict__ doneo, double *__restrict__ noko,        \
      int *__restrict__ status, double *__restrict__ dxout, double *__restrict__ dhout, int *__restrict__ count,       \
      int B, double rtol, double atol, int max_steps, int adapt, const int *__restrict__ perm,                         \
      const unsigned *__restrict__ blob, const int *__restrict__ rf, const double *__restrict__ nasa, double *scratch, \
      const unsigned *__restrict__ sst, const u64 *__restrict__ jt, const u64 *__restrict__ lu,                        \
      const int *__restrict__ optr, int sround, int jround, const double *__restrict__ Jin, const double *dJin
#define JVP_PASS                                                                                                       \
  x0, Ta, Pa, dta, hin, lm, rc, dx0, dTa, dPa, ddta, dhin, dlm, drc, D, xout, hout, doneo, noko, status, dxout, dhout, \
      count, B, rtol, atol, max_steps, adapt, perm, blob, rf, nasa, scratch, sst, jt, lu, optr, sround, jround, Jin,   \
      dJin

// The call and its tangents, one warp per cell. D tangent directions share the call: direction d's inputs and outputs
// at [d * B + cell] (dx0, dxout, dlm, dJin by rows of NS, NR or NJ), drc at [d * NR * 9]; dlm, drc, dJin may be null;
// Jin [B, NJ] (null when NJ = 0). DN = false: every cell, the call kernel's attempts (COUNT_QUEUE). DN = true: the
// first count[COUNT_RECOMPUTED] cells of list, the recover kernel's attempts (dense W in wscr, one NS x NS per warp;
// max_steps = the recovery's tries; COUNT_RECOVERY_QUEUE), each result then tested as the recover kernel tests it: redo
// = 1 and COUNT_RECOVERED, or redo = 2, COUNT_STILL_FAILED and NaN tangents.
// size-limit-ok: the entries' arguments; its parts share one cell's state and each direction's tangents
template <bool DN>
__device__ __forceinline__ void jvp_cells(JVP_ARGS, int* redo, const int* list, double* wscr,
                                          const u16* __restrict__ didx, const double* __restrict__ atoms) {
  extern __shared__ double smem[];
  unsigned* tb32 = (unsigned*)(smem + WPB * PERWA);
  const Tab tab = load_tab(tb32, blob);
  const int lane = (int)(threadIdx.x & 31), wib = (int)(threadIdx.x >> 5);
  const int myt = lane < NT ? lane : -1;
  double c[NT];
#pragma unroll
  for (int j = 0; j < NT; j++) c[j] = 0.0;
  double* work = smem + wib * PERWA;
  double *N = work + A_NN, *Nd = work + A_ND, *Y = work + A_Y, *Yd = work + A_YD;
  const size_t wid = (size_t)blockIdx.x * WPB + wib;
  double* kf = scratch + wid * (JVP_STRIDE + D * JVP_DIRECTION_STRIDE);
  // dk and Ng, per direction: d kf, d kr; the state tangent
  double *kr = kf + RATE_KR, *va = kf + JS_VA, *cw = kf + JS_CW, *dk = kf + JVP_STRIDE, *Ng = dk + 2 * NR * D;
  double* Wd = DN ? wscr + wid * NS * NS : nullptr;
  const Streams streams = {sst, jt, lu, optr, didx, nullptr, nullptr, sround, jround, 0};
  const Tolerance tol = {rtol, atol};
  if (lane == 0) kf[RATE_ZERO] = 0.0;       // the zero rate of the Jacobian's padding words
  double hd[DMAX], tdot[DMAX], dlnd[DMAX];  // per direction: h, t, ln nd tangents (warp-uniform)
  double pend[DMAX], phad[DMAX];            // the previous attempt's error and size tangents
  const int nc = DN ? count[COUNT_RECOMPUTED] : B;
  int q = (int)(blockIdx.x * WPB + wib);
  while (q < nc) {
    // the cell's setup, as vjp_cells' (a shared helper would take 19 arguments); test_kernel_copies.py holds the
    // kernels' copies to the same bits
    const int cell = DN ? list[q] : q;
    const double Tc = Ta[cell], dt = dta[cell], Pc = Pa[cell], nd = Pc / (KBOLTZ * Tc);
    double p = 0.0, pm = 0.0;
    for (int i = lane; i < NS; i += 32) {
      N[i] = x0[(size_t)cell * NS + perm[i]];
      screen_x(N[i], p, pm);
    }
    for (int d = 0; d < D; d++)
      for (int i = lane; i < NS; i += 32) Ng[d * NS + i] = dx0[((size_t)d * B + cell) * NS + perm[i]];
    screen_J(Jin, cell, lane, 32, p);
    const double* lmc = lm ? lm + (size_t)cell * NR : nullptr;
    double pl = 0.0;  // a non-finite log multiplier: not stepped, but not refused (the cell fails, flag 2)
    screen_lm(lmc, lane, pl);
    const int st = cell_status(cell_fine(p, pm, Tc, nd, dt), Tc);
    if (!DN && lane == 0) status[cell] = st;
    const bool run = st == 0 && isfinite(wsum(pl));
    if (run) rates(work, kf, kr, cw, Tc, nd, lmc, Jin + (size_t)cell * NJ, rc, rf, nasa, tab, lane);
    const bool hfin = isfinite(hin[cell]);  // a non-finite h acts as dt (no sensitivity to it), and is counted
    if (!DN && !hfin && lane == 0) atomicAdd(count + COUNT_NONFINITE_H, 1);
    const double hi = hfin ? hin[cell] : dt, m = fmin(hi, dt);
    for (int d = 0; d < D; d++) {
      const size_t qd = (size_t)d * B + cell;
      const double dTc = dTa[qd], ddt = adapt ? ddta[qd] : 0.0, dhi = adapt ? dhin[qd] : 0.0;
      dlnd[d] = dPa[qd] / Pc - dTc / Tc;
      if (run)
        rates_tan(work, dk + 2 * NR * d, Tc, nd, dTc, dlnd[d], lmc, dlm ? dlm + qd * NR : nullptr,
                  dJin ? dJin + qd * NJ : nullptr, rc, drc ? drc + (size_t)d * NR * 9 : nullptr, rf, nasa, kf, kr, cw,
                  tab, lane);
      const double md = !hfin ? ddt : (hi < dt ? dhi : (hi > dt ? ddt : 0.5 * (dhi + ddt)));
      hd[d] = m > H_MIN ? md : (m < H_MIN ? 0.0 : 0.5 * md);
      tdot[d] = 0.0;
      pend[d] = 0.0;
      phad[d] = 0.0;
    }
    __syncwarp();
    const Rates coef = {kf, kr, nd};
    StepState step = {fmax(m, H_MIN), 0.0, 0.0, 0.0, 0.0, false};
    int it = (dt > 0.0 && run) ? 0 : max_steps;
    while (it < max_steps && step.t < dt * CALL_END) {
      const double rem = dt - step.t, ha = fmin(step.h, rem), h = step.h;
      const bool first = step.t == 0.0;
      double dfe = 0.0, dfh = 0.0, dfp = 0.0, dfq = 0.0;
      const Attempt a = run_attempt<DN>(work, c, Wd, ha, first, coef, tol, tab, streams, lane, myt);
      const bool ok = a.ok;
      const double en = a.en;
      const double fac = step_factor_partials(en, ok, step.prej, step.pen, step.pha, ha, dfe, dfh, dfp, dfq);
      // the error's tangent where this factor depends on it, or the next attempt's secant may (a rejection, err > 1)
      const bool wte = ok || (adapt && (dfe != 0.0 || (!ok && isfinite(en) && en > 1.0)));
      for (int d = 0; d < D; d++) {  // each direction's tangent on the attempt in place
        const double ddt = adapt ? ddta[(size_t)d * B + cell] : 0.0;
        const double had = !adapt ? 0.0 : (h < rem ? hd[d] : (h > rem ? ddt - tdot[d] : 0.5 * (hd[d] + ddt - tdot[d])));
        double end = 0.0;
        if (wte) {
          double* ng = Ng + d * NS;
          for (int i = lane; i < NS; i += 32) Nd[i] = ng[i];
          __syncwarp();
          end = tangent_step<DN>(work, c, Wd, a, ha, first, had, adapt, coef, dk + 2 * NR * d, dlnd[d], va, tol, tab,
                                 streams, lane, myt);
          if (ok)
            for (int i = lane; i < NS; i += 32) ng[i] = Y[i] >= 0.0 ? Yd[i] : 0.0;
          __syncwarp();
        }
        if (ok) tdot[d] += had;
        hd[d] = adapt ? had * fac + ha * (dfe * end + dfh * had + dfp * pend[d] + dfq * phad[d]) : 0.0;
        pend[d] = end;
        phad[d] = had;
      }
      accept_try(step, N, Y, ok, ha, lane, 32);
      __syncwarp();
      next_try(step, fac, ok, en, ha);
      it++;
    }
    const double s = write_state(N, xout + (size_t)cell * NS, perm, lane, lane, 32);
    for (int d = 0; d < D; d++) {
      const double* ng = Ng + d * NS;
      double pd = 0.0;
      for (int i = lane; i < NS; i += 32) pd += ng[i];
      const double sd = wsum(pd);
      for (int i = lane; i < NS; i += 32) dxout[((size_t)d * B + cell) * NS + perm[i]] = (ng[i] - N[i] / s * sd) / s;
      if (lane == 0) dhout[(size_t)d * B + cell] = hd[d];
    }
    const int dn = (run && step.t >= dt * CALL_END) ? 1 : 0;
    if (lane == 0) {
      hout[cell] = step.h;
      doneo[cell] = dn;
      noko[cell] = step.nok;
    }
    if (DN && recovery_fails(x0, xout, doneo, redo, count, dn, step.nok, atoms, atol, cell, lane))
      for (int d = 0; d < D; d++) {
        for (int i = lane; i < NS; i += 32) dxout[((size_t)d * B + cell) * NS + i] = DNAN;
        if (lane == 0) dhout[(size_t)d * B + cell] = DNAN;
      }
    q = next_queued<DN>(count, lane);
  }
}

// size-limit-ok: a kernel entry: Python and C++ bind each argument by name from _constants.KERNEL_ARGUMENTS
extern "C" __global__ void __launch_bounds__(32 * WPB) jvp(JVP_ARGS) {
  jvp_cells<false>(JVP_PASS, nullptr, nullptr, nullptr, nullptr, nullptr);
}

// After jvp and the guard: the listed cells redone on the recovery's path with their tangents (redo, list: the
// guard's).
// size-limit-ok: a kernel entry: Python and C++ bind each argument by name from _constants.KERNEL_ARGUMENTS
extern "C" __global__ void __launch_bounds__(32 * WPB)
    jvp_rec(JVP_ARGS, int* __restrict__ redo, const int* __restrict__ list, double* __restrict__ wscr,
            const u16* __restrict__ didx, const double* __restrict__ atoms) {
  if ((int)(blockIdx.x * WPB) >= count[COUNT_RECOMPUTED]) return;  // no listed cell for this block
  jvp_cells<true>(JVP_PASS, redo, list, wscr, didx, atoms);
}
