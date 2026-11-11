// The call kernel: every cell's rates, Rodas3 steps and step control in one launch.

// one block per SM is enough: without that bound ptxas may cap the registers below what the block size allows and spill
// size-limit-ok: arguments bound by name (KERNEL_ARGUMENTS); split, it cost 1.1 % (0.654 to 0.661 us with photolysis)
extern "C" __global__ void __launch_bounds__(NL* CPB, 1)
    call(const double* __restrict__ x0, const double* __restrict__ Ta, const double* __restrict__ Pa,
         const double* __restrict__ dta, double* __restrict__ hio, double* __restrict__ xout, int* __restrict__ doneo,
         double* __restrict__ noko, int* __restrict__ status, int* __restrict__ count, int B, double rtol, double atol,
         int max_steps, const int* __restrict__ perm, const unsigned* __restrict__ blob, const double* __restrict__ rc,
         const int* __restrict__ rf, const double* __restrict__ nasa, double* scratch, const unsigned* __restrict__ sst,
         const u64* __restrict__ jt, const u64* __restrict__ lu, const int* __restrict__ optr, int sround,
         int jround J_PARAM) {
  extern __shared__ double smem[];
  unsigned* tb32 = (unsigned*)(smem + CPB * PERC);
  const Tab tab = load_tab(tb32, blob);
  const int lane = (int)(threadIdx.x & 31), cl = (int)(threadIdx.x % NL), grp = (int)(threadIdx.x / NL);
  const bool w0 = cl < 32;                // the cell's first warp: trailing block, solves
  const int myt = lane < NT ? lane : -1;  // this lane's row K0 + myt of the dense trailing block
  double c[NT];                           // that row: M0, then its LU factors
#pragma unroll
  for (int j = 0; j < NT; j++) c[j] = 0.0;
  double* M = smem + grp * PERC + C_M;  // M0 by slot, then its LU factors (leading slots, then the block)
  double *dinv = M + C_DINV, *n = M + C_N0, *N = M + C_NN, *Y = M + C_Y, *F = M + C_F, *u = M + C_U, *K1 = M + C_K1,
         *K2 = M + C_K2, *K3 = M + C_K3, *K4 = M + C_K4;
  // v and part sit in the trailing block's slots, free while that block is in registers, if they fit there (C_V =
  // TB0), else after the other arrays (_tables.call_layout)
  double *v = M + C_V + CR_V, *part = M + C_V + CR_PART;
  int* nextc = (int*)(M + C_NEXT);
  double* kf = scratch + ((size_t)blockIdx.x * CPB + grp) * CALL_STRIDE;  // per cell slot: rates, the copy
  double* kr = kf + RATE_KR;
  double* rsc = kf + S_RSC;          // F, u, -Jn of the last fresh evaluation (a retry at the same state)
  if (cl == 0) kf[RATE_ZERO] = 0.0;  // the zero rate of the Jacobian's padding words
  int cell = (int)(blockIdx.x * CPB + grp);
  while (cell < B) {
    const double Tc = Ta[cell], dt = dta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
    for (int i = cl; i < NS; i += NL) N[i] = x0[(size_t)cell * NS + perm[i]];
    CSYNC(grp);
    double p = 0.0, pm = 0.0;
    for (int i = lane; i < NS; i += 32) screen_x(N[i], p, pm);
#if NJ
    screen_J(Jin, cell, lane, 32, p);
#endif
    const int st = cell_status(cell_fine(p, pm, Tc, nd, dt), Tc);
    if (cl == 0) status[cell] = st;
    if (st == 0) cell_rates(v, kf, kr, Tc, nd, rc, rf, nasa, tab, J_OR_NULL, cell, cl, grp);
    CSYNC(grp);
    const double h0 = isfinite(hio[cell]) ? hio[cell] : dt;  // a non-finite h acts as dt, and is counted
    if (cl == 0 && !isfinite(hio[cell])) atomicAdd(count + COUNT_NONFINITE_H, 1);
    StepState step = {fmax(fmin(h0, dt), H_MIN), 0.0, 0.0, 0.0, 0.0, false};
    int it = (dt > 0.0 && st == 0) ? 0 : max_steps;  // no step for a refused cell
    while (it < max_steps && step.t < dt * CALL_END) {
      const double ha = fmin(step.h, dt - step.t), igh = 1.0 / (GAMMA * ha), ih = 1.0 / ha;
      if (step.prej) {
        // the rejected try's state: its F, u and -Jn, all slots, since v and the S v pieces overwrote the trailing
        // block's
        for (int i = cl; i < NS; i += NL) {
          F[i] = rsc[RSC_F + i];
          u[i] = rsc[RSC_U + i];
        }
        for (int s = cl; s < NNZ; s += NL) M[s] = rsc[RSC_JN + s];
      } else {
        const double ntot = rhs(N, F, u, n, v, part, kf, kr, nd, tab, sst, sround, lane, cl, grp);
        for (int s = cl; s < NNZ; s += NL) M[s] = 0.0;
        CSYNC(grp);
        jacobian_lane(cl, M, Y, kf, n, jt, jround);
        CSYNC(grp);
#if NFIX
        if (cl == 0) jacobian_extras(M, Y, jt, jround);
        CSYNC(grp);
#endif
        // u = w = S v / nd - Jn x
        for (int i = cl; i < NS; i += NL) u[i] = u[i] + jacobian_times_x(i, M, N, tab) / ntot;
        CSYNC(grp);
        for (int i = cl; i < NS; i += NL) {
          rsc[RSC_F + i] = F[i];
          rsc[RSC_U + i] = u[i];
        }
        for (int s = cl; s < NNZ; s += NL) rsc[RSC_JN + s] = M[s];
      }
      CSYNC(grp);
      for (int i = cl; i < NS; i += NL) M[tab.diag[i]] += igh;  // M = M0 = I / (gamma h) - Jn
      CSYNC(grp);
      for (int k = 0; k < K0; k++) {  // LU of the sparse leading columns (updates reach the trailing block)
        const int len = tab.blen[k];
        if (len) {  // a batch of consecutive independent pivots: their 1 / diagonal
          for (int i = cl; i < len; i += NL) dinv[k + i] = 1.0 / M[tab.diag[k + i]];
          CSYNC(grp);
        }
        if (!(LPRE(k) | LTAIL(k))) continue;
        const int oa = optr[k], ob = optr[k + 1];
        u64 op = (oa + cl < ob) ? __ldg(lu + oa + cl) : 0ull;
        const double dk = dinv[k];
        scale_column(k, dk, M, tab, cl, NL);
        CSYNC(grp);
        for (int e = oa + cl; e < ob; e += NL) {  // each lane reads its next word ahead, before the slots it updates
          const u64 nx = (e + NL < ob) ? __ldg(lu + e + NL) : 0ull;
          lu_update(M, op);
          op = nx;
        }
        CSYNC(grp);
      }
      if (w0) {  // dense trailing block in registers; 1 / its diagonal into dinv
        if (myt >= 0) {
#pragma unroll
          for (int j = 0; j < NT; j++) c[j] = M[TB0 + myt * NT + j];
        }
        factor_trailing(c, dinv, myt);
        if (myt >= 0) {  // the factors back into the block's slots for the second warp
#pragma unroll
          for (int j = 0; j < NT; j++) M[TB0 + myt * NT + j] = c[j];
        }
      }
      for (int i = cl; i < NS; i += NL) K1[i] = F[i];
      CSYNC(grp);
      if (cl >= 32 && cl < 64 && myt >= 0) {  // the second warp's copy of the trailing factors (two-vector solve)
#pragma unroll
        for (int j = 0; j < NT; j++) c[j] = M[TB0 + myt * NT + j];
      }
      // Rodas3 stages (gamma = 1/2; docs/method.md) as one loop with one copy of the triangular solve: stage 1 solves
      // K1 on the first warp and u on the second, then c0 = 1 / (1 - sum M0^-1 u); every stage ends with the
      // Sherman-Morrison update on the first warp
      double c0 = 0.0;
#pragma unroll 1
      for (int st = 1; st <= 4; st++) {
        if (st > 2) {
          for (int i = cl; i < NS; i += NL) {
            double y = N[i] + 2.0 * K1[i];
            if (st == 4) y = y + K3[i];
            Y[i] = y;
          }
          CSYNC(grp);
          rhs(Y, F, nullptr, n, v, part, kf, kr, nd, tab, sst, sround, lane, cl, grp);
        }
        double* Ks = st == 1 ? K1 : (st == 2 ? K2 : (st == 3 ? K3 : K4));
        if (st > 1) {
          for (int i = cl; i < NS; i += NL)
            Ks[i] = st == 2
                        ? stage2_rhs(ih, K1[i], F[i])
                        : (st == 3 ? stage3_rhs(ih, K1[i], K2[i], F[i]) : stage4_rhs(ih, K1[i], K2[i], K3[i], F[i]));
          CSYNC(grp);
        }
        if (cl < (st == 1 ? 64 : 32)) {
          const bool uw = st == 1 && !w0;  // uw: this warp solves u
          double* yv = uw ? u : Ks;
          double a0[RP], a1 = 0.0;
          msolve(M, dinv, yv, tab.smap, c, lane, myt, a0, a1);
          if (uw) {
            const double cc = sherman_morrison_factor(a0, a1, lane);
            if (lane == 0) dinv[NS] = cc;
          }
          __syncwarp();
          store_solved(yv, a0, a1, lane);
        }
        CSYNC(grp);
        c0 = dinv[NS];
        if (w0) sherman_morrison_update(Ks, u, c0, lane);
        CSYNC(grp);
      }
      // step control (docs/method.md); every warp of the cell reaches the same decision
      bool ok = false;
      const double en = try_error(N, K1, K3, K4, Y, w0, step.t == 0.0, rtol, atol, lane, ok);
      CSYNC(grp);
      accept_try(step, N, Y, ok, ha, cl, NL);
      CSYNC(grp);
      next_try(step, step_factor(en, ok, step.prej, step.pen, step.pha, ha), ok, en, ha);
      it++;
    }
    write_state(N, xout + (size_t)cell * NS, perm, lane, cl, NL);
    if (cl == 0) {
      hio[cell] = step.h;
      doneo[cell] = (st == 0 && step.t >= dt * CALL_END) ? 1 : 0;  // a refused cell is not done, also at dt = 0
      noko[cell] = step.nok;
    }
    if (cl == 0) *nextc = (int)(gridDim.x * CPB + atomicAdd(count + COUNT_QUEUE, 1));
    CSYNC(grp);
    cell = *nextc;
  }
}
