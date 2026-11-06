// The call kernel: every cell's rates, Rodas3 steps and step control in one launch.

// one block per SM is enough: without that bound ptxas may cap the registers below what the block size allows and spill
extern "C" __global__ void __launch_bounds__(NL* CPB, 1)
    // size-limit-ok: inline helpers could move which operations fuse, so a split must first prove equal bits
    call(const double* __restrict__ x0, const double* __restrict__ Ta, const double* __restrict__ Pa,
         const double* __restrict__ dta, double* __restrict__ hio, double* __restrict__ xout, int* __restrict__ doneo,
         double* __restrict__ noko, int* __restrict__ count, int B, double rtol, double atol, int max_steps,
         const int* __restrict__ perm, const unsigned* __restrict__ blob, const double* __restrict__ rc,
         const int* __restrict__ rf, const double* __restrict__ nasa, double* scratch, const unsigned* __restrict__ sst,
         const u64* __restrict__ jt, const u64* __restrict__ lu, const int* __restrict__ optr, int sround,
         int jround J_PARAM) {
  extern __shared__ double smem[];
  unsigned* tb32 = (unsigned*)(smem + CPB * PERC);
  for (int i = threadIdx.x; i < TBYTES / 4; i += blockDim.x) tb32[i] = blob[i];
  __syncthreads();
  const u8* tb = (const u8*)tb32;
  Tab tab;
  tab.smap = (const u16*)(tb + O_SMAP);
  tab.rowptr = (const u16*)(tb + O_ROWPTR);
  tab.diag = (const u16*)(tb + O_DIAG);
  tab.colof = tb + O_COLOF;
  tab.rx = tb + O_RX;
  tab.xp = tb + O_XP;
  tab.blen = tb + O_BLEN;
  const int lane = threadIdx.x & 31, cl = threadIdx.x % NL, grp = threadIdx.x / NL;
  const bool w0 = cl < 32;                // the cell's first warp: trailing block, solves
  const int myt = lane < NT ? lane : -1;  // this lane's row K0 + myt of the dense trailing block
  double c[NT];                           // that row: M0, then its LU factors
#pragma unroll
  for (int j = 0; j < NT; j++) c[j] = 0.0;
  double* M = smem + grp * PERC;  // M0 by slot, then its LU factors (leading slots, then the block)
  double *dinv = M + NNZ, *n = dinv + NS + 1, *N = n + NS + 1, *Y = N + NS, *F = Y + NS, *u = F + NS, *K1 = u + NS,
         *K2 = K1 + NS, *K3 = K2 + NS, *K4 = K3 + NS;
  double *v = M + VOFF, *part = v + NR;  // in the trailing block's slots while that block is in registers if
                                         // they fit there (VOFF = TB0), else after the other arrays
  int* nextc = (int*)(K4 + NS);
  double* kf = scratch + (size_t)(blockIdx.x * CPB + grp) * (2 * NR + 2 + RSN);  // per cell slot: rates, the copy
  double* kr = kf + NR;
  double* rsc = kf + 2 * NR + 2;  // F, u, -Jn of the last fresh evaluation (a retry at the same state)
  if (cl == 0) kf[2 * NR] = 0.0;  // the zero rate of the Jacobian's padding words
  int cell = blockIdx.x * CPB + grp;
  while (cell < B) {
    const double Tc = Ta[cell], dt = dta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
    for (int i = cl; i < NS; i += NL) N[i] = x0[(size_t)cell * NS + perm[i]];
    CSYNC(grp);
    double p = 0.0, pm = 0.0;
    for (int i = lane; i < NS; i += 32) {
      p += N[i] >= 0.0 ? N[i] : DNAN;
      pm = fmax(pm, N[i]);
    }
#if NJ
    for (int b = lane; b < NJ; b += 32) {
      const double q = Jin[(size_t)cell * NJ + b];
      p += q >= 0.0 ? 0.0 * q : __longlong_as_double(0x7ff8000000000000ull);
    }
#endif
    const double ps = wsum(p);  // bad input (not stepped): non-finite, x < 0, no x > 0, P <= 0 or dt < 0
    const bool fin = isfinite(ps) && wmax(pm) > 0.0 && isfinite(Tc) && isfinite(nd) && nd > 0.0 && isfinite(dt) &&
                     dt >= 0.0,
               cold = Tc < TMIN;
    if (fin && !cold) {  // rate coefficients; three-body ones times nd
      const double lT = log(Tc), iT = 1.0 / Tc, lKT = log(KBOLTZ * Tc / P0BAR);
      double* g = v;  // G / RT of each species
      const double* cf = nasa + (Tc < NASA_SPLIT_T ? 0 : NS * 9);
      for (int i = cl; i < NS; i += NL) {
        const double* a = cf + 9 * i;
        const double T2 = Tc * Tc, T3 = T2 * Tc, T4 = T3 * Tc;
        const double hh = -a[0] * iT * iT + a[1] * lT * iT + a[2] + a[3] * Tc / 2 + a[4] * T2 / 3 + a[5] * T3 / 4 +
                          a[6] * T4 / 5 + a[7] * iT;
        const double ss = -a[0] * iT * iT / 2 - a[1] * iT + a[2] * lT + a[3] * Tc + a[4] * T2 / 2 + a[5] * T3 / 3 +
                          a[6] * T4 / 4 + a[8];
        g[i] = hh - ss;
      }
      if (cl == 0) g[NS] = 0.0;
      CSYNC(grp);
      for (int r = cl; r < NR; r += NL) {
#if NJ
        if (r >= NRT) {
          kf[r] = Jin[(size_t)cell * NJ + r - NRT];
          kr[r] = 0.0;
          continue;
        }
#endif
        const int fl = rf[r];  // law flags (law()): 4 three-body, (dmu + 4) << 3, 256 explicit reverse
        const double k = law(rc + RCW * r, fl, Tc, lT, iT, nd);
        const u8* x = tab.rx + 6 * r;
        const double dg = g[x[3]] + g[x[4]] + g[x[5]] - g[x[0]] - g[x[1]] - g[x[2]];
        const double c3 = (fl & FL_THREE_BODY) ? nd : 1.0;
        kf[r] = k * c3;
        kr[r] = REVERSE_RATE;
        COLL(fl, kf[r], kr[r], nd);
      }
    }
    CSYNC(grp);
    const double h0 = isfinite(hio[cell]) ? hio[cell] : dt;  // a non-finite h acts as dt
    double h = fmax(fmin(h0, dt), H_MIN), t = 0.0, nok = 0.0, pen = 0.0, pha = 0.0;
    bool prej = false;                                    // the last try was rejected (its err pen, its size pha)
    int it = (dt > 0.0 && fin && !cold) ? 0 : max_steps;  // no step for non-finite inputs or cold gas
    while (it < max_steps && t < dt * CALL_END) {
      const double ha = fmin(h, dt - t), igh = 1.0 / (GAMMA * ha), ih = 1.0 / ha;
      if (prej) {  // the rejected try's state: its F, u and -Jn (all slots: v and the
        for (int i = cl; i < NS; i += NL) {
          F[i] = rsc[i];
          u[i] = rsc[NS + i];
        }  // S v pieces overwrite the
        for (int s = cl; s < NNZ; s += NL) M[s] = rsc[2 * NS + s];  // trailing block's slots)
      } else {
        const double ntot = rhs(N, F, u, n, v, part, kf, kr, nd, tab, sst, sround, lane, cl, grp);
        for (int s = cl; s < NNZ; s += NL) M[s] = 0.0;
        CSYNC(grp);
        {  // M = -Jn: lane streams of 32-bit terms (rate index, sign, x2, species a, b, end of slot); slots from jsl
          const unsigned* jw = (const unsigned*)jt;
          const u16* jsl = (const u16*)(jw + jround * NL) + cl * JSMAX;
          double acc = 0.0;
          int ne = 0;
#pragma unroll 4
          for (int q = 0; q < jround; q++) {
            const unsigned word = __ldg(jw + q * NL + cl);  // kr = kf + NR
            const double pr =
                kf[word & J_RATE_MASK] * n[(word >> J_A) & J_SPECIES_MASK] * n[(word >> J_B) & J_SPECIES_MASK];
            const double cf = ((word >> J_DOUBLE) & 1u) ? (((word >> J_SIGN) & 1u) ? -2.0 : 2.0)
                                                        : (((word >> J_SIGN) & 1u) ? -1.0 : 1.0);
            acc = fma(cf, pr, acc);
#if NFIX
            if ((word >> J_END) & 1u) {
              const unsigned d = __ldg(jsl + ne);
              if (d < NNZ)
                M[d] = -acc;
              else
                Y[d - NNZ] = -acc;
              acc = 0.0;
              ne++;
            }
#else
            if ((word >> J_END) & 1u) {
              M[__ldg(jsl + ne)] = -acc;
              acc = 0.0;
              ne++;
            }
#endif
          }
        }
        CSYNC(grp);
#if NFIX
        if (cl == 0) {  // the extra pieces of long slots, in order
          const u16* fx = (const u16*)((const unsigned*)jt + jround * NL) + NL * JSMAX;
          for (int e = 0; e < NFIX; e++) M[__ldg(fx + e)] += Y[e];
        }
        CSYNC(grp);
#endif
        for (int i = cl; i < NS; i += NL) {  // u = w = S v / nd - Jn x
          double acc = 0.0;
          const int s1 = tab.rowptr[i + 1];
          for (int s = tab.rowptr[i]; s < s1; s++) acc += M[s] * N[tab.colof[s]];
          if (i >= K0)
            for (int j = 0; j < NT; j++) acc += M[TB0 + (i - K0) * NT + j] * N[K0 + j];
          u[i] = u[i] + acc / ntot;
        }
        CSYNC(grp);
        for (int i = cl; i < NS; i += NL) {
          rsc[i] = F[i];
          rsc[NS + i] = u[i];
        }
        for (int s = cl; s < NNZ; s += NL) rsc[2 * NS + s] = M[s];
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
        for (int i = cl; i < NS; i += NL) {
          const unsigned q = tab.smap[i * NS + k];
          if (i > k && q != NO_SLOT) M[q] *= dk;
        }
        CSYNC(grp);
        for (int e = oa + cl; e < ob; e += NL) {
          const u64 nx = (e + NL < ob) ? __ldg(lu + e + NL) : 0ull;
          M[op & LU_SLOT_MASK] -= M[(op >> LU_B) & LU_SLOT_MASK] * M[(op >> LU_C) & LU_SLOT_MASK];
          op = nx;
        }
        CSYNC(grp);
      }
      if (w0) {  // dense trailing block in registers; 1 / its diagonal into dinv
        if (myt >= 0) {
#pragma unroll
          for (int j = 0; j < NT; j++) c[j] = M[TB0 + myt * NT + j];
        }
#pragma unroll
        for (int k = 0; k < NT; k++) {
          const double rp = 1.0 / __shfl_sync(FULL, c[k], k);
          if (myt == k) dinv[K0 + k] = rp;
          const bool act = myt > k;
          const double l = act ? c[k] * rp : 0.0;
          if (act) c[k] = l;
#pragma unroll
          for (int j = k + 1; j < NT; j++) {
            const double uu = __shfl_sync(FULL, c[j], k);
            if (act) c[j] -= l * uu;
          }
        }
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
            Ks[i] = st == 2 ? fma(RODAS_C21 * ih, K1[i], F[i])
                            : (st == 3 ? fma(ih, K1[i] - K2[i], F[i])
                                       : fma(ih, fma(-RODAS_C43, K3[i], K1[i] - K2[i]), F[i]));
          CSYNC(grp);
        }
        if (cl < (st == 1 ? 64 : 32)) {
          const bool uw = st == 1 && !w0, ht = lane < NT;  // uw: this warp solves u
          double* yv = uw ? u : Ks;
          double a0[RP], a1;
          msolve(M, dinv, yv, tab.smap, c, lane, myt, a0, a1);
          if (uw) {
            double sb = lane < K0 ? a0[0] : 0.0;
#pragma unroll
            for (int p = 1; p < RP; p++) sb += lane + 32 * p < K0 ? a0[p] : 0.0;
            const double cc = 1.0 / (1.0 - wsum(sb + (ht ? a1 : 0.0)));
            if (lane == 0) dinv[NS] = cc;
          }
          __syncwarp();
#pragma unroll
          for (int p = 0; p < RP; p++)
            if (lane + 32 * p < K0) yv[lane + 32 * p] = a0[p];
          if (ht) yv[K0 + lane] = a1;
        }
        CSYNC(grp);
        c0 = dinv[NS];
        if (w0) {
          const bool ht = lane < NT;
          double a0[RP];
#pragma unroll
          for (int p = 0; p < RP; p++) a0[p] = lane + 32 * p < K0 ? Ks[lane + 32 * p] : 0.0;
          const double a1 = ht ? Ks[K0 + lane] : 0.0;
          double sa = a0[0];
#pragma unroll
          for (int p = 1; p < RP; p++) sa += a0[p];
          const double s = wsum(sa + a1) * c0;
          __syncwarp();
#pragma unroll
          for (int p = 0; p < RP; p++)
            if (lane + 32 * p < K0) Ks[lane + 32 * p] = a0[p] + s * u[lane + 32 * p];
          if (ht) Ks[K0 + lane] = a1 + s * u[K0 + lane];
        }
        CSYNC(grp);
      }
      // step control (docs/method.md); every warp of the cell reaches the same decision
      double e2 = 0.0, em = 0.0;
      bool nn = true;
      for (int i = lane; i < NS; i += 32) {
        const double y = N[i] + 2.0 * K1[i] + K3[i] + K4[i];
        if (w0) Y[i] = y;
        const double sc = atol + rtol * fmax(fabs(N[i]), fabs(y));
        const double q = K4[i] / sc;
        e2 += q * q;
        em = fmax(em, fabs(q));
        nn = nn && (y >= -atol);
      }
      const double er = sqrt(wsum(e2) / NS), emx = wmax(em);
      const double en = (t == 0.0 && isfinite(er)) ? fmax(er, emx) : er;  // the worst species until a step is accepted
      const bool nonneg = __all_sync(FULL, nn), efin = isfinite(en), ok = (en <= 1.0) && nonneg && efin;
      CSYNC(grp);
      if (ok) {
        for (int i = cl; i < NS; i += NL) N[i] = fmax(Y[i], 0.0);
        t += ha;
        nok += 1.0;
      }
      CSYNC(grp);
      h = ha * step_factor(en, ok, prej, pen, pha, ha);
      prej = !ok;
      pen = en;
      pha = ha;
      it++;
    }
    p = 0.0;
    for (int i = lane; i < NS; i += 32) p += N[i];
    const double s = wsum(p);
    for (int i = cl; i < NS; i += NL) xout[(size_t)cell * NS + perm[i]] = N[i] / s;
    if (cl == 0) {
      hio[cell] = h;
      doneo[cell] = (fin && t >= dt * CALL_END) ? 1 : 0;
      noko[cell] = nok;
    }
    if (cl == 0) *nextc = gridDim.x * CPB + atomicAdd(count + COUNT_QUEUE, 1);
    CSYNC(grp);
    cell = *nextc;
  }
}
