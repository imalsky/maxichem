// The recovery kernel: recomputes the flagged cells with a dense pivoted factorization.

// Dense LU of A (row-major NS x NS) on one warp with threshold partial pivoting, in place: row k is exchanged with the
// row of the largest |entry| of column k (ties to the lower row; a NaN entry is never chosen) only when |W_kk|
// is below PIVTAU times that entry or not finite; unit L below the diagonal, U on and above it; pr[i] = the row of the
// original A now in row i; dinv = 1 / U_ii.
__device__ __forceinline__ void dense_lu(double* A, double* dinv, int* pr, int lane) {
  for (int i = lane; i < NS; i += 32) pr[i] = i;
  __syncwarp();
  for (int k = 0; k < NS; k++) {
    double best = -1.0;
    int bi = NS;
    for (int i = k + lane; i < NS; i += 32) {
      const double a = fabs(A[i * NS + k]);
      if (a > best) {
        best = a;
        bi = i;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const double ob = __shfl_xor_sync(FULL, best, o);
      const int oi = __shfl_xor_sync(FULL, bi, o);
      if (ob > best || (ob == best && oi < bi)) {
        best = ob;
        bi = oi;
      }
    }
    if (bi != NS && bi != k && !(fabs(A[k * NS + k]) >= PIVTAU * best)) {
      for (int j = lane; j < NS; j += 32) {
        const double a = A[k * NS + j];
        A[k * NS + j] = A[bi * NS + j];
        A[bi * NS + j] = a;
      }
      if (lane == 0) {
        const int t = pr[k];
        pr[k] = pr[bi];
        pr[bi] = t;
      }
    }
    __syncwarp();
    const double dk = 1.0 / A[k * NS + k];
    if (lane == 0) dinv[k] = dk;
    for (int i = k + 1 + lane; i < NS; i += 32) {
      const double l = A[i * NS + k] * dk;
      A[i * NS + k] = l;
      for (int j = k + 1; j < NS; j++) A[i * NS + j] -= l * A[k * NS + j];
    }
    __syncwarp();
  }
}

// y <- A^-1 y on one warp with the factors of dense_lu. Lane l holds rows l + 32 p.
__device__ __forceinline__ void dense_solve(const double* A, const double* dinv, const int* pr, double* y, int lane) {
  double a[RPD];
#pragma unroll
  for (int p = 0; p < RPD; p++) a[p] = lane + 32 * p < NS ? y[pr[lane + 32 * p]] : 0.0;
#pragma unroll
  for (int k = 0; k < NS - 1; k++) {  // unit L
    const double yk = __shfl_sync(FULL, a[k >> 5], k & 31);
#pragma unroll
    for (int p = k >> 5; p < RPD; p++) {
      const int i = lane + 32 * p;
      if (i > k && i < NS) a[p] -= A[i * NS + k] * yk;
    }
  }
#pragma unroll
  for (int k = NS - 1; k > 0; k--) {  // U
    const double zk = __shfl_sync(FULL, a[k >> 5] * dinv[k], k & 31);
#pragma unroll
    for (int p = 0; p <= (k >> 5); p++)
      if (lane + 32 * p < k) a[p] -= A[(lane + 32 * p) * NS + k] * zk;
  }
  __syncwarp();
#pragma unroll
  for (int p = 0; p < RPD; p++)
    if (lane + 32 * p < NS) y[lane + 32 * p] = a[p] * dinv[lane + 32 * p];
  __syncwarp();
}

// Recomputes the listed cells (one cell per block of NL threads; persistent blocks over the first
// count[COUNT_RECOMPUTED] entries of list) from their inputs with max_steps tries: the call kernel's rates, S v and
// Jacobian streams, stages and step control, with the dense W = I / (gamma h) - Jn - w 1^T factored by dense_lu on the
// first warp; writes the result over the call's (xout, hio, doneo, noko), sets redo to 1 (the result passes the guard's
// test) or 2 and counts it in count[7] / count[8]. Returns at once when the list is empty.
extern "C" __global__ void __launch_bounds__(NL, 1)
    // size-limit-ok: inline helpers could move which operations fuse, so a split must first prove equal bits
    recover(const double* __restrict__ x0, const double* __restrict__ Ta, const double* __restrict__ Pa,
            const double* __restrict__ dta, const double* __restrict__ hin, double* xout, double* hio, int* doneo,
            double* noko, int* redo, const int* __restrict__ list, int* count, double rtol, double atol, int max_steps,
            const int* __restrict__ perm, const unsigned* __restrict__ blob, const double* __restrict__ rc,
            const int* __restrict__ rf, const double* __restrict__ nasa, double* scratch,
            const unsigned* __restrict__ sst, const u64* __restrict__ jt, const u16* __restrict__ didx,
            const double* __restrict__ atoms, int sround, int jround J_PARAM) {
  const int nfl = count[COUNT_RECOMPUTED];
  if ((int)blockIdx.x >= nfl) return;
  extern __shared__ double smem[];
  unsigned* tb32 = (unsigned*)(smem + PERWR);
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
  const int cl = threadIdx.x, lane = cl & 31;
  double* kf = scratch + (size_t)blockIdx.x * (2 * NR + 2 + (WSH ? 0 : NS * NS) + RSN);
  double* kr = kf + NR;
  double* Wd = WSH ? smem : kf + 2 * NR + 2;            // the dense matrix W
  double* rsc = kf + 2 * NR + 2 + (WSH ? 0 : NS * NS);  // F, u, -Jn's slots of the last fresh evaluation
  double *dinv = smem + (WSH ? NS * NS : 0), *n = dinv + NS, *v = n + NS + 1, *N = v + VR, *Y = N + NS, *F = Y + NS,
         *u = F + NS, *K1 = u + NS, *K2 = K1 + NS, *K3 = K2 + NS, *K4 = K3 + NS, *part = K4 + NS;
  int* pr = (int*)(part + NPART);
  int* nextq = pr + NS;
  if (cl == 0) kf[2 * NR] = 0.0;  // the zero rate of the Jacobian's padding words
  int q = blockIdx.x;
  while (q < nfl) {
    const int cell = list[q];
    const double Tc = Ta[cell], dt = dta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
    for (int i = cl; i < NS; i += NL) N[i] = x0[(size_t)cell * NS + perm[i]];
    CSYNC(0);
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
    if (fin && !cold) {  // rate coefficients, as the call kernel
      const double lT = log(Tc), iT = 1.0 / Tc, lKT = log(KBOLTZ * Tc / P0BAR);
      double* g = v;
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
      CSYNC(0);
      for (int r = cl; r < NR; r += NL) {
#if NJ
        if (r >= NRT) {
          kf[r] = Jin[(size_t)cell * NJ + r - NRT];
          kr[r] = 0.0;
          continue;
        }
#endif
        const int fl = rf[r];
        const double k = law(rc + RCW * r, fl, Tc, lT, iT, nd);
        const u8* x = tab.rx + 6 * r;
        const double dg = g[x[3]] + g[x[4]] + g[x[5]] - g[x[0]] - g[x[1]] - g[x[2]];
        const double c3 = (fl & FL_THREE_BODY) ? nd : 1.0;
        kf[r] = k * c3;
        kr[r] = REVERSE_RATE;
        COLL(fl, kf[r], kr[r], nd);
      }
    }
    CSYNC(0);
    const double h0 = isfinite(hin[cell]) ? hin[cell] : dt;  // a non-finite h acts as dt
    double h = fmax(fmin(h0, dt), H_MIN), t = 0.0, nok = 0.0, pen = 0.0, pha = 0.0;
    bool prej = false;
    int it = (dt > 0.0 && fin && !cold) ? 0 : max_steps;
    while (it < max_steps && t < dt * CALL_END) {
      const double ha = fmin(h, dt - t), igh = 1.0 / (GAMMA * ha), ih = 1.0 / ha;
      if (prej) {  // the rejected try's state: its F, u and -Jn (every write of the
        for (int s = cl; s < NS * NS; s += NL) Wd[s] = 0.0;  // Jacobian goes to a slot's dense position)
        for (int i = cl; i < NS; i += NL) {
          F[i] = rsc[i];
          u[i] = rsc[NS + i];
        }
        CSYNC(0);
        for (int s = cl; s < NNZ; s += NL) Wd[__ldg(didx + s)] = rsc[2 * NS + s];
      } else {
        const double ntot = rhs(N, F, u, n, v, part, kf, kr, nd, tab, sst, sround, lane, cl, 0);
        for (int s = cl; s < NS * NS; s += NL) Wd[s] = 0.0;
        CSYNC(0);
        {  // W = -Jn: the call kernel's term streams, written to the dense positions of the slots
          const unsigned* jw = (const unsigned*)jt;
          const u16* jsl = (const u16*)(jw + jround * NL) + cl * JSMAX;
          double acc = 0.0;
          int ne = 0;
#pragma unroll 4
          for (int qq = 0; qq < jround; qq++) {
            const unsigned word = __ldg(jw + qq * NL + cl);
            const double pq =
                kf[word & J_RATE_MASK] * n[(word >> J_A) & J_SPECIES_MASK] * n[(word >> J_B) & J_SPECIES_MASK];
            const double cf = ((word >> J_DOUBLE) & 1u) ? (((word >> J_SIGN) & 1u) ? -2.0 : 2.0)
                                                        : (((word >> J_SIGN) & 1u) ? -1.0 : 1.0);
            acc = fma(cf, pq, acc);
            if ((word >> J_END) & 1u) {
              const unsigned d = __ldg(jsl + ne);
              if (d < NNZ)
                Wd[__ldg(didx + d)] = -acc;
              else
                Y[d - NNZ] = -acc;
              acc = 0.0;
              ne++;
            }
          }
        }
        CSYNC(0);
#if NFIX
        if (cl == 0) {  // the extra pieces of long slots, in order
          const u16* fx = (const u16*)((const unsigned*)jt + jround * NL) + NL * JSMAX;
          for (int e = 0; e < NFIX; e++) Wd[__ldg(didx + __ldg(fx + e))] += Y[e];
        }
        CSYNC(0);
#endif
        for (int i = cl; i < NS; i += NL) {  // u = w = S v / nd - Jn x (the call kernel's sums, columns in order)
          double acc = 0.0;
          for (int j = 0; j < NS; j++)
            if (tab.smap[i * NS + j] != NO_SLOT) acc += Wd[i * NS + j] * N[j];
          u[i] = u[i] + acc / ntot;
        }
        CSYNC(0);
        for (int i = cl; i < NS; i += NL) {
          rsc[i] = F[i];
          rsc[NS + i] = u[i];
        }
        for (int s = cl; s < NNZ; s += NL) rsc[2 * NS + s] = Wd[__ldg(didx + s)];
      }
      CSYNC(0);
      for (int s = cl; s < NS * NS; s += NL) {  // W = I / (gamma h) - Jn - w 1^T
        const int i = s / NS;
        Wd[s] = (s == i * (NS + 1) ? Wd[s] + igh : Wd[s]) - u[i];
      }
      for (int i = cl; i < NS; i += NL) K1[i] = F[i];
      CSYNC(0);
      if (cl < 32) {
        dense_lu(Wd, dinv, pr, lane);
        // Rodas3 stages (gamma = 1/2), as the call kernel
        dense_solve(Wd, dinv, pr, K1, lane);
        for (int i = lane; i < NS; i += 32) K2[i] = fma(RODAS_C21 * ih, K1[i], F[i]);
        __syncwarp();
        dense_solve(Wd, dinv, pr, K2, lane);
        for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i];
      }
      CSYNC(0);
      rhs(Y, F, nullptr, n, v, part, kf, kr, nd, tab, sst, sround, lane, cl, 0);
      if (cl < 32) {
        for (int i = lane; i < NS; i += 32) K3[i] = fma(ih, K1[i] - K2[i], F[i]);
        __syncwarp();
        dense_solve(Wd, dinv, pr, K3, lane);
        for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i] + K3[i];
      }
      CSYNC(0);
      rhs(Y, F, nullptr, n, v, part, kf, kr, nd, tab, sst, sround, lane, cl, 0);
      if (cl < 32) {
        for (int i = lane; i < NS; i += 32) K4[i] = fma(ih, fma(-RODAS_C43, K3[i], K1[i] - K2[i]), F[i]);
        __syncwarp();
        dense_solve(Wd, dinv, pr, K4, lane);
      }
      CSYNC(0);
      // step control (docs/method.md); every warp reaches the same decision
      double e2 = 0.0, em = 0.0;
      bool nn = true;
      for (int i = lane; i < NS; i += 32) {
        const double y = N[i] + 2.0 * K1[i] + K3[i] + K4[i];
        if (cl < 32) Y[i] = y;
        const double sc = atol + rtol * fmax(fabs(N[i]), fabs(y));
        const double q4 = K4[i] / sc;
        e2 += q4 * q4;
        em = fmax(em, fabs(q4));
        nn = nn && (y >= -atol);
      }
      const double er = sqrt(wsum(e2) / NS), emx = wmax(em);
      const double en = (t == 0.0 && isfinite(er)) ? fmax(er, emx) : er;  // the worst species until a step is accepted
      const bool nonneg = __all_sync(FULL, nn), efin = isfinite(en), ok = (en <= 1.0) && nonneg && efin;
      CSYNC(0);
      if (ok) {
        for (int i = cl; i < NS; i += NL) N[i] = fmax(Y[i], 0.0);
        t += ha;
        nok += 1.0;
      }
      CSYNC(0);
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
    const int dn = (fin && t >= dt * CALL_END) ? 1 : 0;
    CSYNC(0);
    if (cl == 0) {
      const int c = check_cell(x0, Ta, Pa, dta, xout, dn, nok, atoms, atol, cell J_ARG);
      hio[cell] = h;
      doneo[cell] = c ? 0 : dn;
      noko[cell] = nok;  // done: finished and passing the guard
      redo[cell] = c ? 2 : 1;
      atomicAdd(count + (c ? COUNT_STILL_FAILED : COUNT_RECOVERED), 1);
      *nextq = gridDim.x + atomicAdd(count + COUNT_RECOVERY_QUEUE, 1);
    }
    CSYNC(0);
    q = *nextq;
    CSYNC(0);
  }
}
