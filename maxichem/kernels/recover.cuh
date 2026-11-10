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
    // every lane reads A[k][k] before lane k & 31 can overwrite it in the exchange
    const bool exchange = bi != NS && bi != k && !(fabs(A[k * NS + k]) >= PIVTAU * best);
    __syncwarp();
    if (exchange) {
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

// W = -Jn on lane cl of the Jacobian's stream (jacobian_lane), written to the dense positions of the slots.
__device__ __forceinline__ void jacobian_lane_dense(int cl, double* Wd, double* Y, const u16* __restrict__ didx,
                                                    const double* kf, const double* n, const u64* __restrict__ jt,
                                                    int jround) {
  const unsigned* jw = (const unsigned*)jt;
  const u16* jsl = (const u16*)(jw + jround * NL) + cl * JSMAX;
  double acc = 0.0;
  int ne = 0;
#pragma unroll 4
  for (int qq = 0; qq < jround; qq++) {
    const unsigned word = __ldg(jw + qq * NL + cl);
    const double pq = kf[word & J_RATE_MASK] * n[(word >> J_A) & J_SPECIES_MASK] * n[(word >> J_B) & J_SPECIES_MASK];
    acc = fma(stream_coefficient(word, J_DOUBLE, J_SIGN), pq, acc);
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

// The extra pieces of long slots, in order, on one thread.
__device__ __forceinline__ void jacobian_extras_dense(double* Wd, const double* Y, const u16* __restrict__ didx,
                                                      const u64* __restrict__ jt, int jround) {
  const u16* fx = (const u16*)((const unsigned*)jt + jround * NL) + NL * JSMAX;
  for (int e = 0; e < NFIX; e++) Wd[__ldg(didx + __ldg(fx + e))] += Y[e];
}

// Row i of the sum in w = S v / nd - Jn x from the dense -Jn: the sparse sums' terms, columns in order.
__device__ __forceinline__ double dense_times_x(int i, const double* Wd, const double* N, const Tab& tab) {
  double acc = 0.0;
  for (int j = 0; j < NS; j++)
    if (tab.smap[i * NS + j] != NO_SLOT) acc += Wd[i * NS + j] * N[j];
  return acc;
}

// W = I / (gamma h) - Jn - w 1^T, from -Jn in Wd, over the entries first, first + stride, ...
__device__ __forceinline__ void form_dense_w(double* Wd, const double* u, double igh, int first, int stride) {
  for (int s = first; s < NS * NS; s += stride) {
    const int i = s / NS;
    Wd[s] = (s == i * (NS + 1) ? Wd[s] + igh : Wd[s]) - u[i];
  }
}

// The recovery's work arrays for a cell: M the dense W, dinv 1 / its pivots, n the concentrations, N the state, Y a
// stage state, F the right-hand side, u the Sherman-Morrison vector, K1..K4 the stages, v the rates of progress, part
// the S v pieces, kf and kr the rate coefficients, rsc the copy kept for a retry.
struct CellArrays {
  double *M, *dinv, *n, *N, *Y, *F, *u, *K1, *K2, *K3, *K4, *v, *part, *kf, *kr, *rsc;
};

// The arrays of the recovery's one cell in shared memory, and its slot of the global scratch; M is the dense W
// (_tables.recovery_layout, scratch_layout).
__device__ __forceinline__ CellArrays recover_arrays(double* smem, double* scratch, size_t slot) {
  CellArrays a;
  a.kf = scratch + slot * RECOVER_STRIDE;
  a.kr = a.kf + RATE_KR;
  a.M = WSH ? smem + R_W : a.kf + SR_W;
  a.rsc = a.kf + SR_RSC;  // F, u, -Jn's slots of the last fresh evaluation
  a.dinv = smem + R_DINV;
  a.n = smem + R_N0;
  a.v = smem + R_V;
  a.N = smem + R_NN;
  a.Y = smem + R_Y;
  a.F = smem + R_F;
  a.u = smem + R_U;
  a.K1 = smem + R_K1;
  a.K2 = smem + R_K2;
  a.K3 = smem + R_K3;
  a.K4 = smem + R_K4;
  a.part = smem + R_PART;
  return a;
}

// A rejected try's retry: its F, u and -Jn (every write of the Jacobian goes to a slot's dense position).
__device__ __forceinline__ void restore_dense_try(const CellArrays& a, const u16* __restrict__ didx, int cl) {
  for (int s = cl; s < NS * NS; s += NL) a.M[s] = 0.0;
  for (int i = cl; i < NS; i += NL) {
    a.F[i] = a.rsc[i];
    a.u[i] = a.rsc[NS + i];
  }
  CSYNC(0);
  for (int s = cl; s < NNZ; s += NL) a.M[__ldg(didx + s)] = a.rsc[2 * NS + s];
}

// A fresh try: F, u = w = S v / nd - Jn x and W = -Jn at N, and their copy for a retry.
__device__ __forceinline__ void fresh_dense_try(const CellArrays& a, double nd, const Tab& tab,
                                                const unsigned* __restrict__ sst, const u64* __restrict__ jt,
                                                const u16* __restrict__ didx, int sround, int jround, int lane,
                                                int cl) {
  const double ntot = rhs(a.N, a.F, a.u, a.n, a.v, a.part, a.kf, a.kr, nd, tab, sst, sround, lane, cl, 0);
  for (int s = cl; s < NS * NS; s += NL) a.M[s] = 0.0;
  CSYNC(0);
  jacobian_lane_dense(cl, a.M, a.Y, didx, a.kf, a.n, jt, jround);
  CSYNC(0);
#if NFIX
  if (cl == 0) jacobian_extras_dense(a.M, a.Y, didx, jt, jround);
  CSYNC(0);
#endif
  for (int i = cl; i < NS; i += NL) a.u[i] = a.u[i] + dense_times_x(i, a.M, a.N, tab) / ntot;
  CSYNC(0);
  for (int i = cl; i < NS; i += NL) {
    a.rsc[i] = a.F[i];
    a.rsc[NS + i] = a.u[i];
  }
  for (int s = cl; s < NNZ; s += NL) a.rsc[2 * NS + s] = a.M[__ldg(didx + s)];
}

// The factorization and Rodas3 stages (gamma = 1/2), as the call kernel's, on the first warp.
__device__ __forceinline__ void dense_stages(const CellArrays& a, int* pr, double ih, double nd, const Tab& tab,
                                             const unsigned* __restrict__ sst, int sround, int lane, int cl) {
  if (cl < 32) {
    dense_lu(a.M, a.dinv, pr, lane);
    dense_solve(a.M, a.dinv, pr, a.K1, lane);
    for (int i = lane; i < NS; i += 32) a.K2[i] = stage2_rhs(ih, a.K1[i], a.F[i]);
    __syncwarp();
    dense_solve(a.M, a.dinv, pr, a.K2, lane);
    for (int i = lane; i < NS; i += 32) a.Y[i] = a.N[i] + 2.0 * a.K1[i];
  }
  CSYNC(0);
  rhs(a.Y, a.F, nullptr, a.n, a.v, a.part, a.kf, a.kr, nd, tab, sst, sround, lane, cl, 0);
  if (cl < 32) {
    for (int i = lane; i < NS; i += 32) a.K3[i] = stage3_rhs(ih, a.K1[i], a.K2[i], a.F[i]);
    __syncwarp();
    dense_solve(a.M, a.dinv, pr, a.K3, lane);
    for (int i = lane; i < NS; i += 32) a.Y[i] = a.N[i] + 2.0 * a.K1[i] + a.K3[i];
  }
  CSYNC(0);
  rhs(a.Y, a.F, nullptr, a.n, a.v, a.part, a.kf, a.kr, nd, tab, sst, sround, lane, cl, 0);
  if (cl < 32) {
    for (int i = lane; i < NS; i += 32) a.K4[i] = stage4_rhs(ih, a.K1[i], a.K2[i], a.K3[i], a.F[i]);
    __syncwarp();
    dense_solve(a.M, a.dinv, pr, a.K4, lane);
  }
  CSYNC(0);
}

// Recomputes the listed cells (one cell per block of NL threads; persistent blocks over the first
// count[COUNT_RECOMPUTED] entries of list) from their inputs with max_steps tries: the call kernel's rates, S v and
// Jacobian streams, stages and step control, with the dense W = I / (gamma h) - Jn - w 1^T factored by dense_lu on the
// first warp; writes the result over the call's (xout, hio, doneo, noko), sets redo to 1 (the result passes the guard's
// test) or 2 and counts it in COUNT_RECOVERED or COUNT_STILL_FAILED. Returns at once when the list is empty.
// size-limit-ok: a kernel entry: Python and C++ bind each argument by name from _constants.KERNEL_ARGUMENTS
extern "C" __global__ void __launch_bounds__(NL, 1)
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
  const Tab tab = load_tab(tb32, blob);
  const int cl = (int)threadIdx.x, lane = cl & 31;
  const CellArrays a = recover_arrays(smem, scratch, (size_t)blockIdx.x);
  double* N = a.N;
  int* pr = (int*)(smem + R_PR);
  int* nextq = pr + NS;
  if (cl == 0) a.kf[RATE_ZERO] = 0.0;  // the zero rate of the Jacobian's padding words
  int q = (int)blockIdx.x;
  while (q < nfl) {
    const int cell = list[q];
    const double Tc = Ta[cell], dt = dta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
    for (int i = cl; i < NS; i += NL) N[i] = x0[(size_t)cell * NS + perm[i]];
    CSYNC(0);
    // a listed cell passed the call kernel's screening (status 0), so it is not screened again
    cell_rates(a.v, a.kf, a.kr, Tc, nd, rc, rf, nasa, tab, J_OR_NULL, cell, cl, 0);
    CSYNC(0);
    const double h0 = isfinite(hin[cell]) ? hin[cell] : dt;  // a non-finite h acts as dt (the call kernel counted it)
    StepState s = {fmax(fmin(h0, dt), H_MIN), 0.0, 0.0, 0.0, 0.0, false};
    int it = dt > 0.0 ? 0 : max_steps;
    while (it < max_steps && s.t < dt * CALL_END) {
      const double ha = fmin(s.h, dt - s.t), igh = 1.0 / (GAMMA * ha), ih = 1.0 / ha;
      if (s.prej)
        restore_dense_try(a, didx, cl);
      else
        fresh_dense_try(a, nd, tab, sst, jt, didx, sround, jround, lane, cl);
      CSYNC(0);
      form_dense_w(a.M, a.u, igh, cl, NL);
      for (int i = cl; i < NS; i += NL) a.K1[i] = a.F[i];
      CSYNC(0);
      dense_stages(a, pr, ih, nd, tab, sst, sround, lane, cl);
      // step control (docs/method.md); every warp reaches the same decision
      bool ok = false;
      const double en = try_error(N, a.K1, a.K3, a.K4, a.Y, cl < 32, s.t == 0.0, rtol, atol, lane, ok);
      CSYNC(0);
      accept_try(s, N, a.Y, ok, ha, cl, NL);
      CSYNC(0);
      next_try(s, step_factor(en, ok, s.prej, s.pen, s.pha, ha), ok, en, ha);
      it++;
    }
    write_state(N, xout + (size_t)cell * NS, perm, lane, cl, NL);
    const int dn = (s.t >= dt * CALL_END) ? 1 : 0;
    CSYNC(0);
    if (cl == 0) {
      const int c = check_cell(x0, xout, dn, s.nok, atoms, atol, cell);
      hio[cell] = s.h;
      doneo[cell] = c ? 0 : dn;  // done: finished and passing the guard
      noko[cell] = s.nok;
      redo[cell] = c ? 2 : 1;
      atomicAdd(count + (c ? COUNT_STILL_FAILED : COUNT_RECOVERED), 1);
      *nextq = (int)(gridDim.x + atomicAdd(count + COUNT_RECOVERY_QUEUE, 1));
    }
    CSYNC(0);
    q = *nextq;
    CSYNC(0);
  }
}
