// The guard: checks each cell after the call, and puts its element shares back.

// What the guard's test found in a cell: a value for the first two, bits for the others, 0 when it passed.
constexpr int GUARD_COLD = 1;
constexpr int GUARD_BAD_INPUT = 2;
constexpr int GUARD_NONFINITE = 4;
constexpr int GUARD_UNFINISHED = 8;
constexpr int GUARD_DRIFT = 16;

// The guard's test of one cell (network species order): GUARD_COLD, GUARD_BAD_INPUT, else the bits GUARD_NONFINITE,
// GUARD_UNFINISHED, GUARD_DRIFT (0: passed). A cell that passes has its element shares put back
// (docs/method.md): x' = x (1 + sum_e lambda_e a_ie), with (A X A^T) lambda = e0 s1 / s0 - e1 by LDL^T; an
// element whose pivot vanishes (absent, or tied to the others) is left out. Explicit fma: x depends on these sums,
// and every kernel that calls this must get the same bits.
// size-limit-ok: inline helpers could move which operations fuse, so a split must first prove equal bits
__device__ int check_cell(const double* __restrict__ x0, const double* __restrict__ Ta, const double* __restrict__ Pa,
                          const double* __restrict__ dta, double* xout, int done, double nok,
                          const double* __restrict__ atoms, double atol, int cell J_PARAM) {
  double e0[NEL], e1[NEL], p = 0.0, mx = 0.0;  // atoms of each element in the input and the new state
#pragma unroll
  for (int e = 0; e < NEL; e++) {
    e0[e] = 0.0;
    e1[e] = 0.0;
  }
  bool xfin = true;
  for (int i = 0; i < NS; i++) {
    const double a = x0[(size_t)cell * NS + i], b = xout[(size_t)cell * NS + i];
    p += a >= 0.0 ? a : DNAN;
    mx = fmax(mx, a);
    xfin = xfin && isfinite(b);
#pragma unroll
    for (int e = 0; e < NEL; e++) {
      e0[e] = fma(atoms[i * NEL + e], a, e0[e]);
      e1[e] = fma(atoms[i * NEL + e], b, e1[e]);
    }
  }
#if NJ
  for (int b = 0; b < NJ; b++) {
    const double q = Jin[(size_t)cell * NJ + b];
    p += q >= 0.0 ? 0.0 * q : __longlong_as_double(0x7ff8000000000000ull);
  }
#endif
  const double Tc = Ta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
  if (Tc < TMIN) return GUARD_COLD;
  if (!(isfinite(p) && mx > 0.0 && isfinite(Tc) && isfinite(nd) && nd > 0.0 && isfinite(dta[cell]) &&
        dta[cell] >= 0.0))  // the call kernel's bad-input test (order-free: some x > 0, not the sum)
    return GUARD_BAD_INPUT;
  double s0 = 0.0, s1 = 0.0;
#pragma unroll
  for (int e = 0; e < NEL; e++) {
    s0 += e0[e];
    s1 += e1[e];
  }
  bool drift = false;  // share moved beyond rounding (DRIFT) and what clamping accepted negatives can add
#pragma unroll
  for (int e = 0; e < NEL; e++)
    drift = drift || fabs(e1[e] / s1 - e0[e] / s0) > DRIFT * e0[e] / s0 + 2.0 * atol * nok * ETOT / s0;
  const int c = (xfin ? 0 : GUARD_NONFINITE) | (done ? 0 : GUARD_UNFINISHED) | (drift ? GUARD_DRIFT : 0);
  if (c) return c;
  double* q = xout + (size_t)cell * NS;
  double m[NEL * NEL], m0[NEL], y[NEL], lam[NEL];
  for (int e = 0; e < NEL * NEL; e++) m[e] = 0.0;
  for (int i = 0; i < NS; i++) {
    const double* ai = atoms + i * NEL;
    for (int e = 0; e < NEL; e++) {
      if (ai[e] == 0.0) continue;
      for (int f = e; f < NEL; f++)
        if (ai[f] != 0.0) m[e * NEL + f] = fma(ai[e] * ai[f], q[i], m[e * NEL + f]);
    }
  }
  const double r = s1 / s0;
#pragma unroll
  for (int e = 0; e < NEL; e++) {
    m0[e] = m[e * NEL + e];
    y[e] = fma(e0[e], r, -e1[e]);
  }
  for (int k = 0; k < NEL; k++) {
    if (!(m[k * NEL + k] > ELEMENT_PIVOT_MIN * m0[k])) continue;
    for (int j = k + 1; j < NEL; j++) {
      const double l = m[k * NEL + j] / m[k * NEL + k];
      for (int f = j; f < NEL; f++) m[j * NEL + f] = fma(-l, m[k * NEL + f], m[j * NEL + f]);
      y[j] = fma(-l, y[k], y[j]);
    }
  }
  for (int k = NEL - 1; k >= 0; k--) {
    lam[k] = 0.0;
    if (!(m[k * NEL + k] > ELEMENT_PIVOT_MIN * m0[k])) continue;
    double b = y[k];
    for (int f = k + 1; f < NEL; f++) b = fma(-m[k * NEL + f], lam[f], b);
    lam[k] = b / m[k * NEL + k];
  }
  double sum = 0.0;
  for (int i = 0; i < NS; i++) {
    const double* ai = atoms + i * NEL;
    double g = 1.0;
#pragma unroll
    for (int e = 0; e < NEL; e++)
      if (ai[e] != 0.0) g = fma(lam[e], ai[e], g);
    g = fmin(fmax(g, 1.0 - GMAX), 1.0 + GMAX);
    sum = fma(q[i], g, sum);
    q[i] *= g;
  }
  if (!(isfinite(sum) && sum > 0.0)) return GUARD_NONFINITE;
  const double is = 1.0 / sum;
  for (int i = 0; i < NS; i++) q[i] *= is;
  return 0;
}

// Checks of each cell after the call (one thread per cell); counts in count[1] to count[COUNT_RECOMPUTED]: bad input, T
// < TMIN, else: non-finite state, unfinished, element drift, recompute. A cell to recompute gets redo = 1 and is
// appended to list (the first count[COUNT_RECOMPUTED] entries; their order is not deterministic, the cells' results do
// not depend on it).
extern "C" __global__ void guard(const double* __restrict__ x0, const double* __restrict__ Ta,
                                 const double* __restrict__ Pa, const double* __restrict__ dta,
                                 double* __restrict__ xout, const int* __restrict__ doneo,
                                 const double* __restrict__ noko, const double* __restrict__ atoms,
                                 int* __restrict__ redo, int* __restrict__ list, int* __restrict__ count, int B,
                                 double atol J_PARAM) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= B) return;
  const int c = check_cell(x0, Ta, Pa, dta, xout, doneo[cell], noko[cell], atoms, atol, cell J_ARG);
  int r = 0;
  if (c == GUARD_COLD)
    atomicAdd(count + COUNT_COLD, 1);
  else if (c == GUARD_BAD_INPUT)
    atomicAdd(count + COUNT_BAD_INPUT, 1);
  else {
    if (c & GUARD_NONFINITE) atomicAdd(count + COUNT_NONFINITE, 1);
    if (c & GUARD_UNFINISHED) atomicAdd(count + COUNT_UNFINISHED, 1);
    if (c & GUARD_DRIFT) atomicAdd(count + COUNT_DRIFT, 1);
    if (c) {
      r = 1;
      list[atomicAdd(count + COUNT_RECOMPUTED, 1)] = cell;
    }
  }
  redo[cell] = r;
}
