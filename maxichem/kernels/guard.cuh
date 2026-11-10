// The guard: checks each cell after the call, and puts its element shares back.

// What the guard's test found in a stepped cell's result: bits, 0 when it passed.
constexpr int GUARD_NONFINITE = 4;
constexpr int GUARD_UNFINISHED = 8;
constexpr int GUARD_DRIFT = 16;

// Whether an element's share moved beyond rounding (DRIFT) and what clamping accepted negatives can add.
__device__ __forceinline__ bool shares_drifted(const double (&e0)[NEL], const double (&e1)[NEL], double s0, double s1,
                                               double nok, double atol) {
  bool drift = false;
#pragma unroll
  for (int e = 0; e < NEL; e++)
    drift = drift || fabs(e1[e] / s1 - e0[e] / s0) > DRIFT * e0[e] / s0 + 2.0 * atol * nok * ETOT / s0;
  return drift;
}

// lambda of (A X A^T) lambda = e0 s1 / s0 - e1, by LDL^T; an element whose pivot vanishes (absent, or tied to the
// others) is left out. Explicit fma, as in the rest of the correction.
__device__ __forceinline__ void element_multipliers(const double* q, const double* __restrict__ atoms,
                                                    const double (&e0)[NEL], const double (&e1)[NEL], double r,
                                                    double (&lam)[NEL]) {
  double m[NEL * NEL], m0[NEL], y[NEL];
  for (int e = 0; e < NEL * NEL; e++) m[e] = 0.0;
  for (int i = 0; i < NS; i++) {
    const double* ai = atoms + i * NEL;
    for (int e = 0; e < NEL; e++) {
      if (ai[e] == 0.0) continue;
      for (int f = e; f < NEL; f++)
        if (ai[f] != 0.0) m[e * NEL + f] = fma(ai[e] * ai[f], q[i], m[e * NEL + f]);
    }
  }
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
}

// The guard's test of the result of a cell that was stepped (network species order): the bits GUARD_NONFINITE,
// GUARD_UNFINISHED, GUARD_DRIFT (0: passed). A cell that passes has its element shares put back (docs/method.md):
// x' = x (1 + sum_e lambda_e a_ie). Explicit fma: x depends on these sums, and every kernel that calls this must get
// the same bits. It does not screen the inputs: the kernel that stepped the cell did, once (its status).
__device__ int check_cell(const double* __restrict__ x0, double* xout, int done, double nok,
                          const double* __restrict__ atoms, double atol, int cell) {
  double e0[NEL], e1[NEL];  // atoms of each element in the input and the new state
#pragma unroll
  for (int e = 0; e < NEL; e++) {
    e0[e] = 0.0;
    e1[e] = 0.0;
  }
  bool xfin = true;
  for (int i = 0; i < NS; i++) {
    const double a = x0[(size_t)cell * NS + i], b = xout[(size_t)cell * NS + i];
    xfin = xfin && isfinite(b);
#pragma unroll
    for (int e = 0; e < NEL; e++) {
      e0[e] = fma(atoms[i * NEL + e], a, e0[e]);
      e1[e] = fma(atoms[i * NEL + e], b, e1[e]);
    }
  }
  double s0 = 0.0, s1 = 0.0;
#pragma unroll
  for (int e = 0; e < NEL; e++) {
    s0 += e0[e];
    s1 += e1[e];
  }
  const bool drift = shares_drifted(e0, e1, s0, s1, nok, atol);
  const int c = (xfin ? 0 : GUARD_NONFINITE) | (done ? 0 : GUARD_UNFINISHED) | (drift ? GUARD_DRIFT : 0);
  if (c) return c;
  double* q = xout + (size_t)cell * NS;
  double lam[NEL];
  element_multipliers(q, atoms, e0, e1, s1 / s0, lam);
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

// Checks of each cell after the call (one thread per cell); counts in count[1] to count[COUNT_RECOMPUTED]: a refused
// cell by its status (bad input, T < TMIN, T > TMAX), else: non-finite state, unfinished, element drift, recompute. A
// cell to recompute gets redo = 1 and is appended to list (the first count[COUNT_RECOMPUTED] entries; their order is
// not deterministic, the cells' results do not depend on it).
extern "C" __global__ void guard(const double* __restrict__ x0, double* __restrict__ xout,
                                 const int* __restrict__ doneo, const double* __restrict__ noko,
                                 const int* __restrict__ status, const double* __restrict__ atoms,
                                 int* __restrict__ redo, int* __restrict__ list, int* __restrict__ count, int B,
                                 double atol) {
  const int cell = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell >= B) return;
  const int s = status[cell];
  int r = 0;
  if (s == REFUSED_COLD)
    atomicAdd(count + COUNT_COLD, 1);
  else if (s == REFUSED_HOT)
    atomicAdd(count + COUNT_HOT, 1);
  else if (s == REFUSED_BAD_INPUT)
    atomicAdd(count + COUNT_BAD_INPUT, 1);
  else {
    const int c = check_cell(x0, xout, doneo[cell], noko[cell], atoms, atol, cell);
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
