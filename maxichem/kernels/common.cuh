// Constants, the shared helpers, the right-hand side and the triangular solves of the call kernels.

#define NL (32 * W)
// The method's numbers (docs/method.md). Times are in s and temperatures in K; the others have no unit.
constexpr double GAMMA = 0.5;              // Rodas3's diagonal coefficient
constexpr double RODAS_C21 = 4.0;          // Rodas3's stage coupling C_21
constexpr double RODAS_C43 = 8.0 / 3.0;    // Rodas3's stage coupling, -C_43
constexpr double H_MIN = 1e-12;            // s: the smallest first try of a call
constexpr double CALL_END = 1.0 - 1e-14;   // a call ends at this share of dt, so rounding does not force one more step
constexpr double STEP_SAFETY = 0.9;        // the safety factor on err^(-1/3)
constexpr double ERR_FLOOR = 1e-10;        // keeps the step factor finite at err = 0
constexpr double FAC_NONFINITE = 0.1;      // the step factor after a non-finite err
constexpr double FACACC = 0.2;             // the smallest step factor after an accepted step
constexpr double FACREJ = 0.2;             // the smallest step factor after a rejected try
constexpr double FACMAX = 20.0;            // the largest step factor
constexpr double FAC_AFTER_REJECT = 0.5;   // the largest step factor after a rejected try
constexpr double SECMIN = 1e-4;            // the smallest secant shrink
constexpr double SECANT_MIN_ORDER = 0.05;  // below this fitted order of err in h, the shrink is SECMIN
constexpr double SECANT_MAX_ORDER = 3.0;   // the method's error order: the largest order the fit may use
constexpr double SECANT_TARGET = 0.5;      // the err a secant shrink aims at
constexpr double PIVTAU = 0.1;             // recovery: exchange rows only below this share of the column's largest
constexpr double ELEMENT_PIVOT_MIN = 1e-10;  // the guard leaves out an element whose pivot falls below this share
constexpr double NASA_SPLIT_T = 1000.0;      // K: where the NASA-9 fits change range
// The Troe factor's published coefficients (Venot et al. 2012, eq. C.2-C.5).
constexpr double TROE_C0 = 0.4;
constexpr double TROE_C1 = 0.67;
constexpr double TROE_N0 = 0.75;
constexpr double TROE_N1 = 1.27;
constexpr double TROE_D = 0.14;
#define RSN (2 * NS + NNZ)  // the copy of F, u and -Jn kept for the try after a rejection (doubles per cell)
#define FULL 0xffffffffu
constexpr double GMAX = 1e-3;  // the element-share correction changes a species by at most this, relative
#define DNAN __longlong_as_double(0x7ff8000000000000ll)  // poisons the bad-input sum; a failed cell's derivatives
#define CSYNC(g) asm volatile("bar.sync %0, %1;" ::"r"((g) + 1), "r"(NL) : "memory")  // the cell's threads
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;

#if NJ
#define J_PARAM , const double* __restrict__ Jin
#define J_ARG , Jin
#else
#define J_PARAM
#define J_ARG
#endif

struct Tab {                         // small tables, copied to shared memory once per block
  const u16 *smap, *rowptr, *diag;   // slot of (row, col) or NO_SLOT; first leading slot of each row; diagonal slots
  const u8 *colof, *rx, *xp, *blen;  // column of each leading slot; species of each reaction (3 + 3, pad NS); pieces;
};  // pivot batches

__device__ __forceinline__ double wsum(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v;
}

__device__ __forceinline__ double wmax(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_xor_sync(FULL, v, o));
  return v;
}

// Step-size factor after a try of size ha with err en, accepted or not (ok); prej, pen, pha: the call's previous try
// was rejected, its err, its size (docs/method.md).
__device__ __forceinline__ double step_factor(double en, bool ok, bool prej, double pen, double pha, double ha) {
  const bool efin = isfinite(en);
  double fac = efin ? STEP_SAFETY / cbrt(fmax(en, ERR_FLOOR)) : FAC_NONFINITE;
  fac = fmin(fmax(fac, ok ? FACACC : FACREJ), FACMAX);
  if (!ok && efin && prej && en > 1.0 && pen > 1.0 && pha > ha) {  // secant: err ~ h^pe through the two tries
    const double pe = log(pen / en) / log(pha / ha);
    fac = pe > SECANT_MIN_ORDER ? fmax(pow(SECANT_TARGET / en, 1.0 / fmin(pe, SECANT_MAX_ORDER)), SECMIN) : SECMIN;
  }
  if (!ok) fac = fmin(fac, FAC_AFTER_REJECT);
  return fac;
}

// Rate coefficient of the law in row q (_law) at T, ln T, 1 / T, n_tot; three-body n_tot excluded.
// The collision limit: q, the larger of the two directions' bimolecular-equivalent rates (k, M folded in, times
// nd^(molecules - 2), for directions of two or three molecules); if q > K_COLL both directions take K_COLL / q.
#define COLL(fl, kf, kr, nd)                                                                                    \
  do {                                                                                                          \
    const int mf_ = ((fl) >> FL_FORWARD_SHIFT) & 3, mr_ = ((fl) >> FL_REVERSE_SHIFT) & 3;                       \
    const double q_ =                                                                                           \
        fmax(mf_ >= 2 ? (kf) * (mf_ == 3 ? (nd) : 1.0) : 0.0, mr_ >= 2 ? (kr) * (mr_ == 3 ? (nd) : 1.0) : 0.0); \
    if (q_ > KCOLL) {                                                                                           \
      const double c_ = KCOLL / q_;                                                                             \
      (kf) *= c_;                                                                                               \
      (kr) *= c_;                                                                                               \
    }                                                                                                           \
  } while (0)

__device__ __forceinline__ double law(const double* q, int fl, double Tc, double lT, double iT, double nd) {
  double k = exp(q[0] + q[1] * lT - q[2] * iT);
  if (fl & FL_SECOND_K0) k += exp(q[3] + q[4] * lT - q[5] * iT);
#if HAS_TROE
  if (fl & FL_FALLOFF) {
    const double pr = k * nd / exp(q[6] + q[7] * lT - q[8] * iT);
    k = k / (1.0 + pr);
    if (fl & FL_TROE) {  // Troe factor (a, T3, T1, T2 = q[9..12])
      const double fc = log10((1.0 - q[9]) * exp(-Tc / q[10]) + q[9] * exp(-Tc / q[11]) + exp(-q[12] / Tc));
      const double x = (fl & FL_VM11) ? log10(pr) : log10(pr) - TROE_C0 - TROE_C1 * fc;
      const double y = (fl & FL_VM11) ? x / (TROE_N0 - TROE_N1 * fc) : x / (TROE_N0 - TROE_N1 * fc - TROE_D * x);
      k *= exp10(fc / (1.0 + y * y));
    }
  }
#else
  if (fl & FL_FALLOFF) k = k / (1.0 + k * nd / exp(q[6] + q[7] * lT - q[8] * iT));
#endif
#if HAS_EFF
  if (fl & FL_EFF) k *= q[13];
#endif
  return k;
}

// The reverse rate coefficient, inside a kernel's rate loop: from the equilibrium constant, or an explicit reverse law.
#if HAS_REV
#define REVERSE_RATE                                                         \
  ((fl & FL_REV) ? law(rc + RCW * (NR + r), rf[NR + r], Tc, lT, iT, nd) * c3 \
                 : (k * exp(dg + (double)(((fl >> FL_DMU_SHIFT) & 7) - 4) * lKT)) * c3)
#else
#define REVERSE_RATE ((k * exp(dg + (double)(((fl >> FL_DMU_SHIFT) & 7) - 4) * lKT)) * c3)
#endif

// F = (n_tot / nd) S v(n) at the state Y (n = nd Y / n_tot, n_tot = sum Y); u0 = S v / nd unless null; returns n_tot.
// Every warp of the cell sums n_tot itself; the rest is spread over the cell's NL lanes (cl).
// size-limit-ok: inline helpers could move which operations fuse, so a split must first prove equal bits
__device__ __noinline__ double rhs(const double* Y, double* F, double* u0, double* n, double* v, double* part,
                                   const double* kf, const double* kr, double nd, const Tab& tab,
                                   const unsigned* __restrict__ sst, int sround, int lane, int cl, int grp) {
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += Y[i];
  const double ntot = wsum(p), xf = nd / ntot, ixf = ntot / nd;
  for (int i = cl; i < NS; i += NL) n[i] = Y[i] * xf;
  for (int i = cl; i < NPART; i += NL) part[i] = 0.0;
  if (cl == 0) n[NS] = 1.0;
  CSYNC(grp);
#pragma unroll
  for (int qq = 0; qq < (NR + NL - 1) / NL; qq++) {  // each round of NL reactions with its widest sides
    const int r = cl + NL * qq, na = VSA(qq), nb = VSB(qq);
    if (r < NR) {
      const u8* q = tab.rx + 6 * r;
      if (na == 3 || nb == 3) {
        const double a = kf[r] * n[q[0]] * n[q[1]] * n[q[2]];
        const double b = kr[r] * n[q[3]] * n[q[4]] * n[q[5]];
        v[r] = __dsub_rn(a, b);
      } else {
        double a = kf[r] * n[q[0]], b = kr[r] * n[q[3]];
        if (na == 2) a = a * n[q[1]];
        if (nb == 2) b = b * n[q[4]];
        v[r] = __dsub_rn(a, b);
      }
    }
  }
  CSYNC(grp);
  {  // S v as lane streams: reaction, sign, x2, destination piece, end of piece
    double acc = 0.0;
#pragma unroll 4
    for (int q = 0; q < sround; q++) {
      const unsigned word = __ldg(sst + q * NL + cl);
      if (word >> 31) {
        const double cf = ((word >> SV_DOUBLE) & 1u) ? (((word >> SV_SIGN) & 1u) ? -2.0 : 2.0)
                                                     : (((word >> SV_SIGN) & 1u) ? -1.0 : 1.0);
        acc = fma(cf, v[word & SV_REACTION_MASK], acc);
        if ((word >> SV_END) & 1u) {
          part[(word >> SV_PIECE) & SV_PIECE_MASK] = acc;
          acc = 0.0;
        }
      }
    }
  }
  CSYNC(grp);
  for (int i = cl; i < NS; i += NL) {
    double s = part[i];
    for (int e = tab.xp[i]; e < tab.xp[i + 1]; e++) s += part[NS + e];
    F[i] = s * ixf;
    if (u0) u0[i] = s / nd;
  }
  CSYNC(grp);
  return ntot;
}

// a = M0^-1 y on one warp: lane l returns rows l + 32 p (< K0) in a0[p] and K0 + l (< NS) in a1. L and U of the sparse
// leading columns come from shared memory, the trailing block from registers c.
// size-limit-ok: inline helpers could move which operations fuse, so a split must first prove equal bits
__device__ __forceinline__ void msolve(const double* M, const double* dinv, const double* y, const u16* smap,
                                       const double (&c)[NT], int lane, int myt, double (&a0)[RP], double& a1r) {
  const bool ht = lane < NT;
  double d0[RP];
#pragma unroll
  for (int p = 0; p < RP; p++) {
    const int i = lane + 32 * p;
    a0[p] = i < K0 ? y[i] : 0.0;
    d0[p] = i < K0 ? dinv[i] : 0.0;
  }
  double a1 = ht ? y[K0 + lane] : 0.0;
  const double d1 = ht ? dinv[K0 + lane] : 0.0;
#pragma unroll
  for (int k = 0; k < K0; k++) {  // unit L, sparse columns
    if (!(LPRE(k) | LTAIL(k))) continue;
    const double yk = __shfl_sync(FULL, a0[k >> 5], k & 31);
    if (LPRE(k)) {
#pragma unroll
      for (int p = k >> 5; p < RP; p++) {
        const int i = lane + 32 * p;
        if (i > k && i < K0) {
          const unsigned s = smap[i * NS + k];
          if (s != NO_SLOT) a0[p] -= M[s] * yk;
        }
      }
    }
    if (LTAIL(k) && ht) {
      const unsigned s = smap[(K0 + lane) * NS + k];
      if (s != NO_SLOT) a1 -= M[s] * yk;
    }
  }
#pragma unroll
  for (int k = 0; k < NT - 1; k++) {  // unit L, trailing columns
    const double yk = __shfl_sync(FULL, a1, k);
    if (myt > k) a1 -= c[k] * yk;
  }
#pragma unroll
  for (int k = NT - 1; k >= 0; k--) {  // U, trailing columns
    if (k == 0 && !UTAIL(0)) continue;
    const double zk = __shfl_sync(FULL, a1 * d1, k);
    if (UTAIL(k)) {
#pragma unroll
      for (int p = 0; p < RP; p++) {
        const int i = lane + 32 * p;
        if (i < K0) {
          const unsigned s = smap[i * NS + K0 + k];
          if (s != NO_SLOT) a0[p] -= M[s] * zk;
        }
      }
    }
    if (k > 0 && myt >= 0 && myt < k) a1 -= c[k] * zk;
  }
#pragma unroll
  for (int k = K0 - 1; k >= 0; k--) {  // U, sparse columns
    if (!UPRE(k)) continue;
    const double zk = __shfl_sync(FULL, a0[k >> 5] * d0[k >> 5], k & 31);
#pragma unroll
    for (int p = 0; p <= (k >> 5); p++) {
      const int i = lane + 32 * p;
      if (i < k) {
        const unsigned s = smap[i * NS + k];
        if (s != NO_SLOT) a0[p] -= M[s] * zk;
      }
    }
  }
#pragma unroll
  for (int p = 0; p < RP; p++) a0[p] *= d0[p];
  a1r = a1 * d1;
}
