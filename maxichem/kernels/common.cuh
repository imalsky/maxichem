// Constants, the shared helpers, the right-hand side and the triangular solves of the call kernels.

#define NL (32 * W)
// Rodas3's coefficients (Sandu et al. 1997; docs/method.md).
constexpr double GAMMA = 0.5;            // the diagonal coefficient
constexpr double RODAS_C21 = 4.0;        // the stage coupling C_21
constexpr double RODAS_C43 = 8.0 / 3.0;  // the stage coupling, -C_43
// The method's settings, from maxichem/_constants.py through the header, where each has its unit and reason.
constexpr double H_MIN = H_MIN_VALUE;
constexpr double CALL_END = CALL_END_VALUE;
constexpr double STEP_SAFETY = STEP_SAFETY_VALUE;
constexpr double ERR_FLOOR = ERR_FLOOR_VALUE;
constexpr double FAC_NONFINITE = FAC_NONFINITE_VALUE;
constexpr double FACACC = FACACC_VALUE;
constexpr double FACREJ = FACREJ_VALUE;
constexpr double FACMAX = FACMAX_VALUE;
constexpr double FAC_AFTER_REJECT = FAC_AFTER_REJECT_VALUE;
constexpr double SECMIN = SECMIN_VALUE;
constexpr double SECANT_MIN_ORDER = SECANT_MIN_ORDER_VALUE;
constexpr double SECANT_MAX_ORDER = SECANT_MAX_ORDER_VALUE;
constexpr double SECANT_TARGET = SECANT_TARGET_VALUE;
constexpr double PIVTAU = PIVTAU_VALUE;
constexpr double ELEMENT_PIVOT_MIN = ELEMENT_PIVOT_MIN_VALUE;
constexpr double REC_RETRIES = REC_RETRIES_VALUE;
constexpr double REC_RETRY_SPLIT = REC_RETRY_SPLIT_VALUE;
constexpr double NASA_SPLIT_T = NASA_SPLIT_T_K;  // K, from maxichem/_constants.py
// The Troe factor's published coefficients (Venot et al. 2012, eq. C.2-C.5).
constexpr double TROE_C0 = 0.4;
constexpr double TROE_C1 = 0.67;
constexpr double TROE_N0 = 0.75;
constexpr double TROE_N1 = 1.27;
constexpr double TROE_D = 0.14;
#define FULL 0xffffffffu
constexpr double GMAX = GMAX_VALUE;                      // from maxichem/_constants.py
#define DNAN __longlong_as_double(0x7ff8000000000000ll)  // poisons the bad-input sum; a failed cell's derivatives
#define CSYNC(g) asm volatile("bar.sync %0, %1;" ::"r"((g) + 1), "r"(NL) : "memory")  // the cell's threads
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;

#if NJ
#define J_PARAM , const double* __restrict__ Jin
#define J_ARG , Jin
#define J_OR_NULL Jin
#else
#define J_PARAM
#define J_ARG
#define J_OR_NULL nullptr
#endif

// The small tables, copied to shared memory once per block. smap: slot of (row, col) or NO_SLOT; rowptr: first leading
// slot of each row; diag: diagonal slots; colof: column of each leading slot; rx: species of each reaction (3 + 3, pad
// NS); xp: each row's extra S v pieces; blen: pivot batches.
struct Tab {
  const u16 *smap, *rowptr, *diag;
  const u8 *colof, *rx, *xp, *blen;
};

// Copies the small tables to shared memory (tb32, after the block's work arrays) and returns where they are.
__device__ __forceinline__ Tab load_tab(unsigned* tb32, const unsigned* __restrict__ blob) {
  for (int i = (int)threadIdx.x; i < TBYTES / 4; i += (int)blockDim.x) tb32[i] = blob[i];
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
  return tab;
}

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
// was rejected, its err, its size (docs/method.md). The plain law's cube root is err^(-1 / SECANT_MAX_ORDER).
static_assert(SECANT_MAX_ORDER == 3.0, "step_factor takes cbrt, and step_factor_partials its derivative");
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

// The step control's state over a call: the next try's size h and the time reached t (s), the accepted steps nok, and
// the previous try's error pen, size pha (s) and rejection prej.
struct StepState {
  double h, t, nok, pen, pha;
  bool prej;
};

// An accepted try of size ha: the new state N = max(Y, 0) over the rows first, first + stride, ..., and the step
// counted. The caller's barrier follows, then next_try.
__device__ __forceinline__ void accept_try(StepState& s, double* N, const double* Y, bool ok, double ha, int first,
                                           int stride) {
  if (ok) {
    for (int i = first; i < NS; i += stride) N[i] = fmax(Y[i], 0.0);
    s.t += ha;
    s.nok += 1.0;
  }
}

// The recovery's starting step on pass k (0: the call's, m = min(h, dt)): from pass 1 on at most dt REC_RETRY_SPLIT^k,
// s. A pass k > 0 runs only when the pass before it finished and failed the guard's element test (should_retry).
__device__ __forceinline__ double recovery_start(double m, double dt, int k) {
  return k == 0 ? m : fmin(m, dt * pow(REC_RETRY_SPLIT, (double)k));
}

// The next try's size from the step factor fac after a try of size ha with error en, and what the next factor reads
// of this one.
__device__ __forceinline__ void next_try(StepState& s, double fac, bool ok, double en, double ha) {
  s.h = ha * fac;
  s.prej = !ok;
  s.pen = en;
  s.pha = ha;
}

// Rate coefficient of the law in row q (_law) at T, ln T, 1 / T, n_tot; three-body n_tot excluded.
__device__ __forceinline__ double law(const double* q, int fl, double Tc, double lT, double iT, double nd) {
  double k = exp(q[0] + q[1] * lT - q[2] * iT);
  if (fl & FL_SECOND_K0) k += exp(q[3] + q[4] * lT - q[5] * iT);
  // Falloff at the ends of Pr: k0 = 0 has no rate, also where k_inf = 0 makes Pr 0 / 0; Pr = inf (k_inf = 0) gives
  // the high-pressure rate 0, Pr = 0 (k_inf overflowing) the low-pressure rate k0 times the Troe factor's limit
#if HAS_TROE
  if (fl & FL_FALLOFF) {
    const double pr = k * nd / exp(q[6] + q[7] * lT - q[8] * iT);
    k = k > 0.0 ? k / (1.0 + pr) : 0.0;
    if ((fl & FL_TROE) && k > 0.0) {  // Troe factor (a, T3, T1, T2 = q[9..12])
      const double fc = log10((1.0 - q[9]) * exp(-Tc / q[10]) + q[9] * exp(-Tc / q[11]) + exp(-q[12] / Tc));
      const double x = (fl & FL_VM11) ? log10(pr) : log10(pr) - TROE_C0 - TROE_C1 * fc;
      // at Pr = 0 the standard ratio tends to -1 / d, so F to Fc^(d^2 / (1 + d^2)); the VM11 one to -inf, F to 1
      const double y = (fl & FL_VM11) ? x / (TROE_N0 - TROE_N1 * fc)
                                      : (isfinite(x) ? x / (TROE_N0 - TROE_N1 * fc - TROE_D * x) : -1.0 / TROE_D);
      k *= exp10(fc / (1.0 + y * y));
    }
  }
#else
  if (fl & FL_FALLOFF) k = k > 0.0 ? k / (1.0 + k * nd / exp(q[6] + q[7] * lT - q[8] * iT)) : 0.0;
#endif
#if HAS_EFF
  if (fl & FL_EFF) k *= q[13];
#endif
  return k;
}

// A direction's bimolecular-equivalent rate: k (M folded in, three_body) times nd^(molecules - 2), where molecules plus
// M make at least two collision partners: k / nd for a dissociation A + M, 0 for one molecule without M.
__device__ __forceinline__ double bimolecular_rate(int molecules, bool three_body, double k, double nd) {
  return molecules >= 2 ? k * (molecules == 3 ? nd : 1.0) : (three_body ? k / nd : 0.0);
}

// The collision limit: q, the larger of the two directions' bimolecular-equivalent rates; if q > K_COLL both directions
// take K_COLL / q.
__device__ __forceinline__ void collision_limit(int fl, double& kf, double& kr, double nd) {
  const int mf = (fl >> FL_FORWARD_SHIFT) & 3, mr = (fl >> FL_REVERSE_SHIFT) & 3;
  const bool three_body = (fl & FL_THREE_BODY) != 0;
  const double q = fmax(bimolecular_rate(mf, three_body, kf, nd), bimolecular_rate(mr, three_body, kr, nd));
  if (q > KCOLL) {
    const double c = KCOLL / q;
    kf *= c;
    kr *= c;
  }
}

// H / RT and S / R of a species from its NASA-9 coefficients a, at T (with its log, inverse and powers).
__device__ __forceinline__ double enthalpy_rt(const double* a, double Tc, double lT, double iT, double T2, double T3,
                                              double T4) {
  return -a[0] * iT * iT + a[1] * lT * iT + a[2] + a[3] * Tc / 2 + a[4] * T2 / 3 + a[5] * T3 / 4 + a[6] * T4 / 5 +
         a[7] * iT;
}

__device__ __forceinline__ double entropy_r(const double* a, double Tc, double lT, double iT, double T2, double T3,
                                            double T4) {
  return -a[0] * iT * iT / 2 - a[1] * iT + a[2] * lT + a[3] * Tc + a[4] * T2 / 2 + a[5] * T3 / 3 + a[6] * T4 / 4 + a[8];
}

// G / RT of the species first, first + stride, ... into g.
__device__ __forceinline__ void gibbs_rt(double* g, const double* __restrict__ nasa, double Tc, double lT, double iT,
                                         int first, int stride) {
  const double* cf = nasa + (Tc < NASA_SPLIT_T ? 0 : NS * 9);
  for (int i = first; i < NS; i += stride) {
    const double* a = cf + 9 * i;
    const double T2 = Tc * Tc, T3 = T2 * Tc, T4 = T3 * Tc;
    g[i] = enthalpy_rt(a, Tc, lT, iT, T2, T3, T4) - entropy_r(a, Tc, lT, iT, T2, T3, T4);
  }
}

// Molecules on the reverse side minus the forward side, from a reaction's flags.
__device__ __forceinline__ double reaction_dmu(int fl) { return (double)(((fl >> FL_DMU_SHIFT) & 7) - 4); }

// Thermal reaction r's rate coefficients before the collision limit: three-body ones times nd; the reverse from the
// species' G / RT g.
__device__ __forceinline__ void rate_pair(int r, int fl, const double* __restrict__ rc, const u8* x, const double* g,
                                          double Tc, double lT, double iT, double lKT, double nd, double& kf,
                                          double& kr) {
  const double k = law(rc + RCW * r, fl, Tc, lT, iT, nd);
  const double dg = g[x[3]] + g[x[4]] + g[x[5]] - g[x[0]] - g[x[1]] - g[x[2]];
  const double c3 = (fl & FL_THREE_BODY) ? nd : 1.0;
  kf = k * c3;
  kr = (k * exp(dg + reaction_dmu(fl) * lKT)) * c3;
}

// Rate coefficients of a cell across the cell's lanes (grp: its named barrier); photolysis rows kf = J, kr = 0.
// size-limit-ok: in the call kernel; a struct of its one group, rc, rf and nasa, saves 2 for an untimed call change
__device__ __forceinline__ void cell_rates(double* g, double* kf, double* kr, double Tc, double nd,
                                           const double* __restrict__ rc, const int* __restrict__ rf,
                                           const double* __restrict__ nasa, const Tab& tab,
                                           const double* __restrict__ Jc, int cell, int cl, int grp) {
  const double lT = log(Tc), iT = 1.0 / Tc, lKT = log(KBOLTZ * Tc / P0BAR);
  gibbs_rt(g, nasa, Tc, lT, iT, cl, NL);
  if (cl == 0) g[NS] = 0.0;
  CSYNC(grp);
  for (int r = cl; r < NR; r += NL) {
#if NJ
    if (r >= NRT) {
      kf[r] = Jc[(size_t)cell * NJ + r - NRT];
      kr[r] = 0.0;
      continue;
    }
#endif
    const int fl = rf[r];
    rate_pair(r, fl, rc, tab.rx + 6 * r, g, Tc, lT, iT, lKT, nd, kf[r], kr[r]);
    collision_limit(fl, kf[r], kr[r], nd);
  }
}

// One x of a cell into the bad-input test's sum (a NaN when x is negative or not finite) and maximum; screen_J adds the
// cell's J to that sum.
__device__ __forceinline__ void screen_x(double xi, double& p, double& pm) {
  p += xi >= 0.0 ? xi : DNAN;
  pm = fmax(pm, xi);
}

__device__ __forceinline__ void screen_J(const double* __restrict__ Jin, int cell, int first, int stride, double& p) {
  for (int b = first; b < NJ; b += stride) {
    const double q = Jin[(size_t)cell * NJ + b];
    p += q >= 0.0 ? 0.0 * q : DNAN;
  }
}

// A cell's screening result: 0 (stepped) or why it is refused. The kernel that steps the cell decides it once and
// writes it to status; the guard counts it from there, and the host names the refused cells from there.
constexpr int REFUSED_COLD = 1;
constexpr int REFUSED_BAD_INPUT = 2;
constexpr int REFUSED_HOT = 3;

// The screening result from the bad-input test (fine) and T: below TMIN, above TMAX (a NaN T is neither: bad input).
__device__ __forceinline__ int cell_status(bool fine, double Tc) {
  return Tc < TMIN ? REFUSED_COLD : (Tc > TMAX ? REFUSED_HOT : (fine ? 0 : REFUSED_BAD_INPUT));
}

// The bad-input test's forcing: T and nd finite, P > 0, dt finite and at least 0.
__device__ __forceinline__ bool forcing_fine(double Tc, double nd, double dt) {
  return isfinite(Tc) && isfinite(nd) && nd > 0.0 && isfinite(dt) && dt >= 0.0;
}

// The bad-input test on one warp (not stepped when false): non-finite, x < 0, |sum x - 1| > XSUMTOL (written so that a
// NaN sum fails), no x > 0, P <= 0 or dt < 0. p, pm: this lane's screen_x sum and maximum.
__device__ __forceinline__ bool cell_fine(double p, double pm, double Tc, double nd, double dt) {
  const double ps = wsum(p);
  return fabs(ps - 1.0) <= XSUMTOL && wmax(pm) > 0.0 && forcing_fine(Tc, nd, dt);
}

// The net rate v[r] of reaction r; na, nb: the widest sides of its round, so a side of fewer multiplies by n[NS] = 1.
__device__ __forceinline__ void reaction_rate(int r, int na, int nb, const double* n, double* v, const double* kf,
                                              const double* kr, const Tab& tab) {
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

// A stream word's coefficient: +-1 or +-2, from its bits.
__device__ __forceinline__ double stream_coefficient(unsigned word, int double_bit, int sign_bit) {
  return ((word >> double_bit) & 1u) ? (((word >> sign_bit) & 1u) ? -2.0 : 2.0)
                                     : (((word >> sign_bit) & 1u) ? -1.0 : 1.0);
}

// S v on lane cl of the stream: reaction, sign, x2, destination piece, end of piece.
__device__ __forceinline__ void sv_lane(int cl, const double* v, double* part, const unsigned* __restrict__ sst,
                                        int sround) {
  double acc = 0.0;
#pragma unroll 4
  for (int q = 0; q < sround; q++) {
    const unsigned word = __ldg(sst + q * NL + cl);
    if (word >> 31) {
      acc = fma(stream_coefficient(word, SV_DOUBLE, SV_SIGN), v[word & SV_REACTION_MASK], acc);
      if ((word >> SV_END) & 1u) {
        part[(word >> SV_PIECE) & SV_PIECE_MASK] = acc;
        acc = 0.0;
      }
    }
  }
}

// F and u0 of row i from its S v pieces.
__device__ __forceinline__ void gather_row(int i, const double* part, double* F, double* u0, double nd, double ixf,
                                           const Tab& tab) {
  double s = part[i];
  for (int e = tab.xp[i]; e < tab.xp[i + 1]; e++) s += part[NS + e];
  F[i] = s * ixf;
  if (u0) u0[i] = s / nd;
}

// F = (n_tot / nd) S v(n) at the state Y (n = nd Y / n_tot, n_tot = sum Y); u0 = S v / nd unless null; returns n_tot.
// Every warp of the cell sums n_tot itself; the rest is spread over the cell's NL lanes (cl).
// size-limit-ok: 15; __noinline__, so a struct by reference would have to sit in local memory
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
    const int r = cl + NL * qq;
    if (r < NR) reaction_rate(r, VSA(qq), VSB(qq), n, v, kf, kr, tab);
  }
  CSYNC(grp);
  sv_lane(cl, v, part, sst, sround);
  CSYNC(grp);
  for (int i = cl; i < NS; i += NL) gather_row(i, part, F, u0, nd, ixf, tab);
  CSYNC(grp);
  return ntot;
}

// M = -Jn on lane cl of the Jacobian's stream: rate index, sign, x2, species a, b, end of slot; slots from jsl. Y holds
// the extra pieces of long slots.
__device__ __forceinline__ void jacobian_lane(int cl, double* M, [[maybe_unused]] double* Y, const double* kf,
                                              const double* n, const u64* __restrict__ jt, int jround) {
  const unsigned* jw = (const unsigned*)jt;
  const u16* jsl = (const u16*)(jw + jround * NL) + cl * JSMAX;
  double acc = 0.0;
  int ne = 0;
#pragma unroll 4
  for (int q = 0; q < jround; q++) {
    const unsigned word = __ldg(jw + q * NL + cl);  // kr = kf + NR
    const double pr = kf[word & J_RATE_MASK] * n[(word >> J_A) & J_SPECIES_MASK] * n[(word >> J_B) & J_SPECIES_MASK];
    acc = fma(stream_coefficient(word, J_DOUBLE, J_SIGN), pr, acc);
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

// The extra pieces of long slots, in order, on one thread.
__device__ __forceinline__ void jacobian_extras(double* M, const double* Y, const u64* __restrict__ jt, int jround) {
  const u16* fx = (const u16*)((const unsigned*)jt + jround * NL) + NL * JSMAX;
  for (int e = 0; e < NFIX; e++) M[__ldg(fx + e)] += Y[e];
}

// Row i of w = S v / nd - Jn x from the sparse -Jn (the first half is u[i] on entry).
__device__ __forceinline__ double jacobian_times_x(int i, const double* M, const double* N, const Tab& tab) {
  double acc = 0.0;
  const int s1 = tab.rowptr[i + 1];
  for (int s = tab.rowptr[i]; s < s1; s++) acc += M[s] * N[tab.colof[s]];
  if (i >= K0)
    for (int j = 0; j < NT; j++) acc += M[TB0 + (i - K0) * NT + j] * N[K0 + j];
  return acc;
}

// One pivot of the sparse LU over the threads first, first + stride, ...: its column scaled by dk.
__device__ __forceinline__ void scale_column(int k, double dk, double* M, const Tab& tab, int first, int stride) {
  for (int i = first; i < NS; i += stride) {
    const unsigned q = tab.smap[i * NS + k];
    if (i > k && q != NO_SLOT) M[q] *= dk;
  }
}

// One update of the sparse LU: slot -= slot b * slot c.
__device__ __forceinline__ void lu_update(double* M, u64 op) {
  M[op & LU_SLOT_MASK] -= M[(op >> LU_B) & LU_SLOT_MASK] * M[(op >> LU_C) & LU_SLOT_MASK];
}

// The dense trailing block's factors in registers c (this lane's row myt, or none); 1 / its diagonal into dinv.
__device__ __forceinline__ void factor_trailing(double (&c)[NT], double* dinv, int myt) {
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
}

// msolve's unit L over the sparse leading columns.
__device__ __forceinline__ void lower_sparse(const double* M, const u16* smap, int lane, double (&a0)[RP], double& a1) {
  const bool ht = lane < NT;
#pragma unroll
  for (int k = 0; k < K0; k++) {
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
}

// msolve's unit L over the trailing columns, then U over them; the U's updates reach the leading rows.
__device__ __forceinline__ void solve_trailing(const double* M, const u16* smap, const double (&c)[NT], double d1,
                                               int lane, int myt, double (&a0)[RP], double& a1) {
#pragma unroll
  for (int k = 0; k < NT - 1; k++) {
    const double yk = __shfl_sync(FULL, a1, k);
    if (myt > k) a1 -= c[k] * yk;
  }
#pragma unroll
  for (int k = NT - 1; k >= 0; k--) {
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
}

// msolve's U over the sparse leading columns.
__device__ __forceinline__ void upper_sparse(const double* M, const u16* smap, const double (&d0)[RP], int lane,
                                             double (&a0)[RP]) {
#pragma unroll
  for (int k = K0 - 1; k >= 0; k--) {
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
}

// a = M0^-1 y on one warp: lane l returns rows l + 32 p (< K0) in a0[p] and K0 + l (< NS) in a1. L and U of the sparse
// leading columns come from shared memory, the trailing block from registers c.
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
  lower_sparse(M, smap, lane, a0, a1);
  solve_trailing(M, smap, c, d1, lane, myt, a0, a1);
  upper_sparse(M, smap, d0, lane, a0);
#pragma unroll
  for (int p = 0; p < RP; p++) a0[p] *= d0[p];
  a1r = a1 * d1;
}

// 1 / (1 - sum of the solved u): the Sherman-Morrison factor c0, from msolve's lane values (every lane returns the
// same).
__device__ __forceinline__ double sherman_morrison_factor(const double (&a0)[RP], double a1, int lane) {
  double sb = lane < K0 ? a0[0] : 0.0;
#pragma unroll
  for (int p = 1; p < RP; p++) sb += lane + 32 * p < K0 ? a0[p] : 0.0;
  return 1.0 / (1.0 - wsum(sb + (lane < NT ? a1 : 0.0)));
}

// msolve's lane values into y.
__device__ __forceinline__ void store_solved(double* y, const double (&a0)[RP], double a1, int lane) {
#pragma unroll
  for (int p = 0; p < RP; p++)
    if (lane + 32 * p < K0) y[lane + 32 * p] = a0[p];
  if (lane < NT) y[K0 + lane] = a1;
}

// The Sherman-Morrison update of a solved stage on one warp: y <- y + c0 (sum y) u.
__device__ __forceinline__ void sherman_morrison_update(double* y, const double* u, double c0, int lane) {
  const bool ht = lane < NT;
  double a0[RP];
#pragma unroll
  for (int p = 0; p < RP; p++) a0[p] = lane + 32 * p < K0 ? y[lane + 32 * p] : 0.0;
  const double a1 = ht ? y[K0 + lane] : 0.0;
  double sa = a0[0];
#pragma unroll
  for (int p = 1; p < RP; p++) sa += a0[p];
  const double s = wsum(sa + a1) * c0;
  __syncwarp();
#pragma unroll
  for (int p = 0; p < RP; p++)
    if (lane + 32 * p < K0) y[lane + 32 * p] = a0[p] + s * u[lane + 32 * p];
  if (ht) y[K0 + lane] = a1 + s * u[K0 + lane];
}

// Rodas3's stage right-hand sides (docs/method.md): F plus the earlier stages, from 1 / h.
__device__ __forceinline__ double stage2_rhs(double ih, double k1, double f) { return fma(RODAS_C21 * ih, k1, f); }

__device__ __forceinline__ double stage3_rhs(double ih, double k1, double k2, double f) { return fma(ih, k1 - k2, f); }

__device__ __forceinline__ double stage4_rhs(double ih, double k1, double k2, double k3, double f) {
  return fma(ih, fma(-RODAS_C43, k3, k1 - k2), f);
}

// A try's error on one warp (the RMS, and while first, no step of the call accepted yet, the larger of RMS and
// max-norm), Y = its new state where keep; ok: accepted.
__device__ __forceinline__ double try_error(const double* N, const double* K1, const double* K3, const double* K4,
                                            double* Y, bool keep, bool first, double rtol, double atol, int lane,
                                            bool& ok) {
  double e2 = 0.0, em = 0.0;
  bool nn = true;
  for (int i = lane; i < NS; i += 32) {
    const double y = N[i] + 2.0 * K1[i] + K3[i] + K4[i];
    if (keep) Y[i] = y;
    const double sc = atol + rtol * fmax(fabs(N[i]), fabs(y));
    const double q = K4[i] / sc;
    e2 += q * q;
    em = fmax(em, fabs(q));
    nn = nn && (y >= -atol);
  }
  const double er = sqrt(wsum(e2) / NS), emx = wmax(em);
  const double en = (first && isfinite(er)) ? fmax(er, emx) : er;
  const bool nonneg = __all_sync(FULL, nn), efin = isfinite(en);
  ok = (en <= 1.0) && nonneg && efin;
  return en;
}

// x = N / sum N at the cell's output row, over the rows first, first + stride, ...; returns the sum (one warp sums it).
__device__ __forceinline__ double write_state(const double* N, double* __restrict__ out, const int* __restrict__ perm,
                                              int lane, int first, int stride) {
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += N[i];
  const double s = wsum(p);
  for (int i = first; i < NS; i += stride) out[perm[i]] = N[i] / s;
  return s;
}
