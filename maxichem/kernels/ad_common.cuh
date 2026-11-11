// What the tangent and adjoint kernels share: the call's arithmetic on one warp, rates and their derivatives.

__device__ const unsigned char PX[] = {PX_SLOTS};  // photolysis branches' product slots, 3 per branch (padding NS)

// The network's term streams in global memory and their rounds (_tables.py, _ad_tables.py): S v, the Jacobian, the LU
// updates and each column's first update, the slots' dense positions (the recovery's), and the adjoint's transposed
// Jacobian with its piece offsets (null in the tangent kernels).
struct Streams {
  const unsigned* __restrict__ sst;
  const u64* __restrict__ jt;
  const u64* __restrict__ lu;
  const int* __restrict__ optr;
  const u16* __restrict__ didx;
  const u64* __restrict__ gt;
  const int* __restrict__ xg;
  int sround, jround, ground;
};

// A cell's rate coefficients, forward and reverse, at its number density nd (cm^-3).
struct Rates {
  const double *kf, *kr;
  double nd;
};

// The call's error tolerances: relative, and absolute in mole fraction.
struct Tolerance {
  double rtol, atol;
};

// What an attempt leaves: its error, its Sherman-Morrison factor (0 on the recovery's dense path) and its acceptance.
struct Attempt {
  double en, c0;
  bool ok;
};

__device__ __forceinline__ double sgn(double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0); }

// A non-finite log multiplier makes the bad-input sum NaN, so its cell takes no try; a finite one leaves the sum's
// value.
__device__ __forceinline__ void screen_lm(const double* lmc, int lane, double& p) {
  if (lmc)
    for (int r = lane; r < NR; r += 32) p += 0.0 * lmc[r];
}

// The call kernel's rhs on one warp, kept apart because it has warp barriers where rhs has the cell's named barrier;
// the primal test in test_ad.py holds the two to the same bits. F = (n_tot / nd) S v(n) at Y, u0 = S v / nd unless
// null; each lane walks W of the call kernel's NL lane streams, so every sum is the call kernel's. Returns n_tot.
// size-limit-ok: 13, rhs's list; __noinline__, so a struct by reference would have to sit in local memory
__device__ __noinline__ double rhs1(const double* Y, double* F, double* u0, double* n, double* v, double* part,
                                    const double* kf, const double* kr, double nd, const Tab& tab,
                                    const unsigned* __restrict__ sst, int sround, int lane) {
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += Y[i];
  const double ntot = wsum(p), xf = nd / ntot, ixf = ntot / nd;
  for (int i = lane; i < NS; i += 32) n[i] = Y[i] * xf;
  for (int i = lane; i < NPART; i += 32) part[i] = 0.0;
  if (lane == 0) n[NS] = 1.0;
  __syncwarp();
#pragma unroll
  for (int qq = 0; qq < (NR + NL - 1) / NL; qq++) {
#pragma unroll
    for (int m = 0; m < W; m++) {
      const int r = lane + 32 * m + NL * qq;
      if (r < NR) reaction_rate(r, VSA(qq), VSB(qq), n, v, kf, kr, tab);
    }
  }
  __syncwarp();
#pragma unroll
  for (int m = 0; m < W; m++) sv_lane(lane + 32 * m, v, part, sst, sround);
  __syncwarp();
  for (int i = lane; i < NS; i += 32) gather_row(i, part, F, u0, nd, ixf, tab);
  __syncwarp();
  return ntot;
}

// o1 = S w1, o2 = S w2 in one pass of the S v lane streams (part holds 2 NPART doubles).
__device__ __forceinline__ void stoichiometry_times_two(const double* w1, const double* w2, double* o1, double* o2,
                                                        double* part, const Tab& tab, const unsigned* __restrict__ sst,
                                                        int sround, int lane) {
  for (int i = lane; i < 2 * NPART; i += 32) part[i] = 0.0;
  __syncwarp();
  for (int m = 0; m < W; m++) {
    const int cl = lane + 32 * m;
    double a1 = 0.0, a2 = 0.0;
    for (int q = 0; q < sround; q++) {
      const unsigned x = __ldg(sst + q * NL + cl);
      if (x >> 31) {
        const int r = (int)(x & SV_REACTION_MASK);
        const double cf = stream_coefficient(x, SV_DOUBLE, SV_SIGN);
        a1 += cf * w1[r];
        a2 += cf * w2[r];
        if ((x >> SV_END) & 1u) {
          const int d = (int)((x >> SV_PIECE) & SV_PIECE_MASK);
          part[2 * d] = a1;
          part[2 * d + 1] = a2;
          a1 = 0.0;
          a2 = 0.0;
        }
      }
    }
  }
  __syncwarp();
  for (int i = lane; i < NS; i += 32) {
    double s1 = part[2 * i], s2 = part[2 * i + 1];
    for (int e = tab.xp[i]; e < tab.xp[i + 1]; e++) {
      s1 += part[2 * (NS + e)];
      s2 += part[2 * (NS + e) + 1];
    }
    o1[i] = s1;
    o2[i] = s2;
  }
  __syncwarp();
}

// The collision limit's derivative weights of a reaction from its pair before the limit (collision_limit's q): wA = 1
// if q > K_COLL, 1/2 at q = K_COLL (the tie of the min), else 0; wG = wA v, v = the reverse direction's share of the
// larger (1/2 at a tie; a direction of fewer than two molecules does not count).
__device__ __forceinline__ void collision_limit_weights(int fl, double kf, double kr, double nd, double& wA,
                                                        double& wG) {
  const int mf = (fl >> FL_FORWARD_SHIFT) & 3, mr = (fl >> FL_REVERSE_SHIFT) & 3;
  const double qf = bimolecular_rate(mf, kf, nd), qr = bimolecular_rate(mr, kr, nd);
  const double q = fmax(qf, qr), weight = q > KCOLL ? 1.0 : (q == KCOLL ? 0.5 : 0.0);
  wA = weight;
  wG = weight * (qr > qf ? 1.0 : (qr < qf ? 0.0 : 0.5));
}

// The collision limit's weight on ln nd (rates_tan's wE): wA - wG for a forward side of 3 molecules, wG for a reverse
// one.
__device__ __forceinline__ double collision_limit_nd_weight(int fl, double wA, double wG) {
  return (wA - wG) * (((fl >> FL_FORWARD_SHIFT) & 3) == 3 ? 1.0 : 0.0) +
         wG * (((fl >> FL_REVERSE_SHIFT) & 3) == 3 ? 1.0 : 0.0);
}

// Rate coefficients of the cell (the call kernel's code: the law, the reverse from Gibbs, the collision limit;
// photolysis rows kf = J, kr = 0), times exp(lm) when per-cell log multipliers are given; cw[r], cw[NR + r] = the
// collision limit's weights (collision_limit_weights).
// size-limit-ok: its one group, the rate tables rc, rf and nasa, would save 2 for an untimed change of this code
__device__ __forceinline__ void rates(double* work, double* kf, double* kr, double* cw, double Tc, double nd,
                                      const double* lmc, const double* Jc, const double* __restrict__ rc,
                                      const int* __restrict__ rf, const double* __restrict__ nasa, const Tab& tab,
                                      int lane) {
  const double lT = log(Tc), iT = 1.0 / Tc, lKT = log(KBOLTZ * Tc / P0BAR);
  double* g = work + A_V;  // G / RT of each species
  gibbs_rt(g, nasa, Tc, lT, iT, lane, 32);
  if (lane == 0) g[NS] = 0.0;
  __syncwarp();
  for (int r = lane; r < NR; r += 32) {
    if (r >= NRT) {
      kf[r] = Jc[r - NRT];
      kr[r] = 0.0;
      cw[r] = 0.0;
      cw[NR + r] = 0.0;
      if (lmc) kf[r] *= exp(lmc[r]);
      continue;
    }
    const int fl = rf[r];
    rate_pair(r, fl, rc, tab.rx + 6 * r, g, Tc, lT, iT, lKT, nd, kf[r], kr[r]);
    collision_limit_weights(fl, kf[r], kr[r], nd, cw[r], cw[NR + r]);
    collision_limit(fl, kf[r], kr[r], nd);
    if (lmc) {
      const double e = exp(lmc[r]);
      kf[r] *= e;
      kr[r] *= e;
    }
  }
  __syncwarp();
}

// Repeats the law's terms because their derivatives need ea, eb, ec apart, where law() folds them; test_ad.py compares
// them with finite differences. Partial derivatives of ln kf of a thermal reaction (rate law q = its rc row, flags fl;
// the collision limit not included) at (T, nd): dT = d/dT, dn = d/d ln nd, D[j] = d/d q[j] for the nine Arrhenius
// columns (the Troe parameters and the efficiency are not differentiated). ln kr adds d(sum G/RT, products -
// reactants)/dT + dmu / T to dT. Troe factor: log10 F = f / (1 + y^2), f = log10 Fc(T), z = log10 Pr, N = 0.75 - 1.27
// f; vm11 y = z / N, standard x = z - 0.4 - 0.67 f, y = x / (N - 0.14 x); s = d log10 F / dz enters d ln k / d ln
// k_low, d ln k_inf, d ln nd.
__device__ __forceinline__ void rate_law_partials(const double* q, int fl, double Tc, double lT, double iT, double nd,
                                                  double& dT, double& dn, double* D) {
  const double e0 = exp(q[0] + q[1] * lT - q[2] * iT),
               e1 = (fl & FL_SECOND_K0) ? exp(q[3] + q[4] * lT - q[5] * iT) : 0.0;
  const double kl = e0 + e1, w0 = kl > 0.0 ? e0 / kl : 0.0, w1 = kl > 0.0 ? e1 / kl : 0.0;
  double ph = 1.0, pc = 0.0, fT = 0.0;  // d ln k / d ln k_low, d ln k / d ln k_inf, d ln F / dT through Fc
  if ((fl & FL_FALLOFF) && kl > 0.0) {  // as law(): k0 = 0 has no rate; Pr = inf the high-pressure rate 0
    const double rr = kl * nd / exp(q[6] + q[7] * lT - q[8] * iT);
    ph = 1.0 / (1.0 + rr);
    pc = isfinite(rr) ? rr / (1.0 + rr) : 1.0;
    if ((fl & FL_TROE) && rr > 0.0 && isfinite(rr)) {
      const double ea = (1.0 - q[9]) * exp(-Tc / q[10]), eb = q[9] * exp(-Tc / q[11]), ec = exp(-q[12] / Tc);
      const double fc = ea + eb + ec, f = log10(fc), z = log10(rr), nn = TROE_N0 - TROE_N1 * f;
      const bool vm = (fl & FL_VM11) != 0;
      const double x = vm ? z : z - TROE_C0 - TROE_C1 * f, dd = vm ? nn : nn - TROE_D * x, y = x / dd;
      const double yz = nn / (dd * dd), yf = (vm ? TROE_N1 * x : TROE_N1 * x - TROE_C1 * nn) / (dd * dd);
      const double u = 1.0 / (1.0 + y * y), sz = -2.0 * f * y * yz * u * u, tf = u - 2.0 * f * y * yf * u * u;
      ph += sz;
      pc -= sz;
      // ln F = ln 10 log10 F; df/dT = Fc' / (Fc ln 10)
      fT = tf * (-ea / q[10] - eb / q[11] + ec * q[12] * iT * iT) / fc;
    } else if ((fl & FL_TROE) && !(fl & FL_VM11) && rr == 0.0) {
      // Pr = 0: law()'s limit F = Fc^s, s = d^2 / (1 + d^2), no Pr dependence; d ln F / dT = s Fc' / Fc (VM11: F = 1)
      const double ea = (1.0 - q[9]) * exp(-Tc / q[10]), eb = q[9] * exp(-Tc / q[11]), ec = exp(-q[12] / Tc);
      const double s = TROE_D * TROE_D / (1.0 + TROE_D * TROE_D);
      fT = s * (-ea / q[10] - eb / q[11] + ec * q[12] * iT * iT) / (ea + eb + ec);
    }
  }
  D[0] = ph * w0;
  D[1] = ph * w0 * lT;
  D[2] = -ph * w0 * iT;
  D[3] = ph * w1;
  D[4] = ph * w1 * lT;
  D[5] = -ph * w1 * iT;
  D[6] = pc;
  D[7] = pc * lT;
  D[8] = -pc * iT;
  dT = (ph * (w0 * (q[1] + q[2] * iT) + w1 * (q[4] + q[5] * iT)) + pc * (q[7] + q[8] * iT)) * iT + fT;
  dn = -pc + ((fl & FL_THREE_BODY) ? 1.0 : 0.0);  // pc already holds -s: d ln k / d ln nd = -Pr / (1 + Pr) + s
}

// d(G/RT)/dT = -(H/RT)/T of each species into g[0..NS] (g[NS] = 0).
__device__ __forceinline__ void gibbs_dT(double* g, double Tc, const double* __restrict__ nasa, int lane) {
  const double lT = log(Tc), iT = 1.0 / Tc;
  const double* cf = nasa + (Tc < NASA_SPLIT_T ? 0 : NS * 9);
  for (int i = lane; i < NS; i += 32) {
    const double* a = cf + 9 * i;
    const double T2 = Tc * Tc, T3 = T2 * Tc, T4 = T3 * Tc;
    g[i] = -enthalpy_rt(a, Tc, lT, iT, T2, T3, T4) * iT;
  }
  if (lane == 0) g[NS] = 0.0;
  __syncwarp();
}

// Tangents of kf, kr (dk[r], dk[NR + r]; absolute, not of ln k: a photolysis row's kf = J exp(lm) may be 0) for the
// tangents of T (dTc), ln nd (dlnd), the log multipliers (dlmc, may be null), J (dJc, may be null) and the Arrhenius
// table (drc [R, 9], may be null); kf, kr, cw: the cell's rates and collision-limit weights (rates()). With A = ln of
// the pair's forward coefficient before the limit, G = ln kr - A, L = ln nd, the limit adds
// d ln c = -(wA dA + wG dG + wE dL) to both directions (wE = (wA - wG) [forward of 3 molecules] + wG [reverse of 3]).
// size-limit-ok: a tangent per rate-law input beside rates()'s; its one group, the rate tables, would save 2 of 18
__device__ __forceinline__ void rates_tan(double* work, double* dk, double Tc, double nd, double dTc, double dlnd,
                                          const double* lmc, const double* dlmc, const double* dJc,
                                          const double* __restrict__ rc, const double* drc, const int* __restrict__ rf,
                                          const double* __restrict__ nasa, const double* kf, const double* kr,
                                          const double* cw, const Tab& tab, int lane) {
  const double lT = log(Tc), iT = 1.0 / Tc;
  double* gT = work + A_N1;
  gibbs_dT(gT, Tc, nasa, lane);
  for (int r = lane; r < NR; r += 32) {
    const double dl = dlmc ? dlmc[r] : 0.0;
    if (r >= NRT) {
      dk[r] = (dJc ? (lmc ? exp(lmc[r]) : 1.0) * dJc[r - NRT] : 0.0) + kf[r] * dl;
      dk[NR + r] = 0.0;
      continue;
    }
    const int fl = rf[r];
    double dT = 0.0, dn = 0.0, D[9];
    rate_law_partials(rc + RCW * r, fl, Tc, lT, iT, nd, dT, dn, D);
    double dA = dT * dTc + dn * dlnd;
    if (drc)
      for (int j = 0; j < 9; j++) dA += D[j] * drc[9 * r + j];
    const u8* x = tab.rx + 6 * r;
    const double ddg = gT[x[3]] + gT[x[4]] + gT[x[5]] - gT[x[0]] - gT[x[1]] - gT[x[2]];
    const double dG = (ddg + reaction_dmu(fl) * iT) * dTc;
    const double wA = cw[r], wG = cw[NR + r];
    const double wE = collision_limit_nd_weight(fl, wA, wG);
    const double dc = -(wA * dA + wG * dG + wE * dlnd);
    dk[r] = kf[r] * (dA + dc + dl);
    dk[NR + r] = kr[r] * (dA + dG + dc + dl);
  }
  __syncwarp();
}

// y <- M0^-1 y on one warp (the call kernel's msolve, results stored).
__device__ __forceinline__ void solve_m0(const double* M, const double* dinv, double* y, const u16* smap,
                                         const double (&c)[NT], int lane, int myt) {
  double a0[RP], a1 = 0.0;
  msolve(M, dinv, y, smap, c, lane, myt, a0, a1);
  __syncwarp();
  store_solved(y, a0, a1, lane);
  __syncwarp();
}

// The Sherman-Morrison update of a solved stage, then the warp's barrier.
__device__ __forceinline__ void sherman_morrison_step(double* y, const double* u, double c0, int lane) {
  sherman_morrison_update(y, u, c0, lane);
  __syncwarp();
}

// y <- W^-1 y = M0^-1 y + c0 (1 . M0^-1 y) u with the attempt's factors.
__device__ __forceinline__ void solve_w(const double* M, const double* dinv, double* y, const double* u, double c0,
                                        const u16* smap, const double (&c)[NT], int lane, int myt) {
  solve_m0(M, dinv, y, smap, c, lane, myt);
  sherman_morrison_step(y, u, c0, lane);
}

// y <- W^-1 y with the attempt's factors: the sparse LU of M0 and the Sherman-Morrison step, or (DN: the recovery's
// attempt) the dense threshold-pivoted LU of W in Wd with 1 / U_ii at work + A_DINV and the row order at work + A_M (as
// ints).
template <bool DN>
__device__ __forceinline__ void solve_attempt(double* work, const double* Wd, const double (&c)[NT], double c0,
                                              double* y, const u16* smap, int lane, int myt) {
  if (DN)
    dense_solve(Wd, work + A_DINV, (const int*)(work + A_M), y, lane);
  else
    solve_w(work + A_M, work + A_DINV, y, work + A_U, c0, smap, c, lane, myt);
}

// LU of M0 on one warp: the sparse leading columns, then the dense trailing block in registers c and back in its slots
// (transposed solves); 1 / the pivots in dinv.
__device__ __forceinline__ void factor_warp(double* M, double* dinv, double (&c)[NT], const Tab& tab,
                                            const u64* __restrict__ lu, const int* __restrict__ optr, int lane,
                                            int myt) {
  for (int k = 0; k < K0; k++) {
    const int len = tab.blen[k];
    if (len) {
      for (int i = lane; i < len; i += 32) dinv[k + i] = 1.0 / M[tab.diag[k + i]];
      __syncwarp();
    }
    if (!(LPRE(k) | LTAIL(k))) continue;
    const int oa = optr[k], ob = optr[k + 1];
    const double dk = dinv[k];
    scale_column(k, dk, M, tab, lane, 32);
    __syncwarp();
    for (int e = oa + lane; e < ob; e += 32) lu_update(M, __ldg(lu + e));
    __syncwarp();
  }
  if (myt >= 0) {
#pragma unroll
    for (int j = 0; j < NT; j++) c[j] = M[TB0 + myt * NT + j];
  }
  factor_trailing(c, dinv, myt);
  if (myt >= 0) {
#pragma unroll
    for (int j = 0; j < NT; j++) M[TB0 + myt * NT + j] = c[j];
  }
}

// One Rodas3 attempt, its order repeating the call kernel's, whose two warps split what this one warp does in sequence;
// the primal test in test_ad.py holds the two to the same bits. Size ha from N, with the call kernel's arithmetic on
// one warp (its NL lane streams walked W per lane): the LU of M0 in M (leading slots and the trailing block's factors)
// and c (the block), dinv, u = M0^-1 w, c0, K1..K4, Y = the new state; returns the error (the call kernel's: the RMS,
// max(RMS, max-norm) while first, i.e. no step of the call is accepted yet), c0 and the acceptance.
__device__ __forceinline__ Attempt attempt(double* work, double (&c)[NT], double ha, bool first, const Rates& k,
                                           const Tolerance& tol, const Tab& tab, const Streams& st, int lane, int myt) {
  double *M = work + A_M, *dinv = work + A_DINV, *n = work + A_N0, *v = work + A_V, *N = work + A_NN, *Y = work + A_Y,
         *F = work + A_F, *u = work + A_U, *K1 = work + A_K1, *K2 = work + A_K2, *K3 = work + A_K3, *K4 = work + A_K4,
         *part = work + A_PART;
  Attempt a;
  const double igh = 1.0 / (GAMMA * ha), ih = 1.0 / ha;
  const double ntot = rhs1(N, F, u, n, v, part, k.kf, k.kr, k.nd, tab, st.sst, st.sround, lane);
  for (int s = lane; s < NNZ; s += 32) M[s] = 0.0;
  __syncwarp();
#pragma unroll
  for (int m = 0; m < W; m++) jacobian_lane(lane + 32 * m, M, Y, k.kf, n, st.jt, st.jround);
  __syncwarp();
#if NFIX
  if (lane == 0) jacobian_extras(M, Y, st.jt, st.jround);
  __syncwarp();
#endif
  for (int i = lane; i < NS; i += 32) {  // u = w = S v / nd - Jn x; M = M0 = I / (gamma h) - Jn
    u[i] = u[i] + jacobian_times_x(i, M, N, tab) / ntot;
    M[tab.diag[i]] += igh;
  }
  __syncwarp();
  factor_warp(M, dinv, c, tab, st.lu, st.optr, lane, myt);
  for (int i = lane; i < NS; i += 32) K1[i] = F[i];
  __syncwarp();
  solve_m0(M, dinv, K1, tab.smap, c, lane, myt);  // stage 1: K1 and u = M0^-1 w, c0 = 1 / (1 - sum u)
  {
    double a0[RP], a1 = 0.0;
    msolve(M, dinv, u, tab.smap, c, lane, myt, a0, a1);
    a.c0 = sherman_morrison_factor(a0, a1, lane);
    __syncwarp();
    store_solved(u, a0, a1, lane);
    __syncwarp();
  }
  sherman_morrison_step(K1, u, a.c0, lane);
  for (int i = lane; i < NS; i += 32) K2[i] = stage2_rhs(ih, K1[i], F[i]);
  __syncwarp();
  solve_w(M, dinv, K2, u, a.c0, tab.smap, c, lane, myt);
  for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i];
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, k.kf, k.kr, k.nd, tab, st.sst, st.sround, lane);
  for (int i = lane; i < NS; i += 32) K3[i] = stage3_rhs(ih, K1[i], K2[i], F[i]);
  __syncwarp();
  solve_w(M, dinv, K3, u, a.c0, tab.smap, c, lane, myt);
  for (int i = lane; i < NS; i += 32) {
    double y = N[i] + 2.0 * K1[i];
    y = y + K3[i];
    Y[i] = y;
  }
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, k.kf, k.kr, k.nd, tab, st.sst, st.sround, lane);
  for (int i = lane; i < NS; i += 32) K4[i] = stage4_rhs(ih, K1[i], K2[i], K3[i], F[i]);
  __syncwarp();
  solve_w(M, dinv, K4, u, a.c0, tab.smap, c, lane, myt);
  a.en = try_error(N, K1, K3, K4, Y, true, first, tol.rtol, tol.atol, lane, a.ok);
  __syncwarp();
  return a;
}

// One attempt of the recover kernel on one warp, repeating its order, which spreads the right-hand side over the cell's
// warps (test_ad.py holds the two to the same bits): W = I / (gamma h) - Jn - w 1^T
// formed densely in Wd from the call kernel's streams, dense_lu (1 / U_ii in dinv, the row order in pr = work + A_M as
// ints), the stages with dense_solve; K1..K4, Y = the new state; returns the error (as attempt), c0 = 0 and the
// acceptance.
__device__ __forceinline__ Attempt recovery_attempt(double* work, double* Wd, double ha, bool first, const Rates& k,
                                                    const Tolerance& tol, const Tab& tab, const Streams& st, int lane) {
  double *dinv = work + A_DINV, *n = work + A_N0, *v = work + A_V, *N = work + A_NN, *Y = work + A_Y, *F = work + A_F,
         *u = work + A_U, *K1 = work + A_K1, *K2 = work + A_K2, *K3 = work + A_K3, *K4 = work + A_K4,
         *part = work + A_PART;
  int* pr = (int*)(work + A_M);
  Attempt a;
  a.c0 = 0.0;
  const double igh = 1.0 / (GAMMA * ha), ih = 1.0 / ha;
  const double ntot = rhs1(N, F, u, n, v, part, k.kf, k.kr, k.nd, tab, st.sst, st.sround, lane);
  for (int s = lane; s < NS * NS; s += 32) Wd[s] = 0.0;
  __syncwarp();
#pragma unroll
  for (int m = 0; m < W; m++) jacobian_lane_dense(lane + 32 * m, Wd, Y, st.didx, k.kf, n, st.jt, st.jround);
  __syncwarp();
#if NFIX
  if (lane == 0) jacobian_extras_dense(Wd, Y, st.didx, st.jt, st.jround);
  __syncwarp();
#endif
  // u = w = S v / nd - Jn x
  for (int i = lane; i < NS; i += 32) u[i] = u[i] + dense_times_x(i, Wd, N, tab) / ntot;
  __syncwarp();
  form_dense_w(Wd, u, igh, lane, 32);
  for (int i = lane; i < NS; i += 32) K1[i] = F[i];
  __syncwarp();
  dense_lu(Wd, dinv, pr, lane);
  dense_solve(Wd, dinv, pr, K1, lane);
  for (int i = lane; i < NS; i += 32) K2[i] = stage2_rhs(ih, K1[i], F[i]);
  __syncwarp();
  dense_solve(Wd, dinv, pr, K2, lane);
  for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i];
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, k.kf, k.kr, k.nd, tab, st.sst, st.sround, lane);
  for (int i = lane; i < NS; i += 32) K3[i] = stage3_rhs(ih, K1[i], K2[i], F[i]);
  __syncwarp();
  dense_solve(Wd, dinv, pr, K3, lane);
  for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i] + K3[i];
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, k.kf, k.kr, k.nd, tab, st.sst, st.sround, lane);
  for (int i = lane; i < NS; i += 32) K4[i] = stage4_rhs(ih, K1[i], K2[i], K3[i], F[i]);
  __syncwarp();
  dense_solve(Wd, dinv, pr, K4, lane);
  a.en = try_error(N, K1, K3, K4, Y, true, first, tol.rtol, tol.atol, lane, a.ok);
  __syncwarp();
  return a;
}

// An attempt of the call kernel (attempt) or, DN, of the recover kernel (recovery_attempt).
template <bool DN>
__device__ __forceinline__ Attempt run_attempt(double* work, double (&c)[NT], double* Wd, double ha, bool first,
                                               const Rates& k, const Tolerance& tol, const Tab& tab, const Streams& st,
                                               int lane, int myt) {
  if (DN) return recovery_attempt(work, Wd, ha, first, k, tol, tab, st, lane);
  return attempt(work, c, ha, first, k, tol, tab, st, lane, myt);
}

// Step-size factor of the controller after an attempt of size ha with error en (accepted ok; prej, pen, pha: the
// previous attempt was rejected, its error and size), bitwise the call kernel's step_factor, and its partial
// derivatives on the branch taken: dfe = d fac / d en, dfh = / d ha, dfp = / d pen, dfq = / d pha (0 where a limit
// holds or the factor is a constant of its branch). Secant branch: fac = (SECANT_TARGET / en)^(1 / p),
// p = min(pe, SECANT_MAX_ORDER), pe = ln(pen / en) / L, L = ln(pha / ha). Plain branch: fac = STEP_SAFETY
// en^(-1 / SECANT_MAX_ORDER), so d fac / d en = -fac / (SECANT_MAX_ORDER en).
__device__ __forceinline__ double step_factor_partials(double en, bool ok, bool prej, double pen, double pha, double ha,
                                                       double& dfe, double& dfh, double& dfp, double& dfq) {
  const double fac = step_factor(en, ok, prej, pen, pha, ha);
  const bool efin = isfinite(en);
  dfe = 0.0;
  dfh = 0.0;
  dfp = 0.0;
  dfq = 0.0;
  if (!ok && efin && prej && en > 1.0 && pen > 1.0 && pha > ha) {
    const double L = log(pha / ha), pe = log(pen / en) / L;
    if (pe > SECANT_MIN_ORDER) {
      const double p = fmin(pe, SECANT_MAX_ORDER), fs = pow(SECANT_TARGET / en, 1.0 / p);
      if (fs >= SECMIN && fs <= FAC_AFTER_REJECT) {
        dfe = -fs / (p * en);                                        // through the base, p held
        if (pe <= SECANT_MAX_ORDER) {                                // through p = pe
          const double g = -fs * log(SECANT_TARGET / en) / (p * p);  // d fac / d p
          dfe -= g / (en * L);
          dfp = g / (pen * L);
          dfq = -g * pe / (pha * L);
          dfh = g * pe / (ha * L);
        }
      }
    }
  } else if (efin && en >= ERR_FLOOR) {
    const double f0 = STEP_SAFETY / cbrt(en);
    if (f0 >= (ok ? FACACC : FACREJ) && f0 <= FACMAX && (ok || f0 <= FAC_AFTER_REJECT))
      dfe = -f0 / (SECANT_MAX_ORDER * en);
  }
  return fac;
}

// The recover kernel's test of a recomputed cell's result: its flag (redo), count and done; returns on every lane
// whether it still fails.
__device__ __forceinline__ bool recovery_fails(const double* __restrict__ x0, double* xout, int* doneo, int* redo,
                                               int* count, int dn, double nok, const double* __restrict__ atoms,
                                               double atol, int cell, int lane) {
  __syncwarp();
  int bad = 0;
  if (lane == 0) {
    bad = check_cell(x0, xout, dn, nok, atoms, atol, cell);
    if (bad) doneo[cell] = 0;  // done: finished and passing the guard
    redo[cell] = bad ? 2 : 1;
    atomicAdd(count + (bad ? COUNT_STILL_FAILED : COUNT_RECOVERED), 1);
  }
  return __shfl_sync(FULL, bad, 0) != 0;
}

// The warp's next cell from the queue.
template <bool DN>
__device__ __forceinline__ int next_queued(int* count, int lane) {
  int nxt = 0;
  if (lane == 0) nxt = gridDim.x * WPB + atomicAdd(count + (DN ? COUNT_RECOVERY_QUEUE : COUNT_QUEUE), 1);
  const int q = __shfl_sync(FULL, nxt, 0);
  __syncwarp();
  return q;
}

// One attempt with acceptance and step control (the call kernel's loop body; DN: the recover kernel's) from the step
// state s, which it advances; returns the attempt (its stages and factors stay in place).
template <bool DN>
__device__ __forceinline__ Attempt forward_step(double* work, double (&c)[NT], double* Wd, StepState& s, double dt,
                                                const Rates& k, const Tolerance& tol, const Tab& tab, const Streams& st,
                                                int lane, int myt) {
  double *N = work + A_NN, *Y = work + A_Y;
  const double ha = fmin(s.h, dt - s.t);
  const Attempt a = run_attempt<DN>(work, c, Wd, ha, s.t == 0.0, k, tol, tab, st, lane, myt);
  accept_try(s, N, Y, a.ok, ha, lane, 32);
  __syncwarp();
  next_try(s, step_factor(a.en, a.ok, s.prej, s.pen, s.pha, ha), a.ok, a.en, ha);
  return a;
}
