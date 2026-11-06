// What the tangent and adjoint kernels share: the call's arithmetic on one warp, rates and their derivatives.

#define CAPR (2 * CAP)  // attempts stored per recovered cell (REC_STEPS x max_steps <= CAPR^2)
#define REC \
  (NS + 6)                                         // a stored attempt: N, h, t, accepted, then the previous attempt's
                                                   // error, size and rejection (the step control's history)
__device__ const unsigned char PX[] = {PX_SLOTS};  // photolysis branches' product slots, 3 per branch (padding NS)

__device__ __forceinline__ double sgn(double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0); }

// The call kernel's rhs on one warp: F = (n_tot / nd) S v(n) at Y, u0 = S v / nd unless null; each lane walks W of the
// call kernel's NL lane streams, so every sum is the call kernel's. Returns n_tot.
// size-limit-ok: inline helpers could move which operations fuse, so a split must first prove equal bits
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
    const int na = VSA(qq), nb = VSB(qq);
#pragma unroll
    for (int m = 0; m < W; m++) {
      const int r = lane + 32 * m + NL * qq;
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
  }
  __syncwarp();
#pragma unroll
  for (int m = 0; m < W; m++) {
    const int cl = lane + 32 * m;
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
  __syncwarp();
  for (int i = lane; i < NS; i += 32) {
    double s = part[i];
    for (int e = tab.xp[i]; e < tab.xp[i + 1]; e++) s += part[NS + e];
    F[i] = s * ixf;
    if (u0) u0[i] = s / nd;
  }
  __syncwarp();
  return ntot;
}

// o1 = S w1, o2 = S w2 in one pass of the S v lane streams (part holds 2 NPART doubles).
__device__ __forceinline__ void svs2(const double* w1, const double* w2, double* o1, double* o2, double* part,
                                     const Tab& tab, const unsigned* __restrict__ sst, int sround, int lane) {
  for (int i = lane; i < 2 * NPART; i += 32) part[i] = 0.0;
  __syncwarp();
  for (int m = 0; m < W; m++) {
    const int cl = lane + 32 * m;
    double a1 = 0.0, a2 = 0.0;
    for (int q = 0; q < sround; q++) {
      const unsigned x = __ldg(sst + q * NL + cl);
      if (x >> 31) {
        const int r = x & SV_REACTION_MASK;
        const double cf = (((x >> SV_DOUBLE) & 1u) ? 2.0 : 1.0) * (((x >> SV_SIGN) & 1u) ? -1.0 : 1.0);
        a1 += cf * w1[r];
        a2 += cf * w2[r];
        if ((x >> SV_END) & 1u) {
          const int d = (x >> SV_PIECE) & SV_PIECE_MASK;
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

// The collision limit's derivative weights of a reaction from its pair before the limit (COLL's q): wA = 1 if
// q > K_COLL, 1/2 at q = K_COLL (the tie of the min), else 0; wG = wA v, v = the reverse direction's share of the
// larger (1/2 at a tie; a direction of fewer than two molecules does not count).
__device__ __forceinline__ void cap_w(int fl, double kf, double kr, double nd, double& wA, double& wG) {
  const int mf = (fl >> FL_FORWARD_SHIFT) & 3, mr = (fl >> FL_REVERSE_SHIFT) & 3;
  const double qf = mf >= 2 ? kf * (mf == 3 ? nd : 1.0) : 0.0, qr = mr >= 2 ? kr * (mr == 3 ? nd : 1.0) : 0.0;
  const double q = fmax(qf, qr), weight = q > KCOLL ? 1.0 : (q == KCOLL ? 0.5 : 0.0);
  wA = weight;
  wG = weight * (qr > qf ? 1.0 : (qr < qf ? 0.0 : 0.5));
}

// Rate coefficients of the cell (the call kernel's code: the law, the reverse from Gibbs, the collision limit;
// photolysis rows kf = J, kr = 0), times exp(lm) when per-cell log multipliers are given; cw[r], cw[NR + r] = the
// collision limit's weights (cap_w).
__device__ __forceinline__ void rates(double* work, double* kf, double* kr, double* cw, double Tc, double nd,
                                      const double* lmc, const double* Jc, const double* __restrict__ rc,
                                      const int* __restrict__ rf, const double* __restrict__ nasa, const Tab& tab,
                                      int lane) {
  const double lT = log(Tc), iT = 1.0 / Tc, lKT = log(KBOLTZ * Tc / P0BAR);
  double* g = work + A_V;  // G / RT of each species
  const double* cf = nasa + (Tc < NASA_SPLIT_T ? 0 : NS * 9);
  for (int i = lane; i < NS; i += 32) {
    const double* a = cf + 9 * i;
    const double T2 = Tc * Tc, T3 = T2 * Tc, T4 = T3 * Tc;
    const double hh = -a[0] * iT * iT + a[1] * lT * iT + a[2] + a[3] * Tc / 2 + a[4] * T2 / 3 + a[5] * T3 / 4 +
                      a[6] * T4 / 5 + a[7] * iT;
    const double ss =
        -a[0] * iT * iT / 2 - a[1] * iT + a[2] * lT + a[3] * Tc + a[4] * T2 / 2 + a[5] * T3 / 3 + a[6] * T4 / 4 + a[8];
    g[i] = hh - ss;
  }
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
    const double k = law(rc + RCW * r, fl, Tc, lT, iT, nd);
    const u8* x = tab.rx + 6 * r;
    const double dg = g[x[3]] + g[x[4]] + g[x[5]] - g[x[0]] - g[x[1]] - g[x[2]];
    const double c3 = (fl & FL_THREE_BODY) ? nd : 1.0;
    kf[r] = k * c3;
    kr[r] = REVERSE_RATE;
    cap_w(fl, kf[r], kr[r], nd, cw[r], cw[NR + r]);
    COLL(fl, kf[r], kr[r], nd);
    if (lmc) {
      const double e = exp(lmc[r]);
      kf[r] *= e;
      kr[r] *= e;
    }
  }
  __syncwarp();
}

// Partial derivatives of ln kf of a thermal reaction (rate law q = its rc row, flags fl; the collision limit not
// included) at (T, nd): dT = d/dT, dn = d/d ln nd, D[j] = d/d q[j] for the nine Arrhenius columns (the Troe parameters
// and the efficiency are not differentiated). ln kr adds d(sum G/RT, products - reactants)/dT + dmu / T to dT.
// Troe factor: log10 F = f / (1 + y^2), f = log10 Fc(T), z = log10 Pr, N = 0.75 - 1.27 f; vm11 y = z / N, standard
// x = z - 0.4 - 0.67 f, y = x / (N - 0.14 x); s = d log10 F / dz enters d ln k / d ln k_low, d ln k_inf, d ln nd.
__device__ __forceinline__ void kpart(const double* q, int fl, double Tc, double lT, double iT, double nd, double& dT,
                                      double& dn, double* D) {
  const double e0 = exp(q[0] + q[1] * lT - q[2] * iT),
               e1 = (fl & FL_SECOND_K0) ? exp(q[3] + q[4] * lT - q[5] * iT) : 0.0;
  const double kl = e0 + e1, w0 = kl > 0.0 ? e0 / kl : 0.0, w1 = kl > 0.0 ? e1 / kl : 0.0;
  double ph = 1.0, pc = 0.0, fT = 0.0;  // d ln k / d ln k_low, d ln k / d ln k_inf, d ln F / dT through Fc
  if (fl & FL_FALLOFF) {
    const double rr = kl * nd / exp(q[6] + q[7] * lT - q[8] * iT);
    ph = 1.0 / (1.0 + rr);
    pc = rr / (1.0 + rr);
    if ((fl & FL_TROE) && rr > 0.0) {
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
    const double hh = -a[0] * iT * iT + a[1] * lT * iT + a[2] + a[3] * Tc / 2 + a[4] * T2 / 3 + a[5] * T3 / 4 +
                      a[6] * T4 / 5 + a[7] * iT;
    g[i] = -hh * iT;
  }
  if (lane == 0) g[NS] = 0.0;
  __syncwarp();
}

// Tangents of kf, kr (dk[r], dk[NR + r]; absolute, not of ln k: a photolysis row's kf = J exp(lm) may be 0) for the
// tangents of T (dTc), ln nd (dlnd), the log multipliers (dlmc, may be null), J (dJc, may be null) and the Arrhenius
// table (drc [R, 9], may be null); kf, kr, cw: the cell's rates and collision-limit weights (rates()). With A = ln of
// the pair's forward coefficient before the limit, G = ln kr - A, L = ln nd, the limit adds
// d ln c = -(wA dA + wG dG + wE dL) to both directions (wE = (wA - wG) [forward of 3 molecules] + wG [reverse of 3]).
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
    double dT, dn, D[9];
    kpart(rc + RCW * r, fl, Tc, lT, iT, nd, dT, dn, D);
    double dA = dT * dTc + dn * dlnd;
    if (drc)
      for (int j = 0; j < 9; j++) dA += D[j] * drc[9 * r + j];
    const u8* x = tab.rx + 6 * r;
    const double ddg = gT[x[3]] + gT[x[4]] + gT[x[5]] - gT[x[0]] - gT[x[1]] - gT[x[2]];
    const double dG = (ddg + (double)(((fl >> FL_DMU_SHIFT) & 7) - 4) * iT) * dTc;
    const double wA = cw[r], wG = cw[NR + r];
    const double wE = (wA - wG) * (((fl >> FL_FORWARD_SHIFT) & 3) == 3 ? 1.0 : 0.0) +
                      wG * (((fl >> FL_REVERSE_SHIFT) & 3) == 3 ? 1.0 : 0.0);
    const double dc = -(wA * dA + wG * dG + wE * dlnd);
    dk[r] = kf[r] * (dA + dc + dl);
    dk[NR + r] = kr[r] * (dA + dG + dc + dl);
  }
  __syncwarp();
}

// y <- M0^-1 y on one warp (the call kernel's msolve, results stored).
__device__ __forceinline__ void msol(const double* M, const double* dinv, double* y, const u16* smap,
                                     const double (&c)[NT], int lane, int myt) {
  double a0[RP], a1;
  msolve(M, dinv, y, smap, c, lane, myt, a0, a1);
  __syncwarp();
#pragma unroll
  for (int p = 0; p < RP; p++)
    if (lane + 32 * p < K0) y[lane + 32 * p] = a0[p];
  if (lane < NT) y[K0 + lane] = a1;
  __syncwarp();
}

// The Sherman-Morrison update of a solved stage: y <- y + c0 (sum y) u (the call kernel's arithmetic).
__device__ __forceinline__ void smup(double* y, const double* u, double c0, int lane) {
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
  __syncwarp();
}

// y <- W^-1 y = M0^-1 y + c0 (1 . M0^-1 y) u with the attempt's factors.
__device__ __forceinline__ void wsolve(const double* M, const double* dinv, double* y, const double* u, double c0,
                                       const u16* smap, const double (&c)[NT], int lane, int myt) {
  msol(M, dinv, y, smap, c, lane, myt);
  smup(y, u, c0, lane);
}

// y <- W^-1 y with the attempt's factors: the sparse LU of M0 and the Sherman-Morrison step, or (DN: the recovery's
// attempt) the dense threshold-pivoted LU of W in Wd with 1 / U_ii at S + A_DINV and the row order at S + A_M (as
// ints).
template <bool DN>
__device__ __forceinline__ void xsolve(double* work, const double* Wd, const double (&c)[NT], double c0, double* y,
                                       const u16* smap, int lane, int myt) {
  if (DN)
    dense_solve(Wd, work + A_DINV, (const int*)(work + A_M), y, lane);
  else
    wsolve(work + A_M, work + A_DINV, y, work + A_U, c0, smap, c, lane, myt);
}

// One Rodas3 attempt of size ha from N with the call kernel's arithmetic on one warp (its NL lane streams walked W per
// lane): the LU of M0 in M (leading slots and the trailing block's factors) and c (the block), dinv, u = M0^-1 w, c0,
// K1..K4, Y = the new state; ok = accepted; returns the error (the call kernel's: the RMS, max(RMS, max-norm) while
// first, i.e. no step of the call is accepted yet).
// size-limit-ok: inline helpers could move which operations fuse, so a split must first prove equal bits
__device__ __forceinline__ double attempt(double* work, double (&c)[NT], double ha, bool first, const double* kf,
                                          const double* kr, double nd, double rtol, double atol, double& c0, bool& ok,
                                          const Tab& tab, const unsigned* __restrict__ sst, const u64* __restrict__ jt,
                                          const u64* __restrict__ lu, const int* __restrict__ optr, int sround,
                                          int jround, int lane, int myt) {
  double *M = work + A_M, *dinv = work + A_DINV, *n = work + A_N0, *v = work + A_V, *N = work + A_NN, *Y = work + A_Y,
         *F = work + A_F, *u = work + A_U, *K1 = work + A_K1, *K2 = work + A_K2, *K3 = work + A_K3, *K4 = work + A_K4,
         *part = work + A_PART;
  const double igh = 1.0 / (GAMMA * ha), ih = 1.0 / ha;
  const double ntot = rhs1(N, F, u, n, v, part, kf, kr, nd, tab, sst, sround, lane);
  for (int s = lane; s < NNZ; s += 32) M[s] = 0.0;
  __syncwarp();
  {  // M = -Jn: the call kernel's lane streams of 32-bit terms
    const unsigned* jw = (const unsigned*)jt;
#pragma unroll
    for (int m = 0; m < W; m++) {
      const int cl = lane + 32 * m;
      const u16* jsl = (const u16*)(jw + jround * NL) + cl * JSMAX;
      double acc = 0.0;
      int ne = 0;
#pragma unroll 4
      for (int q = 0; q < jround; q++) {
        const unsigned word = __ldg(jw + q * NL + cl);
        const double pr =
            kf[word & J_RATE_MASK] * n[(word >> J_A) & J_SPECIES_MASK] * n[(word >> J_B) & J_SPECIES_MASK];
        const double cf =
            ((word >> J_DOUBLE) & 1u) ? (((word >> J_SIGN) & 1u) ? -2.0 : 2.0) : (((word >> J_SIGN) & 1u) ? -1.0 : 1.0);
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
  }
  __syncwarp();
#if NFIX
  if (lane == 0) {
    const u16* fx = (const u16*)((const unsigned*)jt + jround * NL) + NL * JSMAX;
    for (int e = 0; e < NFIX; e++) M[__ldg(fx + e)] += Y[e];
  }
  __syncwarp();
#endif
  for (int i = lane; i < NS; i += 32) {  // u = w = S v / nd - Jn x; M = M0 = I / (gamma h) - Jn
    double acc = 0.0;
    const int s1 = tab.rowptr[i + 1];
    for (int s = tab.rowptr[i]; s < s1; s++) acc += M[s] * N[tab.colof[s]];
    if (i >= K0)
      for (int j = 0; j < NT; j++) acc += M[TB0 + (i - K0) * NT + j] * N[K0 + j];
    u[i] = u[i] + acc / ntot;
    M[tab.diag[i]] += igh;
  }
  __syncwarp();
  for (int k = 0; k < K0; k++) {  // LU of the sparse leading columns
    const int len = tab.blen[k];
    if (len) {
      for (int i = lane; i < len; i += 32) dinv[k + i] = 1.0 / M[tab.diag[k + i]];
      __syncwarp();
    }
    if (!(LPRE(k) | LTAIL(k))) continue;
    const int oa = optr[k], ob = optr[k + 1];
    const double dk = dinv[k];
    for (int i = lane; i < NS; i += 32) {
      const unsigned q = tab.smap[i * NS + k];
      if (i > k && q != NO_SLOT) M[q] *= dk;
    }
    __syncwarp();
    for (int e = oa + lane; e < ob; e += 32) {
      const u64 op = __ldg(lu + e);
      M[op & LU_SLOT_MASK] -= M[(op >> LU_B) & LU_SLOT_MASK] * M[(op >> LU_C) & LU_SLOT_MASK];
    }
    __syncwarp();
  }
  if (myt >= 0) {  // dense trailing block in registers; 1 / its diagonal into dinv
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
  if (myt >= 0) {  // the factors back into the block's slots (transposed solves)
#pragma unroll
    for (int j = 0; j < NT; j++) M[TB0 + myt * NT + j] = c[j];
  }
  for (int i = lane; i < NS; i += 32) K1[i] = F[i];
  __syncwarp();
  msol(M, dinv, K1, tab.smap, c, lane, myt);  // stage 1: K1 and u = M0^-1 w, c0 = 1 / (1 - sum u)
  {
    const bool ht = lane < NT;
    double a0[RP], a1;
    msolve(M, dinv, u, tab.smap, c, lane, myt, a0, a1);
    double sb = lane < K0 ? a0[0] : 0.0;
#pragma unroll
    for (int p = 1; p < RP; p++) sb += lane + 32 * p < K0 ? a0[p] : 0.0;
    c0 = 1.0 / (1.0 - wsum(sb + (ht ? a1 : 0.0)));
    __syncwarp();
#pragma unroll
    for (int p = 0; p < RP; p++)
      if (lane + 32 * p < K0) u[lane + 32 * p] = a0[p];
    if (ht) u[K0 + lane] = a1;
    __syncwarp();
  }
  smup(K1, u, c0, lane);
  for (int i = lane; i < NS; i += 32) K2[i] = fma(RODAS_C21 * ih, K1[i], F[i]);
  __syncwarp();
  wsolve(M, dinv, K2, u, c0, tab.smap, c, lane, myt);
  for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i];
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, kf, kr, nd, tab, sst, sround, lane);
  for (int i = lane; i < NS; i += 32) K3[i] = fma(ih, K1[i] - K2[i], F[i]);
  __syncwarp();
  wsolve(M, dinv, K3, u, c0, tab.smap, c, lane, myt);
  for (int i = lane; i < NS; i += 32) {
    double y = N[i] + 2.0 * K1[i];
    y = y + K3[i];
    Y[i] = y;
  }
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, kf, kr, nd, tab, sst, sround, lane);
  for (int i = lane; i < NS; i += 32) K4[i] = fma(ih, fma(-RODAS_C43, K3[i], K1[i] - K2[i]), F[i]);
  __syncwarp();
  wsolve(M, dinv, K4, u, c0, tab.smap, c, lane, myt);
  double e2 = 0.0, em = 0.0;
  bool nn = true;
  for (int i = lane; i < NS; i += 32) {
    const double y = N[i] + 2.0 * K1[i] + K3[i] + K4[i];
    Y[i] = y;
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
  __syncwarp();
  return en;
}

// One attempt of the recover kernel (kincuda) on one warp, its arithmetic bitwise: W = I / (gamma h) - Jn - w 1^T
// formed densely in Wd from the call kernel's streams, dense_lu (1 / U_ii in dinv, the row order in pr = S + A_M as
// ints), the stages with dense_solve; K1..K4, Y = the new state; ok = accepted; returns the error (as attempt).
// size-limit-ok: inline helpers could move which operations fuse, so a split must first prove equal bits
__device__ __forceinline__ double rattempt(double* work, double* Wd, double ha, bool first, const double* kf,
                                           const double* kr, double nd, double rtol, double atol, bool& ok,
                                           const Tab& tab, const unsigned* __restrict__ sst, const u64* __restrict__ jt,
                                           const u16* __restrict__ didx, int sround, int jround, int lane) {
  double *dinv = work + A_DINV, *n = work + A_N0, *v = work + A_V, *N = work + A_NN, *Y = work + A_Y, *F = work + A_F,
         *u = work + A_U, *K1 = work + A_K1, *K2 = work + A_K2, *K3 = work + A_K3, *K4 = work + A_K4,
         *part = work + A_PART;
  int* pr = (int*)(work + A_M);
  const double igh = 1.0 / (GAMMA * ha), ih = 1.0 / ha;
  const double ntot = rhs1(N, F, u, n, v, part, kf, kr, nd, tab, sst, sround, lane);
  for (int s = lane; s < NS * NS; s += 32) Wd[s] = 0.0;
  __syncwarp();
  {  // W = -Jn: the call kernel's term streams, written to the dense positions of the slots
    const unsigned* jw = (const unsigned*)jt;
#pragma unroll
    for (int m = 0; m < W; m++) {
      const int cl = lane + 32 * m;
      const u16* jsl = (const u16*)(jw + jround * NL) + cl * JSMAX;
      double acc = 0.0;
      int ne = 0;
#pragma unroll 4
      for (int qq = 0; qq < jround; qq++) {
        const unsigned word = __ldg(jw + qq * NL + cl);
        const double pq =
            kf[word & J_RATE_MASK] * n[(word >> J_A) & J_SPECIES_MASK] * n[(word >> J_B) & J_SPECIES_MASK];
        const double cf =
            ((word >> J_DOUBLE) & 1u) ? (((word >> J_SIGN) & 1u) ? -2.0 : 2.0) : (((word >> J_SIGN) & 1u) ? -1.0 : 1.0);
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
  }
  __syncwarp();
#if NFIX
  if (lane == 0) {  // the extra pieces of long slots, in order
    const u16* fx = (const u16*)((const unsigned*)jt + jround * NL) + NL * JSMAX;
    for (int e = 0; e < NFIX; e++) Wd[__ldg(didx + __ldg(fx + e))] += Y[e];
  }
  __syncwarp();
#endif
  for (int i = lane; i < NS; i += 32) {  // u = w = S v / nd - Jn x (the call kernel's sums, columns in order)
    double acc = 0.0;
    for (int j = 0; j < NS; j++)
      if (tab.smap[i * NS + j] != NO_SLOT) acc += Wd[i * NS + j] * N[j];
    u[i] = u[i] + acc / ntot;
  }
  __syncwarp();
  for (int s = lane; s < NS * NS; s += 32) {  // W = I / (gamma h) - Jn - w 1^T
    const int i = s / NS;
    Wd[s] = (s == i * (NS + 1) ? Wd[s] + igh : Wd[s]) - u[i];
  }
  for (int i = lane; i < NS; i += 32) K1[i] = F[i];
  __syncwarp();
  dense_lu(Wd, dinv, pr, lane);
  dense_solve(Wd, dinv, pr, K1, lane);
  for (int i = lane; i < NS; i += 32) K2[i] = fma(RODAS_C21 * ih, K1[i], F[i]);
  __syncwarp();
  dense_solve(Wd, dinv, pr, K2, lane);
  for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i];
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, kf, kr, nd, tab, sst, sround, lane);
  for (int i = lane; i < NS; i += 32) K3[i] = fma(ih, K1[i] - K2[i], F[i]);
  __syncwarp();
  dense_solve(Wd, dinv, pr, K3, lane);
  for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i] + K3[i];
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, kf, kr, nd, tab, sst, sround, lane);
  for (int i = lane; i < NS; i += 32) K4[i] = fma(ih, fma(-RODAS_C43, K3[i], K1[i] - K2[i]), F[i]);
  __syncwarp();
  dense_solve(Wd, dinv, pr, K4, lane);
  double e2 = 0.0, em = 0.0;
  bool nn = true;
  for (int i = lane; i < NS; i += 32) {
    const double y = N[i] + 2.0 * K1[i] + K3[i] + K4[i];
    Y[i] = y;
    const double sc = atol + rtol * fmax(fabs(N[i]), fabs(y));
    const double q4 = K4[i] / sc;
    e2 += q4 * q4;
    em = fmax(em, fabs(q4));
    nn = nn && (y >= -atol);
  }
  const double er = sqrt(wsum(e2) / NS), emx = wmax(em);
  const double en = (first && isfinite(er)) ? fmax(er, emx) : er;
  const bool nonneg = __all_sync(FULL, nn), efin = isfinite(en);
  ok = (en <= 1.0) && nonneg && efin;
  __syncwarp();
  return en;
}

// An attempt of the call kernel (attempt) or, DN, of the recover kernel (rattempt; c0 = 0, unused).
template <bool DN>
__device__ __forceinline__ double xattempt(double* work, double (&c)[NT], double* Wd, double ha, bool first,
                                           const double* kf, const double* kr, double nd, double rtol, double atol,
                                           double& c0, bool& ok, const Tab& tab, const unsigned* __restrict__ sst,
                                           const u64* __restrict__ jt, const u64* __restrict__ lu,
                                           const int* __restrict__ optr, const u16* __restrict__ didx, int sround,
                                           int jround, int lane, int myt) {
  if (DN) {
    c0 = 0.0;
    return rattempt(work, Wd, ha, first, kf, kr, nd, rtol, atol, ok, tab, sst, jt, didx, sround, jround, lane);
  }
  return attempt(work, c, ha, first, kf, kr, nd, rtol, atol, c0, ok, tab, sst, jt, lu, optr, sround, jround, lane, myt);
}

// Step-size factor of the controller after an attempt of size ha with error en (accepted ok; prej, pen, pha: the
// previous attempt was rejected, its error and size), bitwise the call kernel's step_factor, and its partial
// derivatives on the branch taken: dfe = d fac / d en, dfh = / d ha, dfp = / d pen, dfq = / d pha (0 where a limit
// holds or the factor is a constant of its branch). Secant branch: fac = (0.5 / en)^(1 / p), p = min(pe, 3),
// pe = ln(pen / en) / L, L = ln(pha / ha).
__device__ __forceinline__ double factor(double en, bool ok, bool prej, double pen, double pha, double ha, double& dfe,
                                         double& dfh, double& dfp, double& dfq) {
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
    if (f0 >= (ok ? FACACC : FACREJ) && f0 <= FACMAX && (ok || f0 <= FAC_AFTER_REJECT)) dfe = -f0 / (3.0 * en);
  }
  return fac;
}

// One attempt with acceptance and step control (the call kernel's loop body; DN: the recover kernel's); returns ok, en
// = its error, c0 = its Sherman-Morrison factor (its stages and factors stay in place).
template <bool DN>
__device__ __forceinline__ bool fstep(double* work, double (&c)[NT], double* Wd, double& h, double& t, double& nok,
                                      bool& prej, double& pen, double& pha, double dt, const double* kf,
                                      const double* kr, double nd, double rtol, double atol, double& en, double& c0,
                                      const Tab& tab, const unsigned* __restrict__ sst, const u64* __restrict__ jt,
                                      const u64* __restrict__ lu, const int* __restrict__ optr,
                                      const u16* __restrict__ didx, int sround, int jround, int lane, int myt) {
  double *N = work + A_NN, *Y = work + A_Y;
  const double ha = fmin(h, dt - t);
  bool ok;
  en = xattempt<DN>(work, c, Wd, ha, t == 0.0, kf, kr, nd, rtol, atol, c0, ok, tab, sst, jt, lu, optr, didx, sround,
                    jround, lane, myt);
  if (ok) {
    for (int i = lane; i < NS; i += 32) N[i] = fmax(Y[i], 0.0);
    t += ha;
    nok += 1.0;
  }
  __syncwarp();
  h = ha * step_factor(en, ok, prej, pen, pha, ha);
  prej = !ok;
  pen = en;
  pha = ha;
  return ok;
}
