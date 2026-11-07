// The adjoint kernels: the call and the cotangents of its inputs.

// solve_w_transposed's U^T: y_j -= U_kj w_k (j > k) over the sparse rows, then the dense rows.
__device__ __forceinline__ void transposed_upper(const double* M, const u16* smap, const bool (&hp)[RP],
                                                 const double (&d0)[RP], double d1, int lane, double (&a0)[RP],
                                                 double& a1) {
  const bool ht = lane < NT;
#pragma unroll
  for (int k = 0; k < K0; k++) {
    if (!((UROW >> k) & 1ull)) continue;
    const double wk = __shfl_sync(FULL, a0[k >> 5] * d0[k >> 5], k & 31);
#pragma unroll
    for (int p = 0; p < RP; p++) {
      const int j = lane + 32 * p;
      if (hp[p] && j > k) {
        const unsigned s = smap[k * NS + j];
        if (s != NO_SLOT) a0[p] -= M[s] * wk;
      }
    }
    if (ht) {
      const unsigned s = smap[k * NS + K0 + lane];
      if (s != NO_SLOT) a1 -= M[s] * wk;
    }
  }
#pragma unroll
  for (int k = 0; k < NT - 1; k++) {
    const double wk = __shfl_sync(FULL, a1 * d1, k);
    if (ht && lane > k) a1 -= M[smap[(K0 + k) * NS + K0 + lane]] * wk;
  }
}

// solve_w_transposed's L^T: w_j -= L_kj x_k (j < k) over the dense rows, then the sparse rows.
__device__ __forceinline__ void transposed_lower(const double* M, const u16* smap, const bool (&hp)[RP], int lane,
                                                 double (&a0)[RP], double& a1) {
  const bool ht = lane < NT;
#pragma unroll
  for (int k = NT - 1; k >= 0; k--) {
    const double xk = __shfl_sync(FULL, a1, k);
    if (ht && lane < k) a1 -= M[smap[(K0 + k) * NS + K0 + lane]] * xk;
#pragma unroll
    for (int p = 0; p < RP; p++)
      if (hp[p]) {
        const unsigned s = smap[(K0 + k) * NS + lane + 32 * p];
        if (s != NO_SLOT) a0[p] -= M[s] * xk;
      }
  }
#pragma unroll
  for (int k = K0 - 1; k >= 0; k--) {
    if (!((LROW >> k) & 1ull)) continue;
    const double xk = __shfl_sync(FULL, a0[k >> 5], k & 31);
#pragma unroll
    for (int p = 0; p < RP; p++) {
      const int j = lane + 32 * p;
      if (hp[p] && j < k) {
        const unsigned s = smap[k * NS + j];
        if (s != NO_SLOT) a0[p] -= M[s] * xk;
      }
    }
  }
}

// y <- W^-T y = M0^-T (y + c0 (u . y) 1) (W = M0 - w 1^T, u = M0^-1 w): U^T then L^T with every factor read from M (the
// dense block's factors written back), lane l holding the leading entries l + 32 p (< K0) and K0 + l.
__device__ __forceinline__ void solve_w_transposed(const double* M, const double* dinv, double* y, const double* u,
                                                   double c0, const u16* smap, int lane) {
  const bool ht = lane < NT;
  bool hp[RP];
  double a0[RP], d0[RP];
  double sl = 0.0;
#pragma unroll
  for (int p = 0; p < RP; p++) {
    hp[p] = lane + 32 * p < K0;
    a0[p] = hp[p] ? y[lane + 32 * p] : 0.0;
    d0[p] = hp[p] ? dinv[lane + 32 * p] : 0.0;
    sl += hp[p] ? u[lane + 32 * p] * a0[p] : 0.0;
  }
  double a1 = ht ? y[K0 + lane] : 0.0;
  const double al = c0 * wsum(sl + (ht ? u[K0 + lane] * a1 : 0.0));
#pragma unroll
  for (int p = 0; p < RP; p++)
    if (hp[p]) a0[p] += al;
  if (ht) a1 += al;
  const double d1 = ht ? dinv[K0 + lane] : 0.0;
  transposed_upper(M, smap, hp, d0, d1, lane, a0, a1);
#pragma unroll
  for (int p = 0; p < RP; p++) a0[p] *= d0[p];
  a1 *= d1;
  transposed_lower(M, smap, hp, lane, a0, a1);
  __syncwarp();
#pragma unroll
  for (int p = 0; p < RP; p++)
    if (hp[p]) y[lane + 32 * p] = a0[p];
  if (ht) y[K0 + lane] = a1;
  __syncwarp();
}

// y <- W^-T y with dense_lu's factors (W = P^T L U, P the row order pr): U^T, then L^T, then the rows back to their
// original order. Lane l holds rows l + 32 p.
__device__ __forceinline__ void dense_solve_transposed(const double* A, const double* dinv, const int* pr, double* y,
                                                       int lane) {
  double a[RPD];
#pragma unroll
  for (int p = 0; p < RPD; p++) a[p] = lane + 32 * p < NS ? y[lane + 32 * p] : 0.0;
#pragma unroll
  for (int k = 0; k < NS - 1; k++) {  // U^T: y_i -= U_ki w_k (i > k), w_k = y_k / U_kk
    const double wk = __shfl_sync(FULL, a[k >> 5] * dinv[k], k & 31);
#pragma unroll
    for (int p = k >> 5; p < RPD; p++) {
      const int i = lane + 32 * p;
      if (i > k && i < NS) a[p] -= A[k * NS + i] * wk;
    }
  }
#pragma unroll
  for (int p = 0; p < RPD; p++)
    if (lane + 32 * p < NS) a[p] *= dinv[lane + 32 * p];
#pragma unroll
  for (int k = NS - 1; k > 0; k--) {  // unit L^T: y_i -= L_ki y_k (i < k)
    const double xk = __shfl_sync(FULL, a[k >> 5], k & 31);
#pragma unroll
    for (int p = 0; p <= (k >> 5); p++) {
      const int i = lane + 32 * p;
      if (i < k) a[p] -= A[k * NS + i] * xk;
    }
  }
  __syncwarp();
#pragma unroll
  for (int p = 0; p < RPD; p++)
    if (lane + 32 * p < NS) y[pr[lane + 32 * p]] = a[p];
  __syncwarp();
}

// y <- W^-T y with the attempt's factors (solve_attempt's two cases).
template <bool DN>
__device__ __forceinline__ void solve_attempt_transposed(double* work, const double* Wd, double c0, double* y,
                                                         const u16* smap, int lane) {
  if (DN)
    dense_solve_transposed(Wd, work + A_DINV, (const int*)(work + A_M), y, lane);
  else
    solve_w_transposed(work + A_M, work + A_DINV, y, work + A_U, c0, smap, lane);
}

// Reverse of F = (n_tot / nd) S v(n), n = nd Y / n_tot, at Y for the cotangent Fb (rate coefficients fixed): Yb +=
// J(Y)^T Fb; lbk (ln k common to kf and kr; ln kr alone) and lbnd (ln nd) += their cotangents. The sums of n_q times
// the species cotangents come reaction by reaction (a product of m factors: sum over its slots = m times the product).
__device__ __forceinline__ void rhs_vjp(double* work, const double* Y, const double* Fb, double* Yb, double* lbk,
                                        double* rho, double& lbnd, const double* kf, const double* kr, double nd,
                                        const Tab& tab, const u64* __restrict__ gt, const int* __restrict__ xg,
                                        int ground, int lane) {
  double *n = work + A_N0, *part = work + A_PART;
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += Y[i];
  const double ntot = wsum(p), xf = nd / ntot, g = ntot / nd;
  for (int i = lane; i < NS; i += 32) n[i] = Y[i] * xf;
  if (lane == 0) n[NS] = 1.0;
  for (int i = lane; i < NPG; i += 32) part[i] = 0.0;
  __syncwarp();
  double ff = 0.0, tl = 0.0;
  for (int r = lane; r < NR; r += 32) {
    const u8* q = tab.rx + 6 * r;
    double sb = 0.0, mre = 0.0, mpr = 0.0;
#pragma unroll
    for (int m = 0; m < 3; m++) {
      if (q[m] < NS) {
        sb -= Fb[q[m]];
        mre += 1.0;
      }
      if (q[3 + m] < NS) {
        sb += Fb[q[3 + m]];
        mpr += 1.0;
      }
      if (r >= NRT && PX[3 * (r - NRT) + m] < NS) sb += Fb[PX[3 * (r - NRT) + m]];  // photolysis products
    }
    const double vb = sb * g;
    const double pp = n[q[0]] * n[q[1]] * n[q[2]];
    const double a = kf[r] * pp, b = kr[r] * (n[q[3]] * n[q[4]] * n[q[5]]), ab = a - b;
    ff += vb * ab;
    tl += vb * (mre * a - mpr * b);
    rho[r] = vb * kf[r];
    rho[NR + r] = -vb * kr[r];
    lbk[r] += r < NRT ? vb * ab : vb * pp;  // photolysis: the cotangent of kf itself
    lbk[NR + r] -= vb * b;
  }
  __syncwarp();
  const double FF = wsum(ff), tau = wsum(tl);
  {
    double acc = 0.0;
    for (int qq = 0; qq < ground; qq++) {
      const u64 word = __ldg(gt + qq * 32 + lane);
      if ((word >> GT_USED) & 1ull) {
        acc = fma(rho[word & GT_RATE_MASK], n[(word >> GT_A) & GT_SPECIES_MASK] * n[(word >> GT_B) & GT_SPECIES_MASK],
                  acc);
        if ((word >> GT_END) & 1ull) {
          part[(word >> GT_PIECE) & GT_PIECE_MASK] = acc;
          acc = 0.0;
        }
      }
    }
  }
  __syncwarp();
  for (int i = lane; i < NS; i += 32) {
    double s = part[i];
    for (int e = xg[i]; e < xg[i + 1]; e++) s += part[NS + e];
    Yb[i] += xf * s + (FF - tau) / ntot;
  }
  lbnd += tau - FF;
  __syncwarp();
}

// Stages' concentration derivatives and the rho of rhs_vjp_n: the reverse terms of one reaction r, added to its sums.
__device__ __forceinline__ void reverse_reaction(int r, const double* work, double* lbk, double* rho, const double* n,
                                                 const double (&rt)[4], double g, const double* kf, const double* kr,
                                                 double (&ff)[4], double (&ffd)[4], double (&tl)[4], double (&tdl)[4],
                                                 double& ffF, double& tF, const Tab& tab) {
  constexpr int OZ[4] = {A_KB1, A_KB2, A_KB3, A_Z}, ON1[4] = {A_N1, A_YB, A_YB2, A_NBD};
  const u8* q = tab.rx + 6 * r;
  double mre = 0.0, mpr = 0.0;
#pragma unroll
  for (int m = 0; m < 3; m++) {
    mre += q[m] < NS ? 1.0 : 0.0;
    mpr += q[3 + m] < NS ? 1.0 : 0.0;
  }
  const double n0 = n[q[0]], n1 = n[q[1]], n2 = n[q[2]], n3 = n[q[3]], n4 = n[q[4]], n5 = n[q[5]];
  const double pp = n0 * n1 * n2;
  const double a = kf[r] * pp, b = kr[r] * (n3 * n4 * n5), ab = a - b;
  double vbF = 0.0, cot_k = 0.0, lr = 0.0, lp = 0.0;  // lp: a photolysis row's cotangent of kf itself
#pragma unroll
  for (int s = 0; s < 4; s++) {
    const double *z = work + OZ[s], *d = work + ON1[s];
    double sb = 0.0;
#pragma unroll
    for (int m = 0; m < 3; m++) {
      if (q[m] < NS) sb -= z[q[m]];
      if (q[3 + m] < NS) sb += z[q[3 + m]];
      if (r >= NRT && PX[3 * (r - NRT) + m] < NS) sb += z[PX[3 * (r - NRT) + m]];
    }
    const double vb = sb * g;
    const double Pd = d[q[0]] * n1 * n2 + n0 * d[q[1]] * n2 + n0 * n1 * d[q[2]];
    const double Qd = d[q[3]] * n4 * n5 + n3 * d[q[4]] * n5 + n3 * n4 * d[q[5]];
    const double ad = kf[r] * Pd, bd = kr[r] * Qd;
    lp += vb * rt[s] * pp + vb * Pd;
    cot_k += vb * rt[s] * ab + vb * (ad - bd);
    lr -= vb * rt[s] * b + vb * bd;
    ff[s] += vb * ab;
    ffd[s] += vb * (ad - bd);
    tl[s] += vb * (mre * a - mpr * b);
    tdl[s] += vb * ((mre - 1.0) * ad - (mpr - 1.0) * bd);
    rho[s * 2 * NR + r] = vb * kf[r];
    rho[s * 2 * NR + NR + r] = -vb * kr[r];
    if (s < 2) vbF += vb;
  }
  cot_k += vbF * ab;
  lp += vbF * pp;
  lr -= vbF * b;
  ffF += vbF * ab;
  tF += vbF * (mre * a - mpr * b);
  lbk[r] += r < NRT ? cot_k : lp;
  lbk[NR + r] += lr;
}

// The gather of rhs_vjp_n: the pieces of Nb from rho, with the concentrations n and their derivatives along each K_i.
__device__ __forceinline__ void reverse_gather(const double* work, const double* rho, const double* n, double* part,
                                               const u64* __restrict__ gt, int ground, int lane) {
  constexpr int ON1[4] = {A_N1, A_YB, A_YB2, A_NBD};
  double acc = 0.0;
  for (int qq = 0; qq < ground; qq++) {
    const u64 word = __ldg(gt + qq * 32 + lane);
    if ((word >> GT_USED) & 1ull) {
      const int k = (int)(word & GT_RATE_MASK), ia = (int)((word >> GT_A) & GT_SPECIES_MASK),
                ib = (int)((word >> GT_B) & GT_SPECIES_MASK);
      const double na = n[ia], nb = n[ib], r0 = rho[k], r1 = rho[2 * NR + k];
      acc = fma(r0 + r1, na * nb, acc);
#pragma unroll
      for (int s = 0; s < 4; s++) {
        const double* d = work + ON1[s];
        acc = fma(s == 0 ? r0 : (s == 1 ? r1 : rho[s * 2 * NR + k]), d[ia] * nb + na * d[ib], acc);
      }
      if ((word >> GT_END) & 1ull) {
        part[(word >> GT_PIECE) & GT_PIECE_MASK] = acc;
        acc = 0.0;
      }
    }
  }
}

// The reverse at N of F(N) for the cotangent z1 + z2 and of the four stages' Jacobian terms: Nb += J(N)^T (z1 + z2) +
// sum_i d/ds [J(N + s K_i)^T z_i] (z_i = W^-T of stage i's cotangent: z1..z3 in Kb1..Kb3, z4 in z), and the matching
// rate cotangents, in one pass over the reactions and one gather (n1_i = the concentrations' derivative along K_i).
__device__ __forceinline__ void rhs_vjp_n(double* work, double* lbk, double* rho, double& lbnd, const double* kf,
                                          const double* kr, double nd, const Tab& tab, const u64* __restrict__ gt,
                                          const int* __restrict__ xg, int ground, int lane) {
  constexpr int ON1[4] = {A_N1, A_YB, A_YB2, A_NBD};
  double *N = work + A_NN, *Nb = work + A_NB, *n = work + A_N0, *part = work + A_PART;
  const double* K = work + A_K1;
  double p = 0.0, pd[4] = {0.0, 0.0, 0.0, 0.0};
  for (int i = lane; i < NS; i += 32) {
    p += N[i];
#pragma unroll
    for (int s = 0; s < 4; s++) pd[s] += K[s * NS + i];
  }
  const double ntot = wsum(p), xf = nd / ntot, g = ntot / nd;
  double rt[4];
#pragma unroll
  for (int s = 0; s < 4; s++) rt[s] = wsum(pd[s]) / ntot;
  for (int i = lane; i < NS; i += 32) {
    n[i] = N[i] * xf;
#pragma unroll
    for (int s = 0; s < 4; s++) work[ON1[s] + i] = xf * (K[s * NS + i] - N[i] * rt[s]);
  }
  if (lane == 0) {
    n[NS] = 1.0;
#pragma unroll
    for (int s = 0; s < 4; s++) work[ON1[s] + NS] = 0.0;
  }
  for (int i = lane; i < NPG; i += 32) part[i] = 0.0;
  __syncwarp();
  double ffF = 0.0, tF = 0.0, ff[4], ffd[4], tl[4], tdl[4];
#pragma unroll
  for (int s = 0; s < 4; s++) {
    ff[s] = 0.0;
    ffd[s] = 0.0;
    tl[s] = 0.0;
    tdl[s] = 0.0;
  }
  for (int r = lane; r < NR; r += 32)
    reverse_reaction(r, work, lbk, rho, n, rt, g, kf, kr, ff, ffd, tl, tdl, ffF, tF, tab);
  __syncwarp();
  const double FF_F = wsum(ffF), tau_F = wsum(tF);
  double sc = (FF_F - tau_F) / ntot, lb = tau_F - FF_F;
#pragma unroll
  for (int s = 0; s < 4; s++) {
    const double FF = wsum(ff[s]), fd = wsum(ffd[s]), tau = wsum(tl[s]);
    const double FFd = fd + FF * rt[s], taud = rt[s] * tau + wsum(tdl[s]) + fd;
    sc += (FFd - taud) / ntot - (FF - tau) * rt[s] / ntot;
    lb += taud - FFd;
  }
  reverse_gather(work, rho, n, part, gt, ground, lane);
  __syncwarp();
  for (int i = lane; i < NS; i += 32) {
    double s = part[i];
    for (int e = xg[i]; e < xg[i + 1]; e++) s += part[NS + e];
    Nb[i] += xf * s + sc;
  }
  lbnd += lb;
  __syncwarp();
}

// Reverse of the stages of the attempt in place (K1..K4, the full LU in M, dinv, u, c0; DN: the recovery's dense
// factors) for the cotangents Yb of the new state and z of K4: Nb += the state's cotangent, lbk, lbnd += the rate
// cotangents; returns the cotangent of the step size (adapt) or 0. The stages' W^-T solves come first (z4 in z, z3..z1
// in place of Kb3..Kb1), with the reverse of F at Y4 and Y3 between them; the terms at N follow in one pass.
template <bool DN>
__device__ __forceinline__ double reverse_stages(double* work, const double* Wd, double ha, double c0, int adapt,
                                                 const double* kf, const double* kr, double* lbk, double* rho,
                                                 double& lbnd, double nd, const Tab& tab, const u64* __restrict__ gt,
                                                 const int* __restrict__ xg, int ground, int lane) {
  double *N = work + A_NN, *Nb = work + A_NB, *Yb = work + A_YB, *Yb2 = work + A_YB2, *z = work + A_Z,
         *Ys = work + A_YS, *K1 = work + A_K1, *K2 = work + A_K2, *K3 = work + A_K3, *K4 = work + A_K4,
         *Kb1 = work + A_KB1, *Kb2 = work + A_KB2, *Kb3 = work + A_KB3;
  const double ih = 1.0 / ha;
  double ihb = 0.0, ighb = 0.0;
  for (int i = lane; i < NS; i += 32) {
    const double yb = Yb[i];
    Nb[i] += yb;
    Kb1[i] = 2.0 * yb;
    Kb2[i] = 0.0;
    Kb3[i] = yb;
    z[i] += yb;
  }
  __syncwarp();
  // stage 4: K4 = W^-1 (F(Y4) + (K1 - K2 - 8/3 K3) / h), Y4 = N + 2 K1 + K3
  solve_attempt_transposed<DN>(work, Wd, c0, z, tab.smap, lane);
  for (int i = lane; i < NS; i += 32) {
    Kb1[i] += ih * z[i];
    Kb2[i] -= ih * z[i];
    Kb3[i] -= RODAS_C43 * ih * z[i];
    ihb += z[i] * (K1[i] - K2[i] - RODAS_C43 * K3[i]);
    ighb -= z[i] * K4[i];
    Ys[i] = N[i] + 2.0 * K1[i] + K3[i];
    Yb2[i] = 0.0;
  }
  __syncwarp();
  rhs_vjp(work, Ys, z, Yb2, lbk, rho, lbnd, kf, kr, nd, tab, gt, xg, ground, lane);
  for (int i = lane; i < NS; i += 32) {
    Nb[i] += Yb2[i];
    Kb1[i] += 2.0 * Yb2[i];
    Kb3[i] += Yb2[i];
  }
  __syncwarp();
  // stage 3: K3 = W^-1 (F(Y3) + (K1 - K2) / h), Y3 = N + 2 K1
  solve_attempt_transposed<DN>(work, Wd, c0, Kb3, tab.smap, lane);
  for (int i = lane; i < NS; i += 32) {
    Kb1[i] += ih * Kb3[i];
    Kb2[i] -= ih * Kb3[i];
    ihb += Kb3[i] * (K1[i] - K2[i]);
    ighb -= Kb3[i] * K3[i];
    Ys[i] = N[i] + 2.0 * K1[i];
    Yb2[i] = 0.0;
  }
  __syncwarp();
  rhs_vjp(work, Ys, Kb3, Yb2, lbk, rho, lbnd, kf, kr, nd, tab, gt, xg, ground, lane);
  for (int i = lane; i < NS; i += 32) {
    Nb[i] += Yb2[i];
    Kb1[i] += 2.0 * Yb2[i];
  }
  __syncwarp();
  // stage 2: K2 = W^-1 (F(N) + 4 K1 / h)
  solve_attempt_transposed<DN>(work, Wd, c0, Kb2, tab.smap, lane);
  for (int i = lane; i < NS; i += 32) {
    Kb1[i] += RODAS_C21 * ih * Kb2[i];
    ihb += RODAS_C21 * Kb2[i] * K1[i];
    ighb -= Kb2[i] * K2[i];
  }
  __syncwarp();
  // stage 1: K1 = W^-1 F(N)
  solve_attempt_transposed<DN>(work, Wd, c0, Kb1, tab.smap, lane);
  for (int i = lane; i < NS; i += 32) ighb -= Kb1[i] * K1[i];
  __syncwarp();
  rhs_vjp_n(work, lbk, rho, lbnd, kf, kr, nd, tab, gt, xg, ground, lane);
  if (!adapt) return 0.0;
  return -(wsum(ihb) + 2.0 * wsum(ighb)) / (ha * ha);
}

// One stored attempt (REC): the state before it and the step control's history.
__device__ __forceinline__ void save_attempt(double* rp, const double* N, double h, double t, bool prej, double pen,
                                             double pha, int lane) {
  for (int i = lane; i < NS; i += 32) rp[i] = N[i];
  if (lane == 0) {
    rp[NS] = h;
    rp[NS + 1] = t;
    rp[NS + 3] = pen;
    rp[NS + 4] = pha;
    rp[NS + 5] = prej ? 1.0 : 0.0;
  }
}

#define VJP_ARGS                                                                                                       \
  const double *__restrict__ x0, const double *__restrict__ Ta, const double *__restrict__ Pa,                         \
      const double *__restrict__ dta, const double *__restrict__ hin, const double *lm, const double *__restrict__ rc, \
      const double *__restrict__ gx, const double *gh, double *__restrict__ xout, double *__restrict__ hout,           \
      int *__restrict__ doneo, double *__restrict__ noko, double *__restrict__ gx0, double *__restrict__ gT,           \
      double *__restrict__ gP, double *__restrict__ gdt, double *__restrict__ ghin, double *__restrict__ glm,          \
      int *__restrict__ count, int B, double rtol, double atol, int max_steps, int adapt,                              \
      const int *__restrict__ perm, const unsigned *__restrict__ blob, const int *__restrict__ rf,                     \
      const double *__restrict__ nasa, double *scratch, double *rec, const unsigned *__restrict__ sst,                 \
      const u64 *__restrict__ jt, const u64 *__restrict__ lu, const int *__restrict__ optr,                            \
      const u64 *__restrict__ gt, const int *__restrict__ xg, int sround, int jround, int ground,                      \
      const double *__restrict__ Jin, double *__restrict__ gJ, double *__restrict__ gA
#define VJP_PASS                                                                                                    \
  x0, Ta, Pa, dta, hin, lm, rc, gx, gh, xout, hout, doneo, noko, gx0, gT, gP, gdt, ghin, glm, count, B, rtol, atol, \
      max_steps, adapt, perm, blob, rf, nasa, scratch, rec, sst, jt, lu, optr, gt, xg, sround, jround, ground, Jin, \
      gJ, gA

// The cotangents of a failed cell's inputs: NaN.
__device__ __forceinline__ void nan_cotangents(double* __restrict__ gx0, double* __restrict__ gT,
                                               double* __restrict__ gP, double* __restrict__ gdt,
                                               double* __restrict__ ghin, double* __restrict__ glm,
                                               double* __restrict__ gJ, double* __restrict__ gA, int cell, int lane) {
  for (int i = lane; i < NS; i += 32) gx0[(size_t)cell * NS + i] = DNAN;
  for (int r = lane; r < NR; r += 32) {
    glm[(size_t)cell * NR + r] = DNAN;
    gA[(size_t)cell * NR + r] = DNAN;
  }
  for (int b = lane; b < NJ; b += 32) gJ[(size_t)cell * NJ + b] = DNAN;
  if (lane == 0) {
    gT[cell] = DNAN;
    gP[cell] = DNAN;
    gdt[cell] = DNAN;
    ghin[cell] = DNAN;
  }
}

// The reverse of the error of the attempt in place, for its cotangent enb: z = the cotangent of K4, and Nb, Yb += those
// of N and Y. en = sqrt(mean q^2), q = K4 / sc, sc = atol + rtol max(|N|, |Y|); on a first try (t = 0 until a step is
// accepted) max(that, max |q|): the selected norm, half each at a tie, the max-norm's cotangent shared evenly among the
// species attaining it.
__device__ __forceinline__ void error_cotangent(const double* N, const double* Y, const double* K4, double* Nb,
                                                double* Yb, double* z, double en, double enb, bool first, double rtol,
                                                double atol, int lane) {
  double er = en, emx = 0.0, wr = 1.0, wm = 0.0;
  if (first) {
    double e2 = 0.0, em = 0.0, nt = 0.0;
    for (int i = lane; i < NS; i += 32) {
      const double qq = K4[i] / (atol + rtol * fmax(fabs(N[i]), fabs(Y[i])));
      e2 += qq * qq;
      em = fmax(em, fabs(qq));
    }
    er = sqrt(wsum(e2) / NS);
    emx = wmax(em);
    for (int i = lane; i < NS; i += 32)
      nt += fabs(K4[i] / (atol + rtol * fmax(fabs(N[i]), fabs(Y[i])))) == emx ? 1.0 : 0.0;
    wr = er > emx ? 1.0 : (er < emx ? 0.0 : 0.5);
    wm = (1.0 - wr) / wsum(nt);
  }
  for (int i = lane; i < NS; i += 32) {
    const double aN = fabs(N[i]), aY = fabs(Y[i]), sc = atol + rtol * fmax(aN, aY);
    const double qq = K4[i] / sc;
    const double qb =
        !first ? enb * qq / (NS * en) : enb * (wr * qq / (NS * er) + (fabs(qq) == emx ? wm * sgn(qq) : 0.0));
    const double scb = -rtol * qb * qq / sc;
    z[i] = qb / sc;
    const double wN = aN > aY ? 1.0 : (aN < aY ? 0.0 : 0.5);
    Nb[i] += wN * scb * sgn(N[i]);
    Yb[i] += (1.0 - wN) * scb * sgn(Y[i]);
  }
}

// The rate cotangents lbk -> the log multipliers glm, J, gA (the forward coefficients before the limit, for arrh), and
// this lane's parts of the cotangents of T (gTl) and ln nd (gnl).
__device__ __forceinline__ void rate_cotangents(double* work, const double* lbk, const double* kf, const double* cw,
                                                const double* lmc, const double* __restrict__ rc,
                                                const int* __restrict__ rf, const double* __restrict__ nasa,
                                                const Tab& tab, double Tc, double nd, int cell,
                                                double* __restrict__ glm, double* __restrict__ gA,
                                                double* __restrict__ gJ, double& gTl, double& gnl, int lane) {
  const double lT = log(Tc), iT = 1.0 / Tc;
  double* gTs = work + A_N1;
  gibbs_dT(gTs, Tc, nasa, lane);
  for (int r = lane; r < NR; r += 32) {
    const double cot_k = lbk[r], lr = lbk[NR + r];  // ln k (forward and reverse together), ln kr alone
    if (r >= NRT) {                                 // photolysis: cot_k = the cotangent of kf = J exp(lm)
      glm[(size_t)cell * NR + r] = kf[r] * cot_k;
      gA[(size_t)cell * NR + r] = 0.0;
      gJ[(size_t)cell * NJ + r - NRT] = (lmc ? exp(lmc[r]) : 1.0) * cot_k;
      continue;
    }
    glm[(size_t)cell * NR + r] = cot_k;
    const int fl = rf[r];
    double dT = 0.0, dn = 0.0, D[9];
    rate_law_partials(rc + RCW * r, fl, Tc, lT, iT, nd, dT, dn, D);
    const u8* x = tab.rx + 6 * r;
    const double ddg = gTs[x[3]] + gTs[x[4]] + gTs[x[5]] - gTs[x[0]] - gTs[x[1]] - gTs[x[2]];
    const double wA = cw[r], wG = cw[NR + r];
    const double wE = collision_limit_nd_weight(fl, wA, wG);
    const double Ab = (1.0 - wA) * cot_k, Gb = lr - wG * cot_k;  // the collision limit (rates_tan)
    gA[(size_t)cell * NR + r] = Ab;
    gTl += Ab * dT + Gb * (ddg + reaction_dmu(fl) * iT);
    gnl += Ab * dn - wE * cot_k;
  }
}

// The call and the cotangents of its inputs, one warp per cell. DN = false: every cell, the call kernel's attempts
// (COUNT_QUEUE; CAP stored attempts per cell in rec). DN = true: the first count[COUNT_RECOMPUTED] cells of list, the
// recover kernel's attempts (dense W in wscr; max_steps = the recovery's tries; CAPR stored attempts;
// COUNT_RECOVERY_QUEUE); its result tested as the recover kernel tests it: redo = 1 and COUNT_RECOVERED, or redo = 2,
// COUNT_STILL_FAILED and NaN cotangents.
template <bool DN>
// size-limit-ok: the forward and reverse passes share one cell's state, which a split would pass by hand
__device__ __forceinline__ void vjp_cells(VJP_ARGS, int* redo, const int* list, double* wscr,
                                          const u16* __restrict__ didx, const double* __restrict__ atoms) {
  constexpr int CP = DN ? CAPR : CAP;
  extern __shared__ double smem[];
  unsigned* tb32 = (unsigned*)(smem + WPB * PERWA);
  const Tab tab = load_tab(tb32, blob);
  const int lane = (int)(threadIdx.x & 31), wib = (int)(threadIdx.x >> 5);
  const int myt = lane < NT ? lane : -1;
  double c[NT];
#pragma unroll
  for (int j = 0; j < NT; j++) c[j] = 0.0;
  double* work = smem + wib * PERWA;
  double *M = work + A_M, *N = work + A_NN, *Y = work + A_Y, *Nb = work + A_NB, *Yb = work + A_YB, *z = work + A_Z,
         *K4 = work + A_K4;
  const size_t wid = (size_t)blockIdx.x * WPB + wib;
  double* kf = scratch + wid * VJP_STRIDE;
  double *kr = kf + NR, *lbk = kr + NR + 2, *rho = lbk + 2 * NR, *cw = rho + 8 * NR;  // rho: 4 x 2 NR
  if (lane == 0) kf[2 * NR] = 0.0;  // the zero rate of the Jacobian's padding words
  double* recA = rec + wid * 2 * CP * REC;
  double* recB = recA + CP * REC;
  double* Wd = DN ? wscr + wid * NS * NS : nullptr;
  const int nc = DN ? count[COUNT_RECOMPUTED] : B;
  int q = (int)(blockIdx.x * WPB + wib);
  while (q < nc) {
    const int cell = DN ? list[q] : q;
    const double Tc = Ta[cell], dt = dta[cell], Pc = Pa[cell], nd = Pc / (KBOLTZ * Tc);
    const double* lmc = lm ? lm + (size_t)cell * NR : nullptr;
    double p = 0.0, pm = 0.0;
    for (int i = lane; i < NS; i += 32) {
      N[i] = x0[(size_t)cell * NS + perm[i]];
      screen_x(N[i], p, pm);
    }
    screen_J(Jin, cell, lane, 32, p);
    screen_lm(lmc, lane, p);
    const bool fin = cell_fine(p, pm, Tc, nd, dt), cold = Tc < TMIN;
    if (fin && !cold) rates(work, kf, kr, cw, Tc, nd, lmc, Jin + (size_t)cell * NJ, rc, rf, nasa, tab, lane);
    __syncwarp();
    const bool hfin = isfinite(hin[cell]);  // a non-finite h acts as dt (no sensitivity to it), and is counted
    if (!DN && !hfin && lane == 0) atomicAdd(count + COUNT_NONFINITE_H, 1);
    const double hi = hfin ? hin[cell] : dt, m0 = fmin(hi, dt), h0 = fmax(m0, H_MIN);
    double h = h0, t = 0.0, nok = 0.0, pen = 0.0, pha = 0.0;
    bool prej = false;
    int na = 0;
    const bool run = dt > 0.0 && fin && !cold;
    double en_l = 0.0, c0_l = 0.0;  // the last attempt run (its stages are in place)
    // forward: the state before each of the first CP attempts (and whether it was accepted)
    while (run && na < max_steps && t < dt * CALL_END) {
      if (na < CP) save_attempt(recA + na * REC, N, h, t, prej, pen, pha, lane);
      const bool ok = forward_step<DN>(work, c, Wd, h, t, nok, prej, pen, pha, dt, kf, kr, nd, rtol, atol, en_l, c0_l,
                                       tab, sst, jt, lu, optr, didx, sround, jround, lane, myt);
      if (na < CP && lane == 0) recA[na * REC + NS + 2] = ok ? 1.0 : 0.0;
      na++;
    }
    const double s = write_state(N, xout + (size_t)cell * NS, perm, lane, lane, 32);
    double gs = 0.0;
    for (int i = lane; i < NS; i += 32) gs += gx[(size_t)cell * NS + perm[i]] * N[i];
    gs = wsum(gs);
    const int dn = (fin && t >= dt * CALL_END) ? 1 : 0;
    if (lane == 0) {
      hout[cell] = h;
      doneo[cell] = dn;
      noko[cell] = nok;
    }
    bool bad = false;
    if (DN) {
      bad = recovery_fails(x0, Ta, Pa, dta, xout, doneo, redo, count, dn, nok, atoms, atol, cell, lane J_ARG);
      if (bad) na = 0;  // no reverse: NaN cotangents below
    }
    // reverse: x = N / sum N
    for (int i = lane; i < NS; i += 32) Nb[i] = gx[(size_t)cell * NS + perm[i]] / s - gs / (s * s);
    for (int r = lane; r < 2 * NR; r += 32) lbk[r] = 0.0;
    __syncwarp();
    double hb = (adapt && gh) ? gh[cell] : 0.0, tb = 0.0, dtb = 0.0, lbnd = 0.0;
    double enc = 0.0, hac = 0.0;  // cotangents of the previous attempt's error and size from a secant factor
    const int st = na <= CP ? 1 : (na + CP - 1) / CP;  // attempts per stored segment
    if (st > 1) {                                      // checkpoints every st attempts
      for (int i = lane; i < NS; i += 32) N[i] = x0[(size_t)cell * NS + perm[i]];
      __syncwarp();
      h = h0;
      t = 0.0;
      nok = 0.0;
      prej = false;
      pen = 0.0;
      pha = 0.0;
      for (int j = 0; j < na; j++) {
        if (j % st == 0) save_attempt(recA + (j / st) * REC, N, h, t, prej, pen, pha, lane);
        forward_step<DN>(work, c, Wd, h, t, nok, prej, pen, pha, dt, kf, kr, nd, rtol, atol, en_l, c0_l, tab, sst, jt,
                         lu, optr, didx, sround, jround, lane, myt);
      }
    }
    const int seg = st > 1 ? st : na;  // attempts per reversed segment
    for (int g = na > 0 ? (na - 1) / seg : -1; g >= 0; g--) {
      const int j0 = g * seg, j1 = min(na, j0 + seg);
      double* rb = recA;
      if (st > 1) {  // the segment's attempts, recomputed and stored
        __syncwarp();
        for (int i = lane; i < NS; i += 32) N[i] = recA[g * REC + i];
        h = recA[g * REC + NS];
        t = recA[g * REC + NS + 1];
        pen = recA[g * REC + NS + 3];
        pha = recA[g * REC + NS + 4];
        prej = recA[g * REC + NS + 5] != 0.0;
        __syncwarp();
        for (int j = j0; j < j1; j++) {
          save_attempt(recB + (j - j0) * REC, N, h, t, prej, pen, pha, lane);
          const bool ok = forward_step<DN>(work, c, Wd, h, t, nok, prej, pen, pha, dt, kf, kr, nd, rtol, atol, en_l,
                                           c0_l, tab, sst, jt, lu, optr, didx, sround, jround, lane, myt);
          if (lane == 0) recB[(j - j0) * REC + NS + 2] = ok ? 1.0 : 0.0;
        }
        rb = recB - j0 * REC;
      }
      for (int j = j1 - 1; j >= j0; j--) {
        __syncwarp();
        const double* rp = rb + j * REC;
        if (!adapt && rp[NS + 2] == 0.0) continue;  // frozen steps: a rejected attempt has no effect
        for (int i = lane; i < NS; i += 32) N[i] = rp[i];
        const double hj = rp[NS], tj = rp[NS + 1], rem = dt - tj, ha = fmin(hj, rem);
        __syncwarp();
        double c0 = c0_l, en = en_l, dfe = 0.0, dfh = 0.0, dfp = 0.0, dfq = 0.0;
        bool ok = rp[NS + 2] != 0.0;
        if (j < j1 - 1)  // else the attempt just run, still in place
          en = run_attempt<DN>(work, c, Wd, ha, tj == 0.0, kf, kr, nd, rtol, atol, c0, ok, tab, sst, jt, lu, optr, didx,
                               sround, jround, lane, myt);
        if (!DN && myt >= 0) {  // the dense block's factors back into M
#pragma unroll
          for (int k = 0; k < NT; k++) M[TB0 + myt * NT + k] = c[k];
        }
        const double fac =
            step_factor_partials(en, ok, rp[NS + 5] != 0.0, rp[NS + 3], rp[NS + 4], ha, dfe, dfh, dfp, dfq);
        double hab = adapt ? hb * fac + hb * ha * dfh + hac : 0.0;
        const double enb = adapt ? hb * ha * dfe + enc : 0.0;
        enc = adapt ? hb * ha * dfp : 0.0;  // to the previous attempt (the next one reversed)
        hac = adapt ? hb * ha * dfq : 0.0;
        for (int i = lane; i < NS; i += 32) {
          if (ok) {
            Yb[i] = Y[i] >= 0.0 ? Nb[i] : 0.0;
            Nb[i] = 0.0;
          } else
            Yb[i] = 0.0;
          z[i] = 0.0;
        }
        if (ok && adapt) hab += tb;
        if (enb != 0.0 && en > 0.0) error_cotangent(N, Y, K4, Nb, Yb, z, en, enb, tj == 0.0, rtol, atol, lane);
        __syncwarp();
        if (ok || enb != 0.0)
          hab += reverse_stages<DN>(work, Wd, ha, c0, adapt, kf, kr, lbk, rho, lbnd, nd, tab, gt, xg, ground, lane);
        if (adapt) {  // ha = min(h, dt - t)
          const double wh = hj < rem ? 1.0 : (hj > rem ? 0.0 : 0.5);
          hb = wh * hab;
          dtb += (1.0 - wh) * hab;
          tb -= (1.0 - wh) * hab;
        } else
          hb = 0.0;
      }
    }
    __syncwarp();
    // h0 = max(min(hin, dt), H_MIN)
    {
      const double wm = m0 > H_MIN ? 1.0 : (m0 < H_MIN ? 0.0 : 0.5);
      const double wh = !hfin ? 0.0 : (hi < dt ? 1.0 : (hi > dt ? 0.0 : 0.5));
      const double mb = run ? wm * hb : 0.0;
      if (lane == 0) {
        ghin[cell] = wh * mb;
        gdt[cell] = dtb + (1.0 - wh) * mb;
      }
    }
    for (int i = lane; i < NS; i += 32) gx0[(size_t)cell * NS + perm[i]] = Nb[i];
    double gTl = 0.0, gnl = 0.0;
    if (run) {
      rate_cotangents(work, lbk, kf, cw, lmc, rc, rf, nasa, tab, Tc, nd, cell, glm, gA, gJ, gTl, gnl, lane);
    } else {
      for (int r = lane; r < NR; r += 32) {
        glm[(size_t)cell * NR + r] = 0.0;
        gA[(size_t)cell * NR + r] = 0.0;
      }
      for (int b = lane; b < NJ; b += 32) gJ[(size_t)cell * NJ + b] = 0.0;
    }
    const double gTt = wsum(gTl), lnd = wsum(gnl) + lbnd;
    if (lane == 0) {
      gT[cell] = run ? gTt - lnd / Tc : 0.0;
      gP[cell] = run ? lnd / Pc : 0.0;
    }
    if (DN && bad) nan_cotangents(gx0, gT, gP, gdt, ghin, glm, gJ, gA, cell, lane);  // still failed after the recovery
    q = next_queued<DN>(count, lane);
  }
}

extern "C" __global__ void __launch_bounds__(32 * WPB) vjp(VJP_ARGS) {
  vjp_cells<false>(VJP_PASS, nullptr, nullptr, nullptr, nullptr, nullptr);
}

// After vjp and the guard: the listed cells redone on the recovery's path with their cotangents (redo, list: the
// guard's; rec holds CAPR attempts per warp).
extern "C" __global__ void __launch_bounds__(32 * WPB)
    vjp_rec(VJP_ARGS, int* __restrict__ redo, const int* __restrict__ list, double* __restrict__ wscr,
            const u16* __restrict__ didx, const double* __restrict__ atoms) {
  if ((int)(blockIdx.x * WPB) >= count[COUNT_RECOMPUTED]) return;  // no listed cell for this block
  vjp_cells<true>(VJP_PASS, redo, list, wscr, didx, atoms);
}

// Cotangent of the Arrhenius table (its nine Arrhenius columns): grc[r, j] = sum over cells of gA[cell, r] d A_r /
// d rc[r, j] (gA: the cotangent of the forward coefficient's log before the collision limit; 0 for photolysis rows),
// one block per reaction, a fixed summation order.
extern "C" __global__ void arrh(const double* __restrict__ gA, const double* __restrict__ Ta,
                                const double* __restrict__ Pa, const double* __restrict__ rc,
                                const int* __restrict__ rf, double* __restrict__ grc, int B) {
  __shared__ double red[9][ARRH_BLOCK_THREADS];
  const int r = (int)blockIdx.x, tid = (int)threadIdx.x, fl = rf[r];
  double acc[9];
#pragma unroll
  for (int j = 0; j < 9; j++) acc[j] = 0.0;
  for (int cell = tid; cell < B && r < NRT; cell += ARRH_BLOCK_THREADS) {
    const double g = gA[(size_t)cell * NR + r];
    if (g == 0.0) continue;
    const double Tc = Ta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
    double dT = 0.0, dn = 0.0, D[9];
    rate_law_partials(rc + RCW * r, fl, Tc, log(Tc), 1.0 / Tc, nd, dT, dn, D);
#pragma unroll
    for (int j = 0; j < 9; j++) acc[j] += g * D[j];
  }
#pragma unroll
  for (int j = 0; j < 9; j++) red[j][tid] = acc[j];
  __syncthreads();
  for (int s = ARRH_BLOCK_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) {
#pragma unroll
      for (int j = 0; j < 9; j++) red[j][tid] += red[j][tid + s];
    }
    __syncthreads();
  }
  if (tid < 9) grc[r * 9 + tid] = red[tid][0];
}
