// A test probe of the rate arithmetic the derivative kernels share, one warp per cell: the rate coefficients after the
// collision limit and the limit's weights (rates), the rate tangents along dT and d ln nd (rates_tan), and the
// cotangents of T and ln nd at fixed ln nd and T from a cotangent on ln kf and ln kr (rate_cotangents). Its header
// is the adjoint kernels' (kincuda_ad.py); tests/test_collision_limit.py builds and runs it.
#include "vjp.cu"

extern "C" __global__ void rate_probe(const double* __restrict__ Ta, const double* __restrict__ Pa,
                                      const double* __restrict__ dTa, const double* __restrict__ dlnda,
                                      const double* __restrict__ lbk, const double* __restrict__ Jin,
                                      const double* __restrict__ rc, const int* __restrict__ rf,
                                      const double* __restrict__ nasa, const unsigned* __restrict__ blob, double* work,
                                      double* kf, double* kr, double* cw, double* dk, double* glm, double* gA,
                                      double* gJ, double* gTn) {
  extern __shared__ unsigned tb32[];
  const Tab tab = load_tab(tb32, blob);
  const int cell = (int)blockIdx.x, lane = (int)threadIdx.x;
  const double Tc = Ta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
  double* w = work + (size_t)cell * PERWA;
  double *kfc = kf + (size_t)cell * NR, *krc = kr + (size_t)cell * NR, *cwc = cw + (size_t)cell * 2 * NR;
  rates(w, kfc, krc, cwc, Tc, nd, nullptr, Jin + (size_t)cell * NJ, rc, rf, nasa, tab, lane);
  rates_tan(w, dk + (size_t)cell * 2 * NR, Tc, nd, dTa[cell], dlnda[cell], nullptr, nullptr, nullptr, rc, nullptr, rf,
            nasa, kfc, krc, cwc, tab, lane);
  double gTl = 0.0, gnl = 0.0;
  rate_cotangents(w, lbk + (size_t)cell * 2 * NR, kfc, cwc, nullptr, rc, rf, nasa, tab, Tc, nd, cell, glm, gA, gJ, gTl,
                  gnl, lane);
  const double gT = wsum(gTl), gn = wsum(gnl);
  if (lane == 0) {
    gTn[2 * cell] = gT;
    gTn[2 * cell + 1] = gn;
  }
}
