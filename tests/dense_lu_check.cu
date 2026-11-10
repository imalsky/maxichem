// Test harness: the recovery's dense_lu and dense_solve on given matrices, one warp each (tests/test_dense_lu.py).
// clang-format off: each file needs the ones before it
#include "kincuda.cu"
// clang-format on

// Factors matrix m of A (row-major NS x NS, in place) with its 1 / U_ii into dinv and row order into pr, then solves
// y <- A^-1 y with the factors. One block of 32 threads per matrix.
extern "C" __global__ void __launch_bounds__(32) dense_lu_check(double* A, double* dinv, int* pr, double* y) {
  const size_t m = blockIdx.x;
  const int lane = (int)(threadIdx.x & 31);
  dense_lu(A + m * NS * NS, dinv + m * NS, pr + m * NS, lane);
  dense_solve(A + m * NS * NS, dinv + m * NS, pr + m * NS, y + m * NS, lane);
}
