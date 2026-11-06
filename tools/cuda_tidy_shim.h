// Stand-in for what NVRTC supplies by itself, so clang-tidy can parse a kernel with no CUDA toolkit installed.
// Force-included through compile_flags.txt. Declarations only: nothing here reaches a compiled kernel.
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __host__ __attribute__((host))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __restrict__ __restrict
#include <__clang_cuda_builtin_vars.h>
#include <stddef.h>
extern "C" __device__ void* memcpy(void*, const void*, size_t);
extern "C" __device__ void* memset(void*, int, size_t);
__device__ double exp(double);
__device__ double log(double);
__device__ double log10(double);
__device__ double sqrt(double);
__device__ double fabs(double);
__device__ double floor(double);
__device__ double ceil(double);
__device__ double trunc(double);
__device__ double rint(double);
__device__ double fmax(double, double);
__device__ double fmin(double, double);
__device__ double pow(double, double);
__device__ double fma(double, double, double);
__device__ int isfinite(double);
__device__ int isnan(double);
__device__ int isinf(double);
__device__ int abs(int);
__device__ int min(int, int);
__device__ int max(int, int);
template <typename T>
__device__ T __shfl_sync(unsigned, T, int, int = 32);
template <typename T>
__device__ T __shfl_xor_sync(unsigned, T, int, int = 32);
template <typename T>
__device__ T __shfl_up_sync(unsigned, T, unsigned, int = 32);
template <typename T>
__device__ T __shfl_down_sync(unsigned, T, unsigned, int = 32);
__device__ void __syncthreads();
__device__ void __syncwarp(unsigned = 0xffffffff);
template <typename T>
__device__ T __ldg(const T*);
template <typename T>
__device__ T atomicAdd(T*, T);
template <typename T>
__device__ T atomicMax(T*, T);
template <typename T>
__device__ T atomicMin(T*, T);
template <typename T>
__device__ T atomicOr(T*, T);
template <typename T>
__device__ T atomicCAS(T*, T, T);
__device__ double __longlong_as_double(long long);
__device__ long long __double_as_longlong(double);
__device__ double __dsub_rn(double, double);
__device__ double __dsub_rz(double, double);
__device__ double __dadd_rn(double, double);
__device__ double __ddiv_rn(double, double);
__device__ double __dmul_rn(double, double);
__device__ double exp10(double);
__device__ double cbrt(double);
__device__ int __all_sync(unsigned, int);
__device__ int __any_sync(unsigned, int);
__device__ unsigned __ballot_sync(unsigned, int);
__device__ int __popc(unsigned);
__device__ int __popcll(unsigned long long);
__device__ int __ffs(int);
__device__ int __ffsll(long long);
__device__ int __clz(int);
__device__ int __clzll(long long);
__device__ unsigned __brev(unsigned);
