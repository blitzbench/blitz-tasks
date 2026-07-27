// Built with -msse4.1. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_ARCH_X86

#include <immintrin.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_int_single {

// 128-bit adds: i16 x8 / i32 x4 / i64 x2 lanes per chain.
BLITZBENCH_ADD_KERNEL(i16_sse4, __m128i, _mm_set1_epi16(0x1D2B), _mm_add_epi16(a, b), 8)
BLITZBENCH_ADD_KERNEL(i32_sse4, __m128i, _mm_set1_epi32(0x1D2B3C4A), _mm_add_epi32(a, b), 4)
BLITZBENCH_ADD_KERNEL(i64_sse4, __m128i, _mm_set1_epi64x(0x1D2B3C4All), _mm_add_epi64(a, b), 2)

}  // namespace cpu_int_single

#endif
