// Built with -mavx2. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_ARCH_X86

#include <immintrin.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_int_multi {

// 256-bit adds: i16 x16 / i32 x8 / i64 x4 lanes per chain.
BLITZBENCH_ADD_KERNEL(i16_avx2, __m256i, _mm256_set1_epi16(0x1D2B),        _mm256_add_epi16(a, b), 16)
BLITZBENCH_ADD_KERNEL(i32_avx2, __m256i, _mm256_set1_epi32(0x1D2B3C4A),    _mm256_add_epi32(a, b),  8)
BLITZBENCH_ADD_KERNEL(i64_avx2, __m256i, _mm256_set1_epi64x(0x1D2B3C4All), _mm256_add_epi64(a, b),  4)

} // namespace cpu_int_multi

#endif
