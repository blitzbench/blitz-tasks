// Built with -mavx512f -mavx512bw. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_ARCH_X86

#include <immintrin.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_int_multi {

// 512-bit adds: i16 x32 / i32 x16 / i64 x8 lanes per chain.
#if defined(__AVX512BW__)
BLITZBENCH_ADD_KERNEL(i16_avx512, __m512i, _mm512_set1_epi16(0x1D2B),      _mm512_add_epi16(a, b), 32)
#endif
BLITZBENCH_ADD_KERNEL(i32_avx512, __m512i, _mm512_set1_epi32(0x1D2B3C4A),  _mm512_add_epi32(a, b), 16)
BLITZBENCH_ADD_KERNEL(i64_avx512, __m512i, _mm512_set1_epi64(0x1D2B3C4All), _mm512_add_epi64(a, b), 8)

} // namespace cpu_int_multi

#endif
