// Built with -mavx2. See BlitzKernelTiers.cmake.
//
// For fp32 the instruction mix is identical to the avx tier (AVX2 adds
// 256-bit INTEGER arithmetic); the tier exists for per-ISA binary parity.
#include <platform.h>

#if BLITZBENCH_ARCH_X86

#include <immintrin.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_fp_multi {

// 256-bit fp32 add: 8 lanes per chain.
BLITZBENCH_ADD_KERNEL(fp32_avx2, __m256, _mm256_set1_ps(1.0f), _mm256_add_ps(a, b), 8)

} // namespace cpu_fp_multi

#endif
