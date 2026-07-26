// Built with -msse3. See BlitzKernelTiers.cmake.
//
// The adds are SSE2 instructions - this tier exists for per-ISA binaries with an identical instruction mix.
#include <platform.h>

#if BLITZBENCH_ARCH_X86

#include <immintrin.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_fp_multi {

// 128-bit fp32 add: 4 lanes per chain.
BLITZBENCH_ADD_KERNEL(fp32_sse3, __m128, _mm_set1_ps(1.0f), _mm_add_ps(a, b), 4)

} // namespace cpu_fp_multi

#endif
