// Built with -mavx2 -mfma. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_ARCH_X86

#include <immintrin.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_fp_single {

// 256-bit fp32 FMA: 8 lanes per chain, 2 flops per lane.
BLITZBENCH_FMA_KERNEL(fp32_avx2_fma, __m256, _mm256_set1_ps(0.999999f), _mm256_set1_ps(0.5f),
                      _mm256_fmadd_ps(a, b, c), 8)

} // namespace cpu_fp_single

#endif
