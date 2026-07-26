// Built with -mavx512f. See BlitzKernelTiers.cmake.
//
// FMA is part of AVX512F, so -mavx512f alone is the safe, minimal flag for this tier's peak-FLOPS kernel.
#include <platform.h>

#if BLITZBENCH_ARCH_X86

#include <immintrin.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_fp_multi {

// 512-bit fp32 FMA: 16 lanes per chain, 2 flops per lane.
BLITZBENCH_FMA_KERNEL(fp32_avx512, __m512, _mm512_set1_ps(0.999999f), _mm512_set1_ps(0.5f),
                      _mm512_fmadd_ps(a, b, c), 16)

} // namespace cpu_fp_multi

#endif
