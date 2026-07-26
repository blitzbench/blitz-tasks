// Built with no ISA flags - NEON is baseline on AArch64. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_HAS_NEON

#include <arm_neon.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_fp_multi {

#if BLITZBENCH_ARCH_ARM64
// 128-bit fp32 FMA: 4 lanes per chain, 2 flops per lane. FMA is
// architecturally mandatory on AArch64, so it is always available and is the
// peak-FLOPS counterpart of the x86 avx2+fma tier. vfmaq(a,b,c) = a + b*c.
BLITZBENCH_FMA_KERNEL(fp32_neon, float32x4_t, vdupq_n_f32(0.999999f), vdupq_n_f32(0.5f), vfmaq_f32(c, a, b), 4)
#else
// ARMv7: no FMA kernels, add-only. 128-bit fp32 add, 4 lanes per chain.
BLITZBENCH_ADD_KERNEL(fp32_neon, float32x4_t, vdupq_n_f32(1.0f), vaddq_f32(a, b), 4)
#endif

}  // namespace cpu_fp_multi

#endif
