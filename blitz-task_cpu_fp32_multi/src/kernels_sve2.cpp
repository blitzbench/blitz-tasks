// Built with -march=armv8-a+sve2 (AArch64). See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_HAS_SVE2

#include <arm_sve.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_fp_multi {

// SVE2 fp32 FMA: identical mix to the sve tier (see kernels_sve.cpp).
BLITZBENCH_FMA_KERNEL(fp32_sve2, svfloat32_t, svdup_n_f32(0.999999f), svdup_n_f32(0.5f),
                      svmla_f32_x(svptrue_b32(), c, a, b), svcntw())

} // namespace cpu_fp_multi

#endif
