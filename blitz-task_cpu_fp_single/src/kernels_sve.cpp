// Built with -march=armv8-a+sve (AArch64). See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_HAS_SVE

#include <arm_sve.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_fp_single {

// SVE fp32 FMA: svcntw() lanes per chain, 2 flops per lane.
BLITZBENCH_FMA_KERNEL(fp32_sve, svfloat32_t, svdup_n_f32(0.999999f), svdup_n_f32(0.5f),
                      svmla_f32_x(svptrue_b32(), c, a, b), svcntw())

} // namespace cpu_fp_single

#endif
