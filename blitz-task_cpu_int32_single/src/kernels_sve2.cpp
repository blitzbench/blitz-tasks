// Built with -march=armv8-a+sve2 (AArch64). See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_HAS_SVE2

#include <arm_sve.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_int_single {

// SVE2 adds: identical mix to the sve tier (see kernels_sve.cpp).
BLITZBENCH_ADD_KERNEL(i16_sve2, svuint16_t, svdup_n_u16(0x1D2B), svadd_u16_x(svptrue_b16(), a, b), svcnth())
BLITZBENCH_ADD_KERNEL(i32_sve2, svuint32_t, svdup_n_u32(0x1D2B3C4Au), svadd_u32_x(svptrue_b32(), a, b), svcntw())
BLITZBENCH_ADD_KERNEL(i64_sve2, svuint64_t, svdup_n_u64(0x1D2B3C4A5E6F70ull), svadd_u64_x(svptrue_b64(), a, b),
                      svcntd())

}  // namespace cpu_int_single

#endif
