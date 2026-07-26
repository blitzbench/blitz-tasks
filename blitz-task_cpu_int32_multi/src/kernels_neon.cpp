// Built with no ISA flags - NEON is baseline on AArch64. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_HAS_NEON

#include <arm_neon.h>
#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_int_multi {

// 128-bit adds: i16 x8 / i32 x4 / i64 x2 lanes per chain.
BLITZBENCH_ADD_KERNEL(i16_neon, uint16x8_t, vdupq_n_u16(0x1D2B),              vaddq_u16(a, b), 8)
BLITZBENCH_ADD_KERNEL(i32_neon, uint32x4_t, vdupq_n_u32(0x1D2B3C4Au),         vaddq_u32(a, b), 4)
BLITZBENCH_ADD_KERNEL(i64_neon, uint64x2_t, vdupq_n_u64(0x1D2B3C4A5E6F70ull), vaddq_u64(a, b), 2)

} // namespace cpu_int_multi

#endif
