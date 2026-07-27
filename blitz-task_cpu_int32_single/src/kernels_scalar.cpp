// Built with no ISA flags - the plain baseline. See BlitzKernelTiers.cmake.
#include <platform.h>

#if BLITZBENCH_ARCH_X86

#include <synthetic_ops.h>

#include "kernels.hpp"

namespace cpu_int_single {

// Scalar adds: one lane per chain. The i16 cast is load-bearing - uint16_t arithmetic promotes to int.
BLITZBENCH_ADD_KERNEL(i16_scalar, std::uint16_t, 0x9E37u, static_cast<std::uint16_t>(a + b), 1)
BLITZBENCH_ADD_KERNEL(i32_scalar, std::uint32_t, 0x9E3779B9u, a + b, 1)
BLITZBENCH_ADD_KERNEL(i64_scalar, std::uint64_t, 0x9E3779B97F4A7C15ull, a + b, 1)

}  // namespace cpu_int_single

#endif
