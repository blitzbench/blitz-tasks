#pragma once

/**
 * @file kernels.hpp
 * @brief Per-tier fp32 throughput kernel wrappers for cpu_fp_multi.
 *
 * Each wrapper is defined in its own TU, compiled with only that tier's -m
 * flags (see BlitzKernelTiers.cmake), and instantiates the generic 8-chain
 * loop from <synthetic_ops.h> with that tier's traits. This header carries no
 * ISA flags and is safe to include anywhere.
 *
 * Only the current architecture's wrappers are declared - and defined - so the
 * dispatch table in cpu_fp_multi.cpp must wire slots under the same guards.
 */

#include <platform.h>

#include <cstdint>

namespace cpu_fp_multi {

/// Runs the tier's fp32 kernel for `iters` loop iterations and returns the
/// exact number of floating-point operations performed
/// (flops/s = retval / elapsed seconds).
using OpsKernel = std::uint64_t (*)(std::uint64_t iters);

#if BLITZBENCH_ARCH_X86
std::uint64_t fp32_scalar(std::uint64_t iters);
std::uint64_t fp32_sse3(std::uint64_t iters);
std::uint64_t fp32_sse4(std::uint64_t iters);
std::uint64_t fp32_avx(std::uint64_t iters);
std::uint64_t fp32_avx2(std::uint64_t iters);
std::uint64_t fp32_avx2_fma(std::uint64_t iters);
std::uint64_t fp32_avx512(std::uint64_t iters);
#elif BLITZBENCH_HAS_NEON
std::uint64_t fp32_neon(std::uint64_t iters);
#if BLITZBENCH_ARCH_ARM64
// SVE is AArch64-only; each is defined by its own +sve / +sve2 TU.
std::uint64_t fp32_sve(std::uint64_t iters);
std::uint64_t fp32_sve2(std::uint64_t iters);
#endif
#endif

} // namespace cpu_fp_multi
