#pragma once

/**
 * @file kernels.hpp
 * @brief Per-tier integer throughput kernel wrappers for cpu_int_single.
 *
 * Each wrapper is defined in its own TU, compiled with only that tier's -m
 * flags (see BlitzKernelTiers.cmake), and instantiates the generic 8-chain
 * loop from <synthetic_ops.h> with that tier's traits. This header carries no
 * ISA flags and is safe to include anywhere.
 *
 * Every tier provides i16 / i32 / i64 kernels; the task dispatches and
 * reports i32 (see cpu_int_single.cpp), the other widths exist for parity and
 * future use. There is no FMA tier: FMA is floating-point only.
 *
 * Only the current architecture's wrappers are declared - and defined - so the
 * dispatch table in cpu_int_single.cpp must wire slots under the same guards.
 */

#include <platform.h>

#include <cstdint>

namespace cpu_int_single {

/// Runs the tier's integer kernel for `iters` loop iterations and returns the
/// exact number of integer operations performed
/// (ops/s = retval / elapsed seconds).
using OpsKernel = std::uint64_t (*)(std::uint64_t iters);

#if BLITZBENCH_ARCH_X86
std::uint64_t i16_scalar(std::uint64_t iters);
std::uint64_t i32_scalar(std::uint64_t iters);
std::uint64_t i64_scalar(std::uint64_t iters);

std::uint64_t i16_sse3(std::uint64_t iters);
std::uint64_t i32_sse3(std::uint64_t iters);
std::uint64_t i64_sse3(std::uint64_t iters);

std::uint64_t i16_sse4(std::uint64_t iters);
std::uint64_t i32_sse4(std::uint64_t iters);
std::uint64_t i64_sse4(std::uint64_t iters);

std::uint64_t i16_avx(std::uint64_t iters);
std::uint64_t i32_avx(std::uint64_t iters);
std::uint64_t i64_avx(std::uint64_t iters);

std::uint64_t i16_avx2(std::uint64_t iters);
std::uint64_t i32_avx2(std::uint64_t iters);
std::uint64_t i64_avx2(std::uint64_t iters);

// i16_avx512 needs AVX512BW (its TU is built with -mavx512f -mavx512bw, see
// CMakeLists.txt); dispatch it only when cpu_features().avx512bw is set.
std::uint64_t i16_avx512(std::uint64_t iters);
std::uint64_t i32_avx512(std::uint64_t iters);
std::uint64_t i64_avx512(std::uint64_t iters);
#elif BLITZBENCH_HAS_NEON
std::uint64_t i16_neon(std::uint64_t iters);
std::uint64_t i32_neon(std::uint64_t iters);
std::uint64_t i64_neon(std::uint64_t iters);

#if BLITZBENCH_ARCH_ARM64
// SVE is AArch64-only; each is defined by its own +sve / +sve2 TU.
std::uint64_t i16_sve(std::uint64_t iters);
std::uint64_t i32_sve(std::uint64_t iters);
std::uint64_t i64_sve(std::uint64_t iters);

std::uint64_t i16_sve2(std::uint64_t iters);
std::uint64_t i32_sve2(std::uint64_t iters);
std::uint64_t i64_sve2(std::uint64_t iters);
#endif
#endif

}  // namespace cpu_int_single
