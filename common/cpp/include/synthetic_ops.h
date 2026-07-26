#pragma once

/**
 * @file synthetic_ops.h
 * @brief Generic compute-throughput kernels: 8 independent dependency chains
 *        of `acc = add(acc, b)` or `acc = fma(acc, b, c)`, entirely in
 *        registers.
 *
 * These templates replace the former macro-generated kernels in
 * cpu/synthetic_kernels{,_neon,_sve}.h. Each benchmark task instantiates them
 * per SIMD tier in its own TU, compiled with only that tier's -m flags (see
 * BlitzKernelTiers.cmake), by providing a small Traits struct that wraps the
 * tier's type and intrinsics:
 *
 * @code
 * struct Fp32Avx2 {
 *   using Vec = __m256;
 *   static Vec operand_b() { return _mm256_set1_ps(1.0f); }
 *   static Vec add(Vec a, Vec b) { return _mm256_add_ps(a, b); }
 *   static std::uint64_t lanes() { return 8; }
 * };
 * std::uint64_t fp32_avx2(std::uint64_t iters) { return bench::add_ops<Fp32Avx2>(iters); }
 * @endcode
 *
 * Design: 8 independent chains ("acc = acc OP b") saturate the ALU/FP ports
 * of every current core with zero register spills; all state lives in
 * registers. Anti-elision is delegated to the typed register barriers in
 * optimization_barrier.h (empty asm on GNU compilers, volatile globals on
 * cl.exe - see there for the full rationale). Narrowing integer types
 * (e.g. uint16_t) must cast in their Traits::add, since scalar arithmetic
 * promotes to int.
 *
 * Sanity checks for the numbers these produce:
 *  - A 4 GHz core with 4 scalar-add ALUs -> ~16 Gops/s scalar i32; AVX2 i32
 *    near ports*8lanes*GHz; fp32 FMA near fma_ports*8*2*GHz. If an FMA kernel
 *    lands far below the add kernels' rate x2, inspect the loop disassembly
 *    for stack traffic (vmovaps [rsp..]) - it must contain only the FMA ops
 *    and the loop counter.
 *  - AVX-512 caveats: some Intel server SKUs downclock under sustained
 *    512-bit work and some have 2x512-bit FMA units vs 1 on others; Zen 4
 *    double-pumps 512-bit ops so avx512 ~= avx2 there. All of that is the
 *    correct measurement of the hardware, not a kernel deficiency - just do
 *    not convert to ops/CYCLE using the base clock.
 *  - SVE is vector-length agnostic: Traits::lanes() is taken at runtime
 *    (svcntw & co), so the returned op count is exact for the actual VL.
 *    Always report cpu_features().sve_vector_bytes next to SVE results.
 */

#include <platform.h>
#include <optimization_barrier.h>

#include <cstdint>

namespace bench {

/// Number of independent dependency chains per kernel. 8 saturates the
/// execution ports of every current x86/ARM core with zero register spills.
inline constexpr unsigned kOpsChains = 8;

/**
 * @fn add_ops
 * @brief Runs `iters` loop iterations of 8 chained `acc = add(acc, b)` ops.
 *
 * The operand is made opaque before the loop and the 8 accumulators are
 * re-declared read+written after every iteration, so the loop can be neither
 * folded nor auto-vectorized. The chains are pairwise-reduced after the loop
 * and sunk as a register input, keeping every chain live without taking an
 * address.
 *
 * @tparam Traits tier traits: Vec, operand_b(), add(Vec, Vec), lanes()
 * @param iters number of loop iterations
 * @return the exact number of operations performed: iters * 8 chains * lanes
 *         (ops/s = retval / elapsed seconds)
 */
template <class Traits>
inline std::uint64_t add_ops(std::uint64_t iters) {
  using V = typename Traits::Vec;
  V b = Traits::operand_b();
  reg_opaque<0>(b);
  V a0 = b, a1 = b, a2 = b, a3 = b, a4 = b, a5 = b, a6 = b, a7 = b;
  BLITZBENCH_NOVEC_LOOP
  for (std::uint64_t i = 0; i < iters; ++i) {
    const V bb = reg_reload<0>(b);
    a0 = Traits::add(a0, bb);
    a1 = Traits::add(a1, bb);
    a2 = Traits::add(a2, bb);
    a3 = Traits::add(a3, bb);
    a4 = Traits::add(a4, bb);
    a5 = Traits::add(a5, bb);
    a6 = Traits::add(a6, bb);
    a7 = Traits::add(a7, bb);
    reg_keep8(a0, a1, a2, a3, a4, a5, a6, a7);
  }
  V s = Traits::add(Traits::add(Traits::add(a0, a1), Traits::add(a2, a3)),
                    Traits::add(Traits::add(a4, a5), Traits::add(a6, a7)));
  reg_sink(s);
  return iters * static_cast<std::uint64_t>(kOpsChains) * static_cast<std::uint64_t>(Traits::lanes());
}

/**
 * @fn fma_ops
 * @brief Runs `iters` loop iterations of 8 chained `acc = acc*b + c` FMA ops
 *        (2 flops per lane).
 *
 * With b < 1 every accumulator converges to c/(1-b) and stays normal forever
 * (no inf, no denormals). All 8 accumulators are sunk through ONE
 * register-input barrier - address-taking sinks would force GCC to spill
 * every accumulator to the stack inside the loop.
 *
 * @tparam Traits tier traits: Vec, operand_b() (the multiplier, < 1),
 *         operand_c() (the addend), fma(acc, b, c) -> acc*b + c, lanes()
 * @param iters number of loop iterations
 * @return the exact number of operations performed:
 *         iters * 8 chains * lanes * 2 (flops/s = retval / elapsed seconds)
 */
template <class Traits>
inline std::uint64_t fma_ops(std::uint64_t iters) {
  using V = typename Traits::Vec;
  V b = Traits::operand_b();
  V c = Traits::operand_c();
  reg_opaque<0>(b);
  reg_opaque<1>(c);
  V a0 = c, a1 = c, a2 = c, a3 = c, a4 = c, a5 = c, a6 = c, a7 = c;
  BLITZBENCH_NOVEC_LOOP
  for (std::uint64_t i = 0; i < iters; ++i) {
    const V bb = reg_reload<0>(b);
    const V cc = reg_reload<1>(c);
    a0 = Traits::fma(a0, bb, cc);
    a1 = Traits::fma(a1, bb, cc);
    a2 = Traits::fma(a2, bb, cc);
    a3 = Traits::fma(a3, bb, cc);
    a4 = Traits::fma(a4, bb, cc);
    a5 = Traits::fma(a5, bb, cc);
    a6 = Traits::fma(a6, bb, cc);
    a7 = Traits::fma(a7, bb, cc);
    reg_keep8(a0, a1, a2, a3, a4, a5, a6, a7);
  }
  reg_sink8(a0, a1, a2, a3, a4, a5, a6, a7);
  return iters * static_cast<std::uint64_t>(kOpsChains) * static_cast<std::uint64_t>(Traits::lanes()) * 2ull;
}

}  // namespace bench

/// Stamps an 8-chain add-throughput kernel `NAME(iters)` (bench::add_ops).
#define BLITZBENCH_ADD_KERNEL(NAME, VT, B_INIT, ADD_EXPR, LANES)              \
  namespace {                                                                 \
  struct NAME##_traits {                                                      \
    using Vec = VT;                                                           \
    static Vec operand_b() { return (B_INIT); }                               \
    static Vec add(Vec a, Vec b) { return (ADD_EXPR); }                       \
    static std::uint64_t lanes() { return (LANES); }                          \
  };                                                                          \
  }                                                                           \
  std::uint64_t NAME(std::uint64_t iters) { return ::bench::add_ops<NAME##_traits>(iters); }

/// Stamps an 8-chain FMA-throughput kernel `NAME(iters)` (bench::fma_ops,
/// 2 flops/lane). Pick B_INIT < 1 so accumulators converge to c/(1-b) and
/// stay normal forever.
#define BLITZBENCH_FMA_KERNEL(NAME, VT, B_INIT, C_INIT, FMA_EXPR, LANES)      \
  namespace {                                                                 \
  struct NAME##_traits {                                                      \
    using Vec = VT;                                                           \
    static Vec operand_b() { return (B_INIT); }                               \
    static Vec operand_c() { return (C_INIT); }                               \
    static Vec fma(Vec a, Vec b, Vec c) { return (FMA_EXPR); }                \
    static std::uint64_t lanes() { return (LANES); }                          \
  };                                                                          \
  }                                                                           \
  std::uint64_t NAME(std::uint64_t iters) { return ::bench::fma_ops<NAME##_traits>(iters); }
