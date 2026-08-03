#pragma once

/**
 * @file kernel_params.hpp
 * @brief Shared parameters + CPU references for the int8 dense dependent-MAC
 *        microbenchmark. Unit is GOPS (giga integer-ops/s); TOPS = GOPS / 1000.
 *
 * Two math models, each with an exact CPU reference (integer math is EXACT — no
 * tolerance; every device output is compared bit-for-bit):
 *
 * PACKED model (the SIMD/dot-product workhorse: OpenCL, Metal, Vulkan-packed,
 * Level-Zero, and the CUDA __dp4a / char4 and ROCm sdot4 / char4 fallbacks).
 * Per thread t, kChains independent dependent chains. Each chain k holds a
 * uint32 accumulator seeded from (t,k); every step reinterprets the four bytes
 * of the accumulator as SIGNED int8 lanes, dot-products them with a fixed signed
 * int8x4 weight W, and adds the (signed) dot back into the accumulator with
 * uint32 wraparound:
 *
 *   b_i  = (int8)(byte i of acc)            // signed, little-endian
 *   w_i  = (int8)(byte i of W)              // W = 0x04030201 -> {1,2,3,4}
 *   dot  = b0*w0 + b1*w1 + b2*w2 + b3*w3    // 4 mul + 3 add
 *   acc  = acc + (uint32)dot                // + 1 add  => 8 int-ops / MAC
 *
 * This is exactly the SIGNED __dp4a(acc, W, acc) / sdot4(acc, W, acc) semantics,
 * so the dp4a and char4 paths share ONE reference. Feeding acc back as the
 * multiplicand makes the chain non-linear in acc, defeating both dead-code
 * elimination and strength-reduction, so the compiler must emit every MAC.
 * out[t] = sum over chains of the final accumulator.
 *
 * OPS ACCOUNTING (documented, per the spec):
 *   * one 4-lane MAC (char4 OR one __dp4a / sdot4)  = 8 int-ops.
 *   * one 16x16x16 s8->s32 MMA instruction (wmma / mfma / coopmat, per
 *     warp/subgroup) = 2*M*N*K = 2*16*16*16 = 8192 int-ops.
 *   packed work = threads * reps * iters * kUnroll * kChains * 8.
 *   tensor work = tiles   * reps * iters * 8192   (tiles = warps/subgroups).
 *
 * TENSOR model (CUDA wmma, ROCm mfma/wmma, Vulkan coopmat). A(16x16) and
 * B(16x16) are filled with signed int8 1s, C(16x16,s32) accumulates A*B over
 * `iters` MMA instructions; every C element therefore equals 16*iters (K=16
 * products of 1*1 per instruction). Exactly reproducible on the CPU.
 */

#include <cstdint>
#include <gpgpu/setup.hpp>

namespace bench {
namespace i8 {

inline constexpr int kChains = 4;                     // independent accumulator chains
inline constexpr int kUnroll = 8;                     // MACs per chain per iteration
inline constexpr std::uint32_t kIters = 1024;         // main-run inner iterations
inline constexpr std::uint32_t kPreflightIters = 16;  // tiny exactness pre-flight
inline constexpr std::uint32_t kPreflightThreads = 256;
inline constexpr std::uint32_t kBlock = 256;  // workgroup / block size

inline constexpr std::uint32_t kWeight = 0x04030201u;  // signed int8 lanes {1,2,3,4}

// Per-(thread,chain) accumulator seed. Golden-ratio / xxhash-style constants so
// neighbouring threads diverge quickly; plain uint32 wraparound.
inline std::uint32_t seed(std::uint32_t t, std::uint32_t k) { return 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * (k + 1u); }

/**
 * @brief Get a approximately good thread count.
 *
 * T = clamp(compute_units * 2048, 65536, 1<<20). (Identical to every compute app.)
 *
 * @param d
 * @return
 */
inline std::uint32_t thread_count(const gpgpu::Device& d) {
  std::uint64_t cu = d.compute_units().value_or(32u);
  std::uint64_t t = cu * 2048ull;
  if (t < 65536ull) t = 65536ull;
  if (t > (1ull << 20)) t = (1ull << 20);
  return static_cast<std::uint32_t>(t);
}

// --- packed model -----------------------------------------------------------

// Signed value of byte `i` (0..3) of a 32-bit word, little-endian.
inline int sbyte(std::uint32_t v, int i) { return static_cast<int>(static_cast<std::int8_t>((v >> (8 * i)) & 0xffu)); }

// One 4-lane signed MAC step: acc += dot4(bytes(acc), bytes(W)), uint32 wrap.
inline std::uint32_t mac_step(std::uint32_t acc, std::uint32_t w) {
  const int dot = sbyte(acc, 0) * sbyte(w, 0) + sbyte(acc, 1) * sbyte(w, 1) + sbyte(acc, 2) * sbyte(w, 2) +
                  sbyte(acc, 3) * sbyte(w, 3);
  return acc + static_cast<std::uint32_t>(dot);
}

// CPU reference for out[t] of the packed kernels. Mirrors every packed device
// kernel exactly.
inline std::uint32_t reference_packed(std::uint32_t t, std::uint32_t iters) {
  std::uint32_t sum = 0;
  for (int k = 0; k < kChains; ++k) {
    std::uint32_t acc = seed(t, static_cast<std::uint32_t>(k));
    for (std::uint32_t i = 0; i < iters; ++i)
      for (int u = 0; u < kUnroll; ++u) acc = mac_step(acc, kWeight);
    sum += acc;
  }
  return sum;
}

inline std::uint64_t ops_packed(std::uint32_t threads, std::uint32_t iters, int reps) {
  return static_cast<std::uint64_t>(threads) * static_cast<std::uint64_t>(reps) * static_cast<std::uint64_t>(iters) *
         kUnroll * kChains * 8ull;
}

// --- tensor model -----------------------------------------------------------

// Every C element after `iters` 16x16x16 s8->s32 MMAs with all-ones s8 inputs.
inline std::int32_t reference_tensor(std::uint32_t iters) { return 16 * static_cast<std::int32_t>(iters); }

inline std::uint64_t ops_tensor(std::uint32_t tiles, std::uint32_t iters, int reps) {
  return static_cast<std::uint64_t>(tiles) * static_cast<std::uint64_t>(reps) * static_cast<std::uint64_t>(iters) *
         8192ull;
}

}  // namespace i8
}  // namespace bench
