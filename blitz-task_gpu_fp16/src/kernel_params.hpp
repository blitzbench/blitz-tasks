#pragma once

/**
 * @file kernel_params.hpp
 * @brief Shared parameters + CPU references for the fp16 throughput microbenchmark.
 *
 * Two kernel shapes are selected at runtime and reported via RunResult.path:
 *
 *   * TENSOR path — a warp/subgroup-collective matrix multiply-accumulate
 *     (CUDA wmma, ROCm MFMA/WMMA builtins, Vulkan cooperative_matrix, Metal
 *     simdgroup_matrix). Every A/B element is the exact half 1/16 (0.0625) and
 *     the accumulator starts at 0, so after `iters` MMAs every C element equals
 *     iters * K * (1/16)^2 EXACTLY in fp32 (K = the instruction's contraction
 *     dim). That makes the tensor path exactly verifiable on the CPU.
 *
 *   * SIMD/packed path — dependent packed-half FMA chains (half2 / half4 /
 *     half8). Each lane runs `acc = fma(acc, m, a)`; with the contracting
 *     constants (m = 1-2^-11, a = 2^-11) every lane converges toward 1.0 and
 *     stays bounded, so the main run is checked with a bounded envelope. A
 *     pre-flight pass runs the SAME kernel with the exact halves (m = 1, a = 1)
 *     so `acc = seed + steps` is integer-exact and verifiable.
 *
 * Every backend kernel MUST use the seed formula and constants below so these
 * references reproduce device output.
 */

#include <cmath>
#include <cstdint>
#include <gpgpu/setup.hpp>

namespace bench {
namespace fp16 {

inline constexpr std::uint32_t kBlock = 256;    // workgroup / block size
inline constexpr std::uint32_t kWarpSize = 32;  // lanes per warp/subgroup for the tensor path

// ---------------------------------------------------------------- SIMD path --
inline constexpr int kChains = 4;                    // packed accumulator chains per thread
inline constexpr int kUnroll = 8;                    // FMAs per chain per iteration
inline constexpr std::uint32_t kIters = 1024;        // main-run inner iterations
inline constexpr std::uint32_t kPreflightIters = 8;  // tiny exact pre-flight
inline constexpr std::uint32_t kPreflightThreads = 256;

inline constexpr float kMainM = 0.9375f;  // 15 * 2^-4
inline constexpr float kMainA = 0.0625f;  // 2^-4
// Exact pre-flight constants: acc = acc*1 + 1 -> acc = seed + steps.
inline constexpr float kPreM = 1.0f;
inline constexpr float kPreA = 1.0f;

// Per-lane seed, identical in every kernel and in the references below. The
// value is a multiple of 2^-8 in [1,2) and therefore exact in fp16.
inline double seed_val(std::uint32_t t, int chain, int lane) {
  const std::uint32_t idx =
      (t + 7u * static_cast<std::uint32_t>(chain) + 13u * static_cast<std::uint32_t>(lane)) & 255u;
  return 1.0 + static_cast<double>(idx) * (1.0 / 256.0);
}

/**
 * @brief Pre-flight (m=1,a=1) exact output for thread t summed over chains*lanes:
 *        element = seed + steps ; out = sum(element).
 *
 * @param t
 * @param lanes
 * @param iters
 * @return
 */
inline double simd_preflight_out(std::uint32_t t, int lanes, std::uint32_t iters) {
  const double steps = static_cast<double>(iters) * kUnroll;  // FMAs per chain
  double s = 0.0;
  for (int k = 0; k < kChains; ++k)
    for (int l = 0; l < lanes; ++l) s += seed_val(t, k, l) + steps;
  return s;
}

/**
 * @brief Match for the pre-flight / tensor checks.
 *
 * @param got
 * @param expected
 * @return
 */
inline bool matches_exact(double got, double expected) {
  return std::fabs(got - expected) <= 2e-2 * std::fabs(expected) + 1e-2;
}

/**
 * @brief Main-run bounded envelope: every lane converges to ~1.0 so the per-thread sum
 *        sits close to chains*lanes.
 *
 * A dead / wrong kernel (0, garbage, NaN, or the un-contracted seed sum ~1.5x) falls
 * outside this band.
 *
 * @param got
 * @param lanes
 * @return
 */
inline bool simd_main_ok(double got, int lanes) {
  const double expected = static_cast<double>(kChains) * lanes;
  return got > 0.90 * expected && got < 1.10 * expected;
}

// flops for a full timed run: one FMA op per (chain*unroll) per iter, each op is
// 2 flops per lane (mul+add). half2 => lanes=2 => 4 flops/op, matching the spec.
inline std::uint64_t simd_flops(std::uint32_t threads, std::uint32_t iters, int reps, int lanes) {
  return static_cast<std::uint64_t>(threads) * static_cast<std::uint64_t>(reps) * static_cast<std::uint64_t>(iters) *
         kUnroll * kChains * 2ull * static_cast<std::uint64_t>(lanes);
}

// --- tensor path ---
inline constexpr float kTensorVal = 0.0625f;          // 1/16, exact in fp16
inline constexpr std::uint32_t kTensorIters = 16384;  // main-run MMAs per warp
inline constexpr std::uint32_t kTensorPreflightIters = 4;

inline double tensor_out(std::uint32_t iters, int K) {
  return static_cast<double>(iters) * static_cast<double>(K) * static_cast<double>(kTensorVal) *
         static_cast<double>(kTensorVal);
}

inline bool tensor_ok(double got, std::uint32_t iters, int K) { return matches_exact(got, tensor_out(iters, K)); }

/**
 * @brief flops for a full timed run of a tensor tile MxNxK
 *
 * 2*M*N*K flops per MMA per warp (e.g. 16x16x16 => 8192).
 *
 * @param warps
 * @param iters
 * @param reps
 * @param M
 * @param N
 * @param K
 * @return
 */
inline std::uint64_t tensor_flops(std::uint32_t warps, std::uint32_t iters, int reps, int M, int N, int K) {
  return static_cast<std::uint64_t>(warps) * static_cast<std::uint64_t>(reps) * static_cast<std::uint64_t>(iters) *
         (2ull * M * N * K);
}

/**
 * @brief Get a approximately good thread count.
 *
 * T = clamp(compute_units * 2048, 65536, 1<<20).
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

}  // namespace fp16
}  // namespace bench
