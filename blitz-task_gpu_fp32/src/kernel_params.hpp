#pragma once

/**
 * @file kernel_params.hpp
 * @brief Shared parameters + CPU reference for the fp32 dense-FMA microbenchmark.
 *        Every runner implements the SAME dependent-FMA chains so the CPU reference
 *        below reproduces device output bit-for-bit (fp32 fma is IEEE-deterministic).
 *
 * Kernel, per thread t:
 *   acc[k] = 1 + float((t+k) & 1023) / 2048           for k in [0,kChains)
 *   repeat `iters` times: kUnroll rounds of acc[k] = fma(acc[k], m, add)
 *   out[t] = sum(acc)
 * m = 0.9999, add = 0.0001 → each chain contracts toward 1.0, so values stay
 * bounded (~1.0..1.5) and never overflow.
 */

#include <cmath>
#include <cstdint>
#include <gpgpu/setup.hpp>

namespace bench {
namespace fp32 {

inline constexpr int kChains = 4;                     // independent accumulator chains
inline constexpr int kUnroll = 8;                     // FMAs per chain per iteration
inline constexpr std::uint32_t kIters = 1024;         // main-run inner iterations
inline constexpr std::uint32_t kPreflightIters = 16;  // tiny exactness pre-flight
inline constexpr std::uint32_t kPreflightThreads = 256;
inline constexpr std::uint32_t kBlock = 256;  // workgroup / block size

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

// flops for a full timed run of `reps` launches of `iters` (FMA = 2 flops).
inline std::uint64_t flops(std::uint32_t threads, std::uint32_t iters, int reps) {
  return static_cast<std::uint64_t>(threads) * static_cast<std::uint64_t>(reps) * static_cast<std::uint64_t>(iters) *
         kUnroll * kChains * 2ull;
}

// CPU reference for output[t]. Must mirror every device kernel exactly.
inline float reference(std::uint32_t t, std::uint32_t iters) {
  float acc0 = 1.0f + static_cast<float>((t + 0u) & 1023u) * (1.0f / 2048.0f);
  float acc1 = 1.0f + static_cast<float>((t + 1u) & 1023u) * (1.0f / 2048.0f);
  float acc2 = 1.0f + static_cast<float>((t + 2u) & 1023u) * (1.0f / 2048.0f);
  float acc3 = 1.0f + static_cast<float>((t + 3u) & 1023u) * (1.0f / 2048.0f);
  const float m = 0.9999f, add = 0.0001f;
  for (std::uint32_t i = 0; i < iters; ++i) {
    for (int u = 0; u < kUnroll; ++u) {
      acc0 = std::fmaf(acc0, m, add);
      acc1 = std::fmaf(acc1, m, add);
      acc2 = std::fmaf(acc2, m, add);
      acc3 = std::fmaf(acc3, m, add);
    }
  }
  return acc0 + acc1 + acc2 + acc3;
}

/**
 * @brief Loose exactness check.
 *
 * Catches a wrong kernel / path while tolerating the last-ulp jitter a
 * software rasterizer might introduce.
 *
 * @param got
 * @param expected
 * @return
 */
inline bool matches(float got, float expected) {
  return std::fabs(got - expected) <= 1e-4f * std::fabs(expected) + 1e-5f;
}

}  // namespace fp32
}  // namespace bench
