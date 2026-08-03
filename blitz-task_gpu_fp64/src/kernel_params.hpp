#pragma once

/**
 * @file kernel_params.hpp
 * @brief Shared parameters + CPU reference for the fp64 dense-FMA microbenchmark.
 *        Every runner implements the SAME dependent-FMA chains so the CPU reference
 *        below reproduces device output (fp64 fma is IEEE-deterministic; with explicit
 *        fma there is no contraction, so the sequence replicates exactly on the host).
 *
 * Kernel, per thread t:
 *   acc[k] = 1 + double((t+k) & 1023) / 2048           for k in [0,kChains)
 *   repeat `iters` times: kUnroll rounds of acc[k] = fma(acc[k], m, add)
 *   out[t] = sum(acc)
 * m = 0.9999, add = 0.0001 -> each chain contracts toward 1.0, so values stay
 * bounded (~1.0..1.5) and never overflow.
 *
 * This is gpu_fp32's kernel_params.hpp with float -> double. The only functional
 * difference between gpu_fp64 and gpu_fp32 is the type plus per-backend fp64
 * feature gating in the runners.
 */

#include <cmath>
#include <cstdint>
#include <gpgpu/setup.hpp>

namespace bench {
namespace fp64 {

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
inline double reference(std::uint32_t t, std::uint32_t iters) {
  double acc0 = 1.0 + static_cast<double>((t + 0u) & 1023u) * (1.0 / 2048.0);
  double acc1 = 1.0 + static_cast<double>((t + 1u) & 1023u) * (1.0 / 2048.0);
  double acc2 = 1.0 + static_cast<double>((t + 2u) & 1023u) * (1.0 / 2048.0);
  double acc3 = 1.0 + static_cast<double>((t + 3u) & 1023u) * (1.0 / 2048.0);
  const double m = 0.9999, add = 0.0001;
  for (std::uint32_t i = 0; i < iters; ++i) {
    for (int u = 0; u < kUnroll; ++u) {
      acc0 = std::fma(acc0, m, add);
      acc1 = std::fma(acc1, m, add);
      acc2 = std::fma(acc2, m, add);
      acc3 = std::fma(acc3, m, add);
    }
  }
  return acc0 + acc1 + acc2 + acc3;
}

/**
 * @brief Loose exactness check.
 *
 * Catches a wrong kernel / path while tolerating the last-ulp jitter a
 * software rasterizer might introduce. fp64 fma is exactly replicable, so
 * this is comfortably slack.
 *
 * @param got
 * @param expected
 * @return
 */
inline bool matches(double got, double expected) {
  return std::fabs(got - expected) <= 1e-4 * std::fabs(expected) + 1e-5;
}

}  // namespace fp64
}  // namespace bench
