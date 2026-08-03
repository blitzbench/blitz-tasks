#pragma once

/**
 * @file kernel_params.hpp
 * @brief Shared sizing / pattern / verification for the gpu_vram_read microbenchmark.
 *
 * The buffer is prefilled (untimed) with pattern u32[i] = i * 2654435761u. A
 * grid-stride kernel of 128-bit (uint4 / uvec4) loads accumulates, per thread,
 * the u32 values it reads with natural 2^32 wraparound and writes ONE uint per
 * thread to a small out-buffer (defeats dead-code elimination). Because every
 * u32 element is read exactly once across the grid, the sum of all per-thread
 * outputs (u32 wraparound) equals, in closed form:
 *
 *     sum_i (i * M)  ==  M * N*(N-1)/2   (mod 2^32),   N = S/4, M = 2654435761u
 *
 * which is independent of how work is partitioned across threads.
 * work = reps * S (reads only).
 */

#include <cstddef>
#include <cstdint>
#include <gpgpu/setup.hpp>

namespace bench {
namespace vram {

inline constexpr std::uint32_t kM = 2654435761u;
inline constexpr std::size_t kMaxBytes = 512ull * 1024 * 1024;
inline constexpr std::uint64_t kDefaultMem = 2ull * 1024 * 1024 * 1024;
inline constexpr std::uint32_t kBlock = 256;

/**
 * @brief Buffer size for the streaming benchmark.
 *
 * S = min(512 MiB, memory/4), floored to a multiple of 16 bytes (uint4 stride).
 *
 * @param d
 * @return
 */
inline std::size_t buffer_bytes(const gpgpu::Device& d) {
  std::uint64_t mem = d.memory().value_or(kDefaultMem);
  std::uint64_t s = mem / 4;
  if (s > kMaxBytes) s = kMaxBytes;
  s &= ~static_cast<std::uint64_t>(15);
  if (s < 16) s = 16;
  return static_cast<std::size_t>(s);
}

inline std::uint32_t pattern(std::uint32_t i) { return i * kM; }

/**
 * @brief Launched threads for the grid-stride kernel.
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

/**
 * @brief Closed-form expected checksum.
 *
 * (M * N*(N-1)/2) mod 2^32, N = bytes/4.
 *
 * @param bytes
 * @return
 */
inline std::uint32_t expected_sum(std::size_t bytes) {
  const std::uint64_t N = static_cast<std::uint64_t>(bytes) / 4;
  const std::uint64_t tri = (N % 2 == 0) ? (N / 2) * (N - 1) : N * ((N - 1) / 2);
  return static_cast<std::uint32_t>((tri & 0xFFFFFFFFull) * static_cast<std::uint64_t>(kM));
}

}  // namespace vram
}  // namespace bench
