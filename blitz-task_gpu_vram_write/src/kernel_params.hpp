#pragma once

/**
 * @file kernel_params.hpp
 * @brief Shared sizing / pattern / verification for the gpu_vram_write microbenchmark.
 *
 * A grid-stride kernel of 128-bit (uint4 / uvec4) stores writes, for every u32
 * element at index i, the value i * 2654435761u (cheap ALU, verifiable, not
 * constant-foldable). The buffer holds S bytes = N u32 = n_vec4 uint4.
 * work = reps * S (writes only).
 */

#include <cstddef>
#include <cstdint>
#include <gpgpu/setup.hpp>

namespace bench {
namespace vram {

inline constexpr std::uint32_t kM = 2654435761u;
inline constexpr std::size_t kMaxBytes = 512ull * 1024 * 1024;
inline constexpr std::uint64_t kDefaultMem = 2ull * 1024 * 1024 * 1024;
inline constexpr std::size_t kWindowU32 = 256u * 1024u;  // 1 MiB window
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

}  // namespace vram
}  // namespace bench
