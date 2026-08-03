#pragma once

/**
 * @file kernel_params.hpp
 * @brief Shared sizing / pattern / verification for the gpu_vram_copy microbenchmark.
 *
 * Two device-local buffers; the source is filled once (untimed) with the
 * pattern u32[i] = i * 2654435761u. The timed loop performs full-buffer
 * device->device copies. A copy both READS and WRITES the whole buffer, so the
 * reported work is 2 * reps * S bytes (column "VRAM GB/s (r+w)").
 */

#include <cstddef>
#include <cstdint>
#include <gpgpu/setup.hpp>

namespace bench {
namespace vram {

inline constexpr std::uint32_t kM = 2654435761u;                         // Knuth multiplicative hash
inline constexpr std::size_t kMaxBytes = 512ull * 1024 * 1024;           // 512 MiB cap
inline constexpr std::uint64_t kDefaultMem = 2ull * 1024 * 1024 * 1024;  // 2 GiB fallback
inline constexpr std::size_t kWindowU32 = 256u * 1024u;                  // 1 MiB verification window

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

/**
 * @brief Pattern value for the u32 element at index i.
 *
 * @param i
 * @return
 */
inline std::uint32_t pattern(std::uint32_t i) { return i * kM; }

}  // namespace vram
}  // namespace bench
