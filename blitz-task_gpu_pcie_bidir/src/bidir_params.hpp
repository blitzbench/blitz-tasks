#pragma once

/**
 * @file bidir_params.hpp
 * @brief Shared sizing / pattern / verification helpers for the bidirectional transfer
 *        benchmark. Header-only, no vendor dependency; included from every runner TU
 *        via "../../bidir_params.hpp" (mirrors how gpu_fp32 shares kernel_params.hpp).
 *
 * The benchmark transfers a large buffer to the device AND back at the same
 * time. We keep SEPARATE buffer pairs per direction: an upload pair
 * (host src -> device dst) and a download pair (device src -> host dst). Each
 * buffer is S bytes where S = min(256 MiB, memory/8). The u32 payload follows
 * pattern[i] = i * 2654435761u so a sampled read-back verifies the copy.
 */

#include <cstddef>
#include <cstdint>

#include <gpgpu/setup.hpp>

namespace bench {
namespace bidir {

inline constexpr std::uint64_t kMiB        = 1024ull * 1024ull;
inline constexpr std::uint64_t kMaxBytes   = 256ull * kMiB;           // 256 MiB cap
inline constexpr std::uint64_t kDefaultMem = 2ull * 1024ull * kMiB;   // 2 GiB fallback
inline constexpr std::uint32_t kMul        = 2654435761u;             // Knuth multiplicative

// Per-direction buffer size in bytes: min(256 MiB, memory/8), 16-byte aligned
// and a multiple of 4 (u32 payload). Never smaller than 4 KiB.
inline std::size_t buffer_bytes(const gpgpu::Device& d) {
    std::uint64_t mem = d.memory().value_or(kDefaultMem);
    std::uint64_t s   = mem / 8;
    if (s > kMaxBytes) s = kMaxBytes;
    s &= ~std::uint64_t(15);            // 16-byte align (keeps it a multiple of 4 too)
    if (s < 4096) s = 4096;
    return static_cast<std::size_t>(s);
}

// Number of u32 elements in a buffer of `bytes` bytes.
inline std::size_t elem_count(std::size_t bytes) { return bytes / 4; }

// The canonical payload value for element i.
inline std::uint32_t pattern(std::size_t i) {
    return static_cast<std::uint32_t>(i) * kMul;
}

// Fill `n` u32 elements of `buf` with the pattern.
inline void fill_pattern(std::uint32_t* buf, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) buf[i] = pattern(i);
}

// Sample-check up to `samples` evenly spaced elements against the pattern.
inline bool verify_sample(const std::uint32_t* buf, std::size_t n,
                          std::size_t samples = 64) {
    if (n == 0) return false;
    if (samples > n) samples = n;
    for (std::size_t s = 0; s < samples; ++s) {
        const std::size_t i =
            (samples <= 1) ? 0 : (s * (n - 1) / (samples - 1));
        if (buf[i] != pattern(i)) return false;
    }
    return true;
}

} // namespace bidir
} // namespace bench
