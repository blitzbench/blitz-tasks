#pragma once

/**
 * @file params.hpp
 * @brief Problem sizing, operand generation and CPU reference shared by the GEMM tasks.
 *
 * The GEMM tasks multiply large square matrices through the vendor BLAS where one
 * exists and through a tiled kernel elsewhere, so unlike the throughput
 * microbenchmarks nothing here can assume a particular kernel shape. What the tasks do
 * share is:
 *
 *   * a size ladder, from which each device takes the largest rung that fits both its
 *     memory and the time budget;
 *   * operands defined by a closed-form hash rather than uploaded, so a 16384-square
 *     problem costs no host memory and no transfer;
 *   * a sampled reference, since a full CPU multiply at these sizes is not viable and
 *     the vendor kernel is not ours to predict.
 *
 * Every backend kernel MUST reproduce operand_value() exactly, in both the fill kernel
 * and any tiled multiply, or the reference below will not match.
 */

#include <cmath>
#include <cstdint>

#include <gpgpu/device.hpp>

namespace bench {
namespace gemm {

// Square problem sizes (M = N = K), smallest first. The bottom rung exists for software
// and low-memory devices; the top one for accelerators that only reach peak on a
// multiply large enough to amortise their cache hierarchy.
inline constexpr std::uint32_t kLadder[] = {1024u, 2048u, 4096u, 8192u, 16384u};

// Share of device memory the operands may occupy.
inline constexpr double kMemoryFraction = 0.60;

// A single multiply predicted to run longer than this steps the ladder down one rung.
inline constexpr double kMaxSingleGemmSeconds = 1.0;

// Scratch a vendor BLAS may request on top of the operands.
inline constexpr std::uint64_t kWorkspaceBytes = 32ull << 20;

// Rows of C read back for verification, and elements sampled from them.
inline constexpr std::uint32_t kVerifyRows = 4u;
inline constexpr std::uint32_t kVerifySamples = 64u;

// Element type of A and B. C and the accumulator are always fp32.
enum class Precision { Fp16, Fp32 };

/**
 * @brief Per-invocation knobs a task hands to its runners.
 *
 * Carries what blitz::DataConfig configures plus the probe's need for a cheap run:
 * probing every setup at the top of the ladder would allocate and fill gigabytes per
 * candidate before the host has even chosen one.
 */
struct RunParams {
    std::uint64_t cap_bytes{0};  // Upper bound on operand bytes; 0 leaves it to the device.
    std::uint32_t seed{1};       // Feeds operand_value(); reruns with the same seed match.
    int pinned_reps{0};          // Timed reps; 0 calibrates against kGemmTargetSeconds.
};

/**
 * @brief Bytes A, B and C occupy at size @p n.
 *
 * @param p
 * @param n
 * @return
 */
inline std::uint64_t operand_bytes(Precision p, std::uint32_t n) {
    const std::uint64_t elems = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n);
    const std::uint64_t ab = (p == Precision::Fp16) ? 2ull : 4ull;
    return elems * (2ull * ab + 4ull);
}

/**
 * @brief Integer hash mixing a matrix element's coordinates with the run seed.
 *
 * @param row
 * @param col
 * @param which 0 for A, 1 for B.
 * @param seed
 * @return
 */
inline std::uint32_t operand_hash(std::uint32_t row, std::uint32_t col, std::uint32_t which, std::uint32_t seed) {
    std::uint32_t h = row * 0x9E3779B1u ^ col * 0x85EBCA77u ^ which * 0xC2B2AE3Du ^ seed;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}

/**
 * @brief Value of an operand element: one of eight multiples of 1/4 in [-1, 0.75].
 *
 * Those eight values are exact in fp16, TF32 and fp32 alike, so every product in the
 * multiply is exact whichever precision the device runs. The only error the reference
 * has to tolerate is the order in which the fp32 accumulator sums them.
 *
 * @param row
 * @param col
 * @param which 0 for A, 1 for B.
 * @param seed
 * @return
 */
inline float operand_value(std::uint32_t row, std::uint32_t col, std::uint32_t which, std::uint32_t seed) {
    return (static_cast<float>(operand_hash(row, col, which, seed) & 7u) - 4.0f) * 0.25f;
}

/**
 * @brief Reference value of C[row][col] for C = A * B, accumulated in double.
 *
 * @param row
 * @param col
 * @param n
 * @param seed
 * @return
 */
inline double reference_element(std::uint32_t row, std::uint32_t col, std::uint32_t n, std::uint32_t seed) {
    double acc = 0.0;
    for (std::uint32_t k = 0; k < n; ++k) {
        acc += static_cast<double>(operand_value(row, k, 0u, seed)) *
               static_cast<double>(operand_value(k, col, 1u, seed));
    }
    return acc;
}

/**
 * @brief Whether a device element agrees with its reference.
 *
 * The absolute term scales with sqrt(n) because the sum is a random walk over n terms
 * of magnitude <= 1, so both the result and its accumulated rounding grow that way.
 *
 * @param got
 * @param expected
 * @param n
 * @return
 */
inline bool element_ok(double got, double expected, std::uint32_t n) {
    const double walk = std::sqrt(static_cast<double>(n));
    return std::fabs(got - expected) <= 1e-4 * (std::fabs(expected) + walk);
}

/**
 * @brief Flops in @p reps multiplies at size @p n.
 *
 * @param n
 * @param reps
 * @return
 */
inline std::uint64_t gemm_flops(std::uint32_t n, int reps) {
    const std::uint64_t nn = static_cast<std::uint64_t>(n);
    return 2ull * nn * nn * nn * static_cast<std::uint64_t>(reps);
}

/**
 * @brief Largest ladder rung whose operands and workspace fit in @p budget_bytes.
 *
 * @param p
 * @param budget_bytes
 * @return The smallest rung when nothing fits; allocation then fails loudly rather than
 *         the task silently reporting no result.
 */
inline std::uint32_t largest_fitting(Precision p, std::uint64_t budget_bytes) {
    std::uint32_t best = kLadder[0];
    for (const std::uint32_t n : kLadder) {
        if (operand_bytes(p, n) + kWorkspaceBytes <= budget_bytes) best = n;
    }
    return best;
}

/**
 * @brief Problem size for @p device, optionally capped by a host-supplied byte budget.
 *
 * @param p
 * @param device
 * @param cap_bytes 0 when the host did not cap it.
 * @return
 */
inline std::uint32_t choose_size(Precision p, const gpgpu::Device& device, std::uint64_t cap_bytes) {
    const std::uint64_t memory = device.memory().value_or(2ull << 30);
    std::uint64_t budget = static_cast<std::uint64_t>(static_cast<double>(memory) * kMemoryFraction);
    if (cap_bytes > 0 && cap_bytes < budget) budget = cap_bytes;
    return largest_fitting(p, budget);
}

/**
 * @brief Largest rung whose multiply is predicted to stay inside the time budget.
 *
 * Cost grows with the cube of the edge length, so one timed multiply at the bottom of
 * the ladder extrapolates every rung above it. Runners size themselves this way rather
 * than timing the memory-derived size directly, because on a slow device that first
 * multiply is exactly the one that must not be attempted. Efficiency improves with size,
 * so extrapolating from a small multiply overestimates the time and errs towards the
 * smaller rung.
 *
 * @param measured_n Edge length that was actually timed.
 * @param measured_seconds Device time for one multiply at @p measured_n.
 * @param memory_limit_n Largest rung the device's memory allows.
 * @return
 */
inline std::uint32_t largest_within_time(std::uint32_t measured_n, double measured_seconds,
                                         std::uint32_t memory_limit_n) {
    if (measured_seconds <= 0.0) return memory_limit_n;
    std::uint32_t best = kLadder[0];
    for (const std::uint32_t n : kLadder) {
        if (n > memory_limit_n) break;
        const double scale = static_cast<double>(n) / static_cast<double>(measured_n);
        if (measured_seconds * scale * scale * scale <= kMaxSingleGemmSeconds) best = n;
    }
    return best;
}

/**
 * @brief Next rung below @p n, or 0 when @p n is already the smallest.
 *
 * @param n
 * @return
 */
inline std::uint32_t step_down(std::uint32_t n) {
    std::uint32_t below = 0;
    for (const std::uint32_t candidate : kLadder) {
        if (candidate >= n) break;
        below = candidate;
    }
    return below;
}

/**
 * @brief Row index of the @p s-th verification sample.
 *
 * Samples spread over the kVerifyRows rows that get read back, and across the full
 * column range, so a tile-indexing error anywhere in the row is caught.
 *
 * @param s
 * @return
 */
inline std::uint32_t sample_row(std::uint32_t s) { return s % kVerifyRows; }

/**
 * @brief Byte cap that pins the ladder to its smallest rung, for cheap probing.
 *
 * @param p
 * @return
 */
inline std::uint64_t smallest_rung_cap(Precision p) { return operand_bytes(p, kLadder[0]) + kWorkspaceBytes; }

/**
 * @brief Column index of the @p s-th verification sample at size @p n.
 *
 * @param s
 * @param n
 * @return
 */
inline std::uint32_t sample_col(std::uint32_t s, std::uint32_t n) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(s) * (n - 1u) / (kVerifySamples - 1u));
}

} // namespace gemm
} // namespace bench
