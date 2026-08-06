#pragma once

#include <cmath>
#include <cstdint>

/**
 * @file config.hpp
 * @brief Single source of truth for benchmark sizing / calibration. Every app pulls
 *        its targets, warmup counts and scoring conversions from here so the whole
 *        suite stays consistent.
 */
namespace bench {

// --- calibration targets (device-side seconds) ---
inline constexpr double kComputeTargetSeconds  = 0.15;  // fp16/fp32/fp64/int8 kernels
inline constexpr double kTransferTargetSeconds = 0.50;  // h2d / d2h / bidir / vram

// --- reps ---
inline constexpr int kWarmups     = 2;   // uncounted warmup launches
inline constexpr int kMinTimedReps = 3;  // floor on timed reps

// Sensible per-family caps on the calibrated rep count (keeps a very slow
// software device — e.g. lavapipe — from running for minutes).
inline constexpr int kComputeRepCap  = 1000;
inline constexpr int kTransferRepCap = 50;

// GEMM tasks time one large matrix multiply per rep, which is far coarser than the
// microbenchmark kernels above: a single multiply at the top of the size ladder can
// run for a second on its own, so kComputeTargetSeconds would round every device
// down to kMinTimedReps.
inline constexpr double kGemmTargetSeconds = 0.50;
inline constexpr int    kGemmRepCap        = 100;

// --- unit conversions ---
inline constexpr double kGB   = 1.0e9;  // 1 GB      = 1e9 bytes
inline constexpr double kGiga = 1.0e9;  // 1 GFLOP/GOP = 1e9 ops

// Run one warm rep, then pick R = clamp(ceil(target / t_once), kMinTimedReps, cap).
// t_once is the device time of a single rep in seconds.
inline int calibrate_repeats(double t_once, double target, int cap) {
    if (t_once <= 0.0) return cap;
    int r = static_cast<int>(std::ceil(target / t_once));
    if (r < kMinTimedReps) r = kMinTimedReps;
    if (r > cap) r = cap;
    return r;
}

// Giga-metric from a raw work count and a device time in seconds:
//   GFLOPS / GOPS / GB-per-s = work / seconds / 1e9.
inline double score_giga(std::uint64_t work, double seconds) {
    if (seconds <= 0.0) return 0.0;
    return static_cast<double>(work) / seconds / kGiga;
}

} // namespace bench
