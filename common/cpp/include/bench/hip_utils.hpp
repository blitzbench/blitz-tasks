#pragma once

/**
 * @file hip_utils.hpp
 * @brief HIP event timing. Included ONLY from a HIP runner TU (.hip) that carries the
 *        ROCm include paths. The device ordinal of a gpgpu::Setup comes from
 *        gpgpu::vendor::find_hip_device.
 */

#include <hip/hip_runtime.h>

#include <chrono>

namespace bench {

// Seconds between two recorded HIP events (0 on failure).
inline std::chrono::duration<double> hip_event_span(hipEvent_t a, hipEvent_t b) {
    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, a, b) != hipSuccess || ms < 0.0f) return {};
    return std::chrono::duration<double>{ms / 1000.0};
}

} // namespace bench
