#pragma once

/**
 * @file cuda_utils.hpp
 * @brief CUDA event timing. Included ONLY from a CUDA runner TU (.cu) that carries
 *        the CUDA Toolkit include paths. The device ordinal of a gpgpu::Setup comes
 *        from gpgpu::vendor::find_cuda_device.
 */

#include <cuda_runtime.h>

#include <chrono>

namespace bench {

// Seconds between two recorded CUDA events (0 on failure).
inline std::chrono::duration<double> cuda_event_span(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, a, b) != cudaSuccess || ms < 0.0f) return {};
    return std::chrono::duration<double>{ms / 1000.0};
}

} // namespace bench
