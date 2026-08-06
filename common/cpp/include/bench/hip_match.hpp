#pragma once

/**
 * @file hip_match.hpp
 * @brief ROCm/HIP device matching. Included ONLY from a HIP runner TU (.hip) that
 *        carries the ROCm include paths. Extracted from the example ROCm runner's
 *        find_matching_device().
 */

#include <hip/hip_runtime.h>

#include <chrono>
#include <string>

#include <gpgpu/setup.hpp>

namespace bench {

// Return the HIP device ordinal whose PCI bus id matches gpgpu::Device::id()
// (the ROCm backend formats it "pci-" + hipDeviceGetPCIBusId), or -1 if none.
inline int find_hip_device(const gpgpu::Device& target) {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n == 0) return -1;
    for (int i = 0; i < n; ++i) {
        char bdf[32] = {0};
        if (hipDeviceGetPCIBusId(bdf, sizeof(bdf), i) != hipSuccess) continue;
        if (target.id() == std::string("pci-") + bdf) return i;
    }
    return -1;
}

// Seconds between two recorded HIP events (0 on failure).
inline std::chrono::duration<double> hip_event_span(hipEvent_t a, hipEvent_t b) {
    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, a, b) != hipSuccess || ms < 0.0f) return {};
    return std::chrono::duration<double>{ms / 1000.0};
}

} // namespace bench
