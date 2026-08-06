#pragma once

/**
 * @file cuda_match.hpp
 * @brief CUDA device matching. Included ONLY from a CUDA runner TU (.cu) that carries
 *        the CUDA Toolkit include paths. Extracted from the example CUDA runner's
 *        find_matching_device().
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>

#include <gpgpu/setup.hpp>

namespace bench {

// Return the CUDA device ordinal whose PCI BDF matches gpgpu::Device::id()
// (formatted "pci-%04x:%02x:%02x.0" by the CUDA backend), or -1 if none.
inline int find_cuda_device(const gpgpu::Device& target) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return -1;
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        char bdf[32];
        std::snprintf(bdf, sizeof(bdf), "pci-%04x:%02x:%02x.0",
                      prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        if (target.id() == bdf) return i;
    }
    return -1;
}

// Seconds between two recorded CUDA events (0 on failure).
inline std::chrono::duration<double> cuda_event_span(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, a, b) != cudaSuccess || ms < 0.0f) return {};
    return std::chrono::duration<double>{ms / 1000.0};
}

} // namespace bench
