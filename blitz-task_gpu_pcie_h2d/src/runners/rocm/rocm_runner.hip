/**
 * @file rocm_runner.hip
 * @brief ROCm/HIP host-to-device transfer runner — SDK-required variant.
 *
 * Pinned host source (hipHostMalloc) → device destination (hipMalloc) via
 * hipMemcpyAsync on a stream, timed with HIP events over a calibrated rep
 * count. Structure mirrors the CUDA runner exactly.
 */

#include "rocm_runner.hpp"

#include <hip/hip_runtime.h>

#include <bench/config.hpp>
#include <bench/hip_match.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace bench {

namespace {

constexpr std::uint64_t kMiB  = 1024ull * 1024ull;
constexpr std::uint64_t k2GiB = 2ull * 1024ull * 1024ull * 1024ull;

std::size_t transfer_bytes(const gpgpu::Device& d) {
    std::uint64_t mem = d.memory().value_or(k2GiB);
    std::uint64_t s   = std::min<std::uint64_t>(256ull * kMiB, mem / 8ull);
    s &= ~std::uint64_t(3);
    if (s < 4) s = 4;
    return static_cast<std::size_t>(s);
}

inline std::uint32_t pattern_at(std::size_t i) {
    return static_cast<std::uint32_t>(i) * 2654435761u;
}

bool sample_ok(const std::uint32_t* host, std::size_t count) {
    constexpr std::size_t kSamples = 64;
    if (count == 0) return false;
    for (std::size_t s = 0; s < kSamples; ++s) {
        std::size_t i = (count == 1) ? 0 : (s * (count - 1) / (kSamples - 1));
        if (host[i] != pattern_at(i)) return false;
    }
    return true;
}

} // namespace

RunResult run_gpu_h2d_rocm(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "pinned h->d (hipMemcpyAsync)";
    r.score_unit = "GB/s";

    const int dev = find_hip_device(setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    const std::size_t S     = transfer_bytes(setup.device);
    const std::size_t count = S / sizeof(std::uint32_t);

    void* h_src = nullptr;
    if (hipHostMalloc(&h_src, S, hipHostMallocDefault) != hipSuccess || !h_src) {
        r.error = "hipHostMalloc failed"; return r;
    }
    {
        std::uint32_t* p = static_cast<std::uint32_t*>(h_src);
        for (std::size_t i = 0; i < count; ++i) p[i] = pattern_at(i);
    }

    void* d_dst = nullptr;
    if (hipMalloc(&d_dst, S) != hipSuccess) {
        hipHostFree(h_src); r.error = "hipMalloc failed"; return r;
    }

    hipStream_t stream = nullptr;
    hipStreamCreate(&stream);
    hipEvent_t a = nullptr, b = nullptr;
    hipEventCreate(&a); hipEventCreate(&b);

    auto time_copies = [&](int reps) -> double {
        hipEventRecord(a, stream);
        for (int i = 0; i < reps; ++i)
            hipMemcpyAsync(d_dst, h_src, S, hipMemcpyHostToDevice, stream);
        hipEventRecord(b, stream);
        hipEventSynchronize(b);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, a, b);
        return ms > 0.0f ? ms / 1000.0 : 0.0;
    };

    for (int w = 0; w < kWarmups; ++w) time_copies(1);
    const double t_once = time_copies(1);
    const int    reps   = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

    const auto wall0 = std::chrono::steady_clock::now();
    double secs = time_copies(reps);
    const auto wall1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(wall1 - wall0).count();
    if (secs <= 0.0) secs = wall;

    // --- verify once ---
    std::vector<std::uint32_t> host(count);
    hipMemcpy(host.data(), d_dst, S, hipMemcpyDeviceToHost);
    const hipError_t err = hipGetLastError();
    const bool ok = (err == hipSuccess) && sample_ok(host.data(), count);

    hipEventDestroy(a); hipEventDestroy(b);
    hipStreamDestroy(stream);
    hipFree(d_dst); hipHostFree(h_src);

    if (err != hipSuccess) {
        r.error = std::string("HIP error: ") + hipGetErrorString(err); return r;
    }

    r.work                  = static_cast<std::uint64_t>(reps) * S;
    r.measured              = std::chrono::duration<double>{secs};
    r.timings.copy_h2d      = r.measured;
    r.timings.copy_h2d_size = S;
    r.timings.total         = std::chrono::duration<double>{wall};
    r.score                 = score_giga(r.work, secs);
    r.correct               = ok;
    if (!ok) r.error = "verification failed";
    return r;
}

} // namespace bench
