/**
 * @file cuda_runner.cu
 * @brief CUDA device-to-host transfer runner — SDK-required variant.
 *
 * Device source (cudaMalloc, prefilled once) → pinned host destination
 * (cudaHostAlloc) via cudaMemcpyAsync on a stream, timed with CUDA events over a
 * calibrated rep count. The received host buffer is sample-checked once, outside
 * the timed loop. Exact mirror of the h2d CUDA runner.
 */

#include "cuda_runner.hpp"

#include <cuda_runtime.h>

#include <bench/config.hpp>
#include <bench/cuda_match.hpp>

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

RunResult run_gpu_d2h_cuda(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "device->pinned h (cudaMemcpyAsync)";
    r.score_unit = "GB/s";

    const int dev = find_cuda_device(setup.device);
    if (dev < 0) { r.error = "no CUDA device matched " + setup.device.id(); return r; }
    if (cudaSetDevice(dev) != cudaSuccess) { r.error = "cudaSetDevice failed"; return r; }

    const std::size_t S     = transfer_bytes(setup.device);
    const std::size_t count = S / sizeof(std::uint32_t);

    void* d_src = nullptr;
    if (cudaMalloc(&d_src, S) != cudaSuccess) { r.error = "cudaMalloc failed"; return r; }

    void* h_dst = nullptr;
    if (cudaHostAlloc(&h_dst, S, cudaHostAllocDefault) != cudaSuccess || !h_dst) {
        cudaFree(d_src); r.error = "cudaHostAlloc failed"; return r;
    }

    // Prefill the device source with the pattern (untimed).
    {
        std::vector<std::uint32_t> seed(count);
        for (std::size_t i = 0; i < count; ++i) seed[i] = pattern_at(i);
        cudaMemcpy(d_src, seed.data(), S, cudaMemcpyHostToDevice);
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaEvent_t a = nullptr, b = nullptr;
    cudaEventCreate(&a); cudaEventCreate(&b);

    auto time_copies = [&](int reps) -> double {
        cudaEventRecord(a, stream);
        for (int i = 0; i < reps; ++i)
            cudaMemcpyAsync(h_dst, d_src, S, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
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

    // --- verify once: the received host buffer ---
    const cudaError_t err = cudaGetLastError();
    const bool ok = (err == cudaSuccess) &&
                    sample_ok(static_cast<const std::uint32_t*>(h_dst), count);

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_dst); cudaFree(d_src);

    if (err != cudaSuccess) {
        r.error = std::string("CUDA error: ") + cudaGetErrorString(err); return r;
    }

    r.work                  = static_cast<std::uint64_t>(reps) * S;
    r.measured              = std::chrono::duration<double>{secs};
    r.timings.copy_d2h      = r.measured;
    r.timings.copy_d2h_size = S;
    r.timings.total         = std::chrono::duration<double>{wall};
    r.score                 = score_giga(r.work, secs);
    r.correct               = ok;
    if (!ok) r.error = "verification failed";
    return r;
}

} // namespace bench
