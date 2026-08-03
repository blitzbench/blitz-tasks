/**
 * @file cuda_runner.cu
 * @brief CUDA device-local VRAM copy runner.
 *
 * Two device buffers; src filled once (untimed) with the pattern. The timed
 * loop issues full-buffer cudaMemcpyAsync(DeviceToDevice) copies bracketed by
 * CUDA events. A copy reads AND writes the buffer, so work = 2 * reps * S.
 */

#include "cuda_runner.hpp"

#include <cuda_runtime.h>

#include "../../kernel_params.hpp"

#include <bench/config.hpp>
#include <bench/cuda_match.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bench {

RunResult run_gpu_vram_copy_cuda(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "device copy (cudaMemcpy D2D)";
    r.score_unit = "GB/s";

    const int dev = find_cuda_device(setup.device);
    if (dev < 0) { r.error = "no CUDA device matched " + setup.device.id(); return r; }
    if (cudaSetDevice(dev) != cudaSuccess) { r.error = "cudaSetDevice failed"; return r; }

    const std::size_t S = vram::buffer_bytes(setup.device);
    const std::uint32_t N = static_cast<std::uint32_t>(S / 4);

    std::uint8_t* d_src = nullptr;
    std::uint8_t* d_dst = nullptr;
    if (cudaMalloc(&d_src, S) != cudaSuccess || cudaMalloc(&d_dst, S) != cudaSuccess) {
        r.error = "cudaMalloc failed";
        if (d_src) cudaFree(d_src);
        if (d_dst) cudaFree(d_dst);
        return r;
    }

    // --- fill source with the pattern (untimed) ---
    std::vector<std::uint32_t> host(N);
    for (std::uint32_t i = 0; i < N; ++i) host[i] = vram::pattern(i);
    cudaMemcpy(d_src, host.data(), S, cudaMemcpyHostToDevice);

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);

    auto time_copies = [&](int reps) -> double {
        cudaEventRecord(a, 0);
        for (int i = 0; i < reps; ++i)
            cudaMemcpyAsync(d_dst, d_src, S, cudaMemcpyDeviceToDevice, 0);
        cudaEventRecord(b, 0);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        return ms > 0.0f ? ms / 1000.0 : 0.0;
    };

    // --- warmup + calibrate ---
    for (int w = 0; w < kWarmups; ++w) time_copies(1);
    const double t_once = time_copies(1);
    const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

    // --- timed run (device time + host wall) ---
    const auto wall0 = std::chrono::steady_clock::now();
    const double secs = time_copies(reps);
    const auto wall1 = std::chrono::steady_clock::now();

    const cudaError_t err = cudaGetLastError();

    // --- verify: first + last 1 MiB window of dst ---
    bool ok = (err == cudaSuccess);
    std::string verr;
    if (ok) {
        const std::uint32_t w = std::min<std::uint32_t>(vram::kWindowU32, N);
        std::vector<std::uint32_t> chk(w);
        auto check_window = [&](std::uint32_t start) {
            cudaMemcpy(chk.data(), d_dst + static_cast<std::size_t>(start) * 4,
                       static_cast<std::size_t>(w) * 4, cudaMemcpyDeviceToHost);
            for (std::uint32_t i = 0; i < w && ok; ++i) {
                const std::uint32_t idx = start + i;
                if (chk[i] != vram::pattern(idx)) {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "copy mismatch at u32[%u]: got=%u expected=%u",
                                  idx, chk[i], vram::pattern(idx));
                    verr = buf; ok = false;
                }
            }
        };
        check_window(0);
        if (ok && N > w) check_window(N - w);
    } else {
        verr = std::string("CUDA error: ") + cudaGetErrorString(err);
    }

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(d_src); cudaFree(d_dst);

    if (!ok && !verr.empty() && err != cudaSuccess) { r.error = verr; return r; }

    r.work                = 2ull * static_cast<std::uint64_t>(reps) * S;
    r.measured            = std::chrono::duration<double>{secs};
    r.timings.copy_h2d    = r.measured;                       // device D2D copy time
    r.timings.copy_h2d_size = static_cast<std::size_t>(reps) * S;
    r.timings.total       = wall1 - wall0;
    r.score               = score_giga(r.work, secs);
    r.correct             = ok;
    if (!ok) r.error = verr;
    return r;
}

} // namespace bench
