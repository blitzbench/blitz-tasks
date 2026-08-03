/**
 * @file cuda_runner.cu
 * @brief CUDA concurrent bidirectional transfer runner.
 *
 * Two non-blocking streams run at the same time: stream A performs R host->device
 * copies (pinned host src -> device dst), stream B performs R device->host copies
 * (device src -> pinned host dst). Because device events cannot meaningfully span
 * two streams, the score is aggregate wall throughput: a single host clock window
 * brackets first-enqueue -> cudaDeviceSynchronize. Per-direction CUDA event spans
 * are recorded for the timings block only. path = "2 streams".
 */

#include "cuda_runner.hpp"

#include <cuda_runtime.h>

#include "../../bidir_params.hpp"

#include <bench/config.hpp>
#include <bench/cuda_match.hpp>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace bench {

RunResult run_gpu_bidir_cuda(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "2 streams";
    r.score_unit = "GB/s";

    const int dev = find_cuda_device(setup.device);
    if (dev < 0) { r.error = "no CUDA device matched " + setup.device.id(); return r; }
    if (cudaSetDevice(dev) != cudaSuccess) { r.error = "cudaSetDevice failed"; return r; }

    const std::size_t S = bidir::buffer_bytes(setup.device);
    const std::size_t N = bidir::elem_count(S);

    std::uint32_t* h_up = nullptr;   // pinned upload source (pattern)
    std::uint32_t* h_dn = nullptr;   // pinned download destination
    void*          d_up = nullptr;   // device upload destination
    void*          d_dn = nullptr;   // device download source (pattern)

    auto fail = [&](const char* what) -> RunResult {
        if (h_up) cudaFreeHost(h_up);
        if (h_dn) cudaFreeHost(h_dn);
        if (d_up) cudaFree(d_up);
        if (d_dn) cudaFree(d_dn);
        r.error = what;
        return r;
    };

    if (cudaHostAlloc(reinterpret_cast<void**>(&h_up), S, cudaHostAllocDefault) != cudaSuccess)
        return fail("cudaHostAlloc (upload src) failed");
    if (cudaHostAlloc(reinterpret_cast<void**>(&h_dn), S, cudaHostAllocDefault) != cudaSuccess)
        return fail("cudaHostAlloc (download dst) failed");
    if (cudaMalloc(&d_up, S) != cudaSuccess) return fail("cudaMalloc (upload dst) failed");
    if (cudaMalloc(&d_dn, S) != cudaSuccess) return fail("cudaMalloc (download src) failed");

    bidir::fill_pattern(h_up, N);
    // Seed the device download source with the pattern (outside any timed window).
    if (cudaMemcpy(d_dn, h_up, S, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("cudaMemcpy (seed download src) failed");

    cudaStream_t sA = nullptr, sB = nullptr;
    if (cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking) != cudaSuccess)
        return fail("cudaStreamCreateWithFlags failed");

    cudaEvent_t h2d0 = nullptr, h2d1 = nullptr, d2h0 = nullptr, d2h1 = nullptr;
    cudaEventCreate(&h2d0); cudaEventCreate(&h2d1);
    cudaEventCreate(&d2h0); cudaEventCreate(&d2h1);

    auto run_window = [&](int reps, double& h2d_secs, double& d2h_secs) -> double {
        const auto t0 = std::chrono::steady_clock::now();
        cudaEventRecord(h2d0, sA);
        for (int i = 0; i < reps; ++i)
            cudaMemcpyAsync(d_up, h_up, S, cudaMemcpyHostToDevice, sA);
        cudaEventRecord(h2d1, sA);
        cudaEventRecord(d2h0, sB);
        for (int i = 0; i < reps; ++i)
            cudaMemcpyAsync(h_dn, d_dn, S, cudaMemcpyDeviceToHost, sB);
        cudaEventRecord(d2h1, sB);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        float ms = 0.0f;
        h2d_secs = (cudaEventElapsedTime(&ms, h2d0, h2d1) == cudaSuccess && ms > 0.f) ? ms / 1000.0 : 0.0;
        d2h_secs = (cudaEventElapsedTime(&ms, d2h0, d2h1) == cudaSuccess && ms > 0.f) ? ms / 1000.0 : 0.0;
        return std::chrono::duration<double>(t1 - t0).count();
    };

    double dummy_h = 0.0, dummy_d = 0.0;
    for (int w = 0; w < kWarmups; ++w) run_window(1, dummy_h, dummy_d);
    const double t_once = run_window(1, dummy_h, dummy_d);
    const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

    double h2d_secs = 0.0, d2h_secs = 0.0;
    const double wall = run_window(reps, h2d_secs, d2h_secs);

    const cudaError_t err = cudaGetLastError();

    // --- verify both directions once, outside the timed window ---
    bool ok = false;
    if (err == cudaSuccess) {
        std::vector<std::uint32_t> back(N);
        cudaMemcpy(back.data(), d_up, S, cudaMemcpyDeviceToHost);   // upload landed?
        const bool up_ok = bidir::verify_sample(back.data(), N);
        const bool dn_ok = bidir::verify_sample(h_dn, N);           // download landed?
        ok = up_ok && dn_ok;
        if (!ok) r.error = up_ok ? "download verification failed" : "upload verification failed";
    } else {
        r.error = std::string("CUDA error: ") + cudaGetErrorString(err);
    }

    cudaEventDestroy(h2d0); cudaEventDestroy(h2d1);
    cudaEventDestroy(d2h0); cudaEventDestroy(d2h1);
    cudaStreamDestroy(sA); cudaStreamDestroy(sB);
    cudaFreeHost(h_up); cudaFreeHost(h_dn);
    cudaFree(d_up); cudaFree(d_dn);

    const std::uint64_t work = 2ull * static_cast<std::uint64_t>(reps) * S;  // up + down
    r.work     = work;
    r.measured = std::chrono::duration<double>{wall};
    r.score    = score_giga(work, wall);
    r.correct  = ok;
    r.timings.copy_h2d      = std::chrono::duration<double>{h2d_secs};
    r.timings.copy_h2d_size = static_cast<std::size_t>(reps) * S;
    r.timings.copy_d2h      = std::chrono::duration<double>{d2h_secs};
    r.timings.copy_d2h_size = static_cast<std::size_t>(reps) * S;
    r.timings.total         = r.measured;
    return r;
}

} // namespace bench
