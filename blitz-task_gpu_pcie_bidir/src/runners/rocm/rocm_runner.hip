/**
 * @file rocm_runner.hip
 * @brief ROCm/HIP concurrent bidirectional transfer runner. Mirrors the CUDA runner.
 *
 * Two non-blocking streams run at the same time: stream A performs R host->device
 * copies (pinned host src -> device dst), stream B performs R device->host copies.
 * Score is aggregate wall throughput over a single host clock window bracketing
 * first-enqueue -> hipDeviceSynchronize. Per-direction HIP event spans feed the
 * timings block only. path = "2 streams". Untestable locally (no ROCm) — CI.
 */

#include "rocm_runner.hpp"

#include <hip/hip_runtime.h>

#include "../../bidir_params.hpp"

#include <bench/config.hpp>
#include <bench/hip_match.hpp>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace bench {

RunResult run_gpu_bidir_rocm(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "2 streams";
    r.score_unit = "GB/s";

    const int dev = find_hip_device(setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    const std::size_t S = bidir::buffer_bytes(setup.device);
    const std::size_t N = bidir::elem_count(S);

    std::uint32_t* h_up = nullptr;   // pinned upload source (pattern)
    std::uint32_t* h_dn = nullptr;   // pinned download destination
    void*          d_up = nullptr;   // device upload destination
    void*          d_dn = nullptr;   // device download source (pattern)

    auto fail = [&](const char* what) -> RunResult {
        if (h_up) hipHostFree(h_up);
        if (h_dn) hipHostFree(h_dn);
        if (d_up) hipFree(d_up);
        if (d_dn) hipFree(d_dn);
        r.error = what;
        return r;
    };

    if (hipHostMalloc(reinterpret_cast<void**>(&h_up), S, hipHostMallocDefault) != hipSuccess)
        return fail("hipHostMalloc (upload src) failed");
    if (hipHostMalloc(reinterpret_cast<void**>(&h_dn), S, hipHostMallocDefault) != hipSuccess)
        return fail("hipHostMalloc (download dst) failed");
    if (hipMalloc(&d_up, S) != hipSuccess) return fail("hipMalloc (upload dst) failed");
    if (hipMalloc(&d_dn, S) != hipSuccess) return fail("hipMalloc (download src) failed");

    bidir::fill_pattern(h_up, N);
    if (hipMemcpy(d_dn, h_up, S, hipMemcpyHostToDevice) != hipSuccess)
        return fail("hipMemcpy (seed download src) failed");

    hipStream_t sA = nullptr, sB = nullptr;
    if (hipStreamCreateWithFlags(&sA, hipStreamNonBlocking) != hipSuccess ||
        hipStreamCreateWithFlags(&sB, hipStreamNonBlocking) != hipSuccess)
        return fail("hipStreamCreateWithFlags failed");

    hipEvent_t h2d0 = nullptr, h2d1 = nullptr, d2h0 = nullptr, d2h1 = nullptr;
    hipEventCreate(&h2d0); hipEventCreate(&h2d1);
    hipEventCreate(&d2h0); hipEventCreate(&d2h1);

    auto run_window = [&](int reps, double& h2d_secs, double& d2h_secs) -> double {
        const auto t0 = std::chrono::steady_clock::now();
        hipEventRecord(h2d0, sA);
        for (int i = 0; i < reps; ++i)
            hipMemcpyAsync(d_up, h_up, S, hipMemcpyHostToDevice, sA);
        hipEventRecord(h2d1, sA);
        hipEventRecord(d2h0, sB);
        for (int i = 0; i < reps; ++i)
            hipMemcpyAsync(h_dn, d_dn, S, hipMemcpyDeviceToHost, sB);
        hipEventRecord(d2h1, sB);
        hipDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        float ms = 0.0f;
        h2d_secs = (hipEventElapsedTime(&ms, h2d0, h2d1) == hipSuccess && ms > 0.f) ? ms / 1000.0 : 0.0;
        d2h_secs = (hipEventElapsedTime(&ms, d2h0, d2h1) == hipSuccess && ms > 0.f) ? ms / 1000.0 : 0.0;
        return std::chrono::duration<double>(t1 - t0).count();
    };

    double dummy_h = 0.0, dummy_d = 0.0;
    for (int w = 0; w < kWarmups; ++w) run_window(1, dummy_h, dummy_d);
    const double t_once = run_window(1, dummy_h, dummy_d);
    const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

    double h2d_secs = 0.0, d2h_secs = 0.0;
    const double wall = run_window(reps, h2d_secs, d2h_secs);

    const hipError_t err = hipGetLastError();

    bool ok = false;
    if (err == hipSuccess) {
        std::vector<std::uint32_t> back(N);
        hipMemcpy(back.data(), d_up, S, hipMemcpyDeviceToHost);
        const bool up_ok = bidir::verify_sample(back.data(), N);
        const bool dn_ok = bidir::verify_sample(h_dn, N);
        ok = up_ok && dn_ok;
        if (!ok) r.error = up_ok ? "download verification failed" : "upload verification failed";
    } else {
        r.error = std::string("HIP error: ") + hipGetErrorString(err);
    }

    hipEventDestroy(h2d0); hipEventDestroy(h2d1);
    hipEventDestroy(d2h0); hipEventDestroy(d2h1);
    hipStreamDestroy(sA); hipStreamDestroy(sB);
    hipHostFree(h_up); hipHostFree(h_dn);
    hipFree(d_up); hipFree(d_dn);

    const std::uint64_t work = 2ull * static_cast<std::uint64_t>(reps) * S;
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
