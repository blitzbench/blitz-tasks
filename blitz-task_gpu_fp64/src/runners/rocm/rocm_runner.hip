/**
 * @file rocm_runner.hip
 * @brief ROCm/HIP fp64 dense-FMA runner — SDK-required variant.
 *
 * Dependent fma (double) chains in registers; sum stored to global memory.
 * Timed with HIP events over a calibrated rep count. Mirrors the CUDA runner.
 *
 * fp64 always compiles under HIP (every AMD GPU executes double fma, at a
 * per-arch rate) — no gating, the score is honest even where the rate is low.
 */

#include "rocm_runner.hpp"

#include <hip/hip_runtime.h>

#include "../../kernel_params.hpp"

#include <bench/config.hpp>
#include <bench/hip_match.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bench {

namespace {

/**
 * @brief Dense dependent-FMA chains kept in registers.
 *
 * @param out
 * @param n
 * @param iters
 */
__global__ void fp64_fma_kernel(double* out, unsigned n, unsigned iters) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    double acc0 = 1.0 + static_cast<double>((t + 0u) & 1023u) * (1.0 / 2048.0);
    double acc1 = 1.0 + static_cast<double>((t + 1u) & 1023u) * (1.0 / 2048.0);
    double acc2 = 1.0 + static_cast<double>((t + 2u) & 1023u) * (1.0 / 2048.0);
    double acc3 = 1.0 + static_cast<double>((t + 3u) & 1023u) * (1.0 / 2048.0);
    const double m = 0.9999, add = 0.0001;
    for (unsigned i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc0 = fma(acc0, m, add);
            acc1 = fma(acc1, m, add);
            acc2 = fma(acc2, m, add);
            acc3 = fma(acc3, m, add);
        }
    }
    out[t] = acc0 + acc1 + acc2 + acc3;
}

/**
 * @brief Device seconds for `reps` launches of `n` threads / `iters` iterations
 *
 * @param d_out
 * @param n
 * @param iters
 * @param reps
 * @return
 */
double time_launches(double* d_out, unsigned n, unsigned iters, int reps) {
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    const unsigned grid = (n + fp64::kBlock - 1) / fp64::kBlock;
    hipEventRecord(a, 0);
    for (int r = 0; r < reps; ++r)
        hipLaunchKernelGGL(fp64_fma_kernel, dim3(grid), dim3(fp64::kBlock), 0, 0, d_out, n, iters);
    hipEventRecord(b, 0);
    hipEventSynchronize(b);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    return ms > 0.0f ? ms / 1000.0 : 0.0;
}

} // namespace

RunResult run_gpu_fp64_rocm(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(fp64 fma)";
    r.score_unit = "GFLOPS";

    const int dev = find_hip_device(setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    const std::uint32_t T = fp64::thread_count(setup.device);
    double* d_out = nullptr;
    if (hipMalloc(&d_out, static_cast<std::size_t>(T) * sizeof(double)) != hipSuccess) {
        r.error = "hipMalloc failed"; return r;
    }
    std::vector<double> host(T);

    // --- pre-flight exactness ---
    time_launches(d_out, fp64::kPreflightThreads, fp64::kPreflightIters, 1);
    hipMemcpy(host.data(), d_out, fp64::kPreflightThreads * sizeof(double), hipMemcpyDeviceToHost);
    for (std::uint32_t t = 0; t < fp64::kPreflightThreads; ++t) {
        const double e = fp64::reference(t, fp64::kPreflightIters);
        if (!fp64::matches(host[t], e)) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t], e);
            r.error = buf; hipFree(d_out); return r;
        }
    }

    // --- warmup + calibrate ---
    for (int w = 0; w < kWarmups; ++w) time_launches(d_out, T, fp64::kIters, 1);
    const double t_once = time_launches(d_out, T, fp64::kIters, 1);
    const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);

    // --- timed run ---
    const double secs = time_launches(d_out, T, fp64::kIters, reps);

    hipMemcpy(host.data(), d_out, static_cast<std::size_t>(T) * sizeof(double), hipMemcpyDeviceToHost);
    const hipError_t err = hipGetLastError();
    hipFree(d_out);
    if (err != hipSuccess) {
        r.error = std::string("HIP error: ") + hipGetErrorString(err); return r;
    }

    // --- post-run sampled verification ---
    bool ok = true;
    constexpr std::uint32_t kSamples = 64;
    for (std::uint32_t s = 0; s < kSamples; ++s) {
        const std::uint32_t t = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(s) * (T - 1) / (kSamples - 1));
        const double e = fp64::reference(t, fp64::kIters);
        if (!fp64::matches(host[t], e)) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "sample mismatch at t=%u: got=%g expected=%g", t, host[t], e);
            r.error = buf; ok = false; break;
        }
    }

    r.work     = fp64::flops(T, fp64::kIters, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score    = score_giga(r.work, secs);
    r.correct  = ok;
    return r;
}

} // namespace bench
