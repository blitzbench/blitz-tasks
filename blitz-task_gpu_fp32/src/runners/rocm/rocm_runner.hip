/**
 * @file rocm_runner.hip
 * @brief ROCm/HIP fp32 dense-FMA runner — SDK-required variant.
 *
 * Dependent fmaf chains in registers; sum stored to global memory. Timed with
 * HIP events over a calibrated rep count. Mirrors the CUDA runner.
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
__global__ void fp32_fma_kernel(float* out, unsigned n, unsigned iters) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float acc0 = 1.0f + static_cast<float>((t + 0u) & 1023u) * (1.0f / 2048.0f);
    float acc1 = 1.0f + static_cast<float>((t + 1u) & 1023u) * (1.0f / 2048.0f);
    float acc2 = 1.0f + static_cast<float>((t + 2u) & 1023u) * (1.0f / 2048.0f);
    float acc3 = 1.0f + static_cast<float>((t + 3u) & 1023u) * (1.0f / 2048.0f);
    const float m = 0.9999f, add = 0.0001f;
    for (unsigned i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc0 = fmaf(acc0, m, add);
            acc1 = fmaf(acc1, m, add);
            acc2 = fmaf(acc2, m, add);
            acc3 = fmaf(acc3, m, add);
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
double time_launches(float* d_out, unsigned n, unsigned iters, int reps) {
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    const unsigned grid = (n + fp32::kBlock - 1) / fp32::kBlock;
    hipEventRecord(a, 0);
    for (int r = 0; r < reps; ++r)
        hipLaunchKernelGGL(fp32_fma_kernel, dim3(grid), dim3(fp32::kBlock), 0, 0, d_out, n, iters);
    hipEventRecord(b, 0);
    hipEventSynchronize(b);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    return ms > 0.0f ? ms / 1000.0 : 0.0;
}

} // namespace

RunResult run_gpu_fp32_rocm(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(fp32 fma)";
    r.score_unit = "GFLOPS";

    const int dev = find_hip_device(setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    const std::uint32_t T = fp32::thread_count(setup.device);
    float* d_out = nullptr;
    if (hipMalloc(&d_out, static_cast<std::size_t>(T) * sizeof(float)) != hipSuccess) {
        r.error = "hipMalloc failed"; return r;
    }
    std::vector<float> host(T);

    // --- pre-flight exactness ---
    time_launches(d_out, fp32::kPreflightThreads, fp32::kPreflightIters, 1);
    hipMemcpy(host.data(), d_out, fp32::kPreflightThreads * sizeof(float), hipMemcpyDeviceToHost);
    for (std::uint32_t t = 0; t < fp32::kPreflightThreads; ++t) {
        const float e = fp32::reference(t, fp32::kPreflightIters);
        if (!fp32::matches(host[t], e)) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t], e);
            r.error = buf; hipFree(d_out); return r;
        }
    }

    // --- warmup + calibrate ---
    for (int w = 0; w < kWarmups; ++w) time_launches(d_out, T, fp32::kIters, 1);
    const double t_once = time_launches(d_out, T, fp32::kIters, 1);
    const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);

    // --- timed run ---
    const double secs = time_launches(d_out, T, fp32::kIters, reps);

    hipMemcpy(host.data(), d_out, static_cast<std::size_t>(T) * sizeof(float), hipMemcpyDeviceToHost);
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
        const float e = fp32::reference(t, fp32::kIters);
        if (!fp32::matches(host[t], e)) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "sample mismatch at t=%u: got=%g expected=%g", t, host[t], e);
            r.error = buf; ok = false; break;
        }
    }

    r.work     = fp32::flops(T, fp32::kIters, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score    = score_giga(r.work, secs);
    r.correct  = ok;
    return r;
}

} // namespace bench
