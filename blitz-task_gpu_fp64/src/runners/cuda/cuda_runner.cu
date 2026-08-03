/**
 * @file cuda_runner.cu
 * @brief CUDA fp64 dense-FMA runner — SDK-required variant.
 *
 * Dependent __fma_rn (double) chains kept entirely in registers; the sum is
 * written to global memory to defeat dead-code elimination and enable
 * verification. Timed with CUDA events over a calibrated rep count.
 *
 * fp64 always compiles under CUDA (all NVIDIA archs execute double fma, though
 * consumer parts run it at a low rate — that is a legitimate, honest score, not
 * an unsupported case). So there is no gating here.
 */

#include "cuda_runner.hpp"

#include <cuda_runtime.h>

#include "../../kernel_params.hpp"

#include <bench/config.hpp>
#include <bench/cuda_match.hpp>

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
            acc0 = __fma_rn(acc0, m, add);
            acc1 = __fma_rn(acc1, m, add);
            acc2 = __fma_rn(acc2, m, add);
            acc3 = __fma_rn(acc3, m, add);
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
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const unsigned grid = (n + fp64::kBlock - 1) / fp64::kBlock;
    cudaEventRecord(a, 0);
    for (int r = 0; r < reps; ++r)
        fp64_fma_kernel<<<grid, fp64::kBlock>>>(d_out, n, iters);
    cudaEventRecord(b, 0);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms > 0.0f ? ms / 1000.0 : 0.0;
}

} // namespace

RunResult run_gpu_fp64_cuda(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(fp64 fma)";
    r.score_unit = "GFLOPS";

    const int dev = find_cuda_device(setup.device);
    if (dev < 0) { r.error = "no CUDA device matched " + setup.device.id(); return r; }
    if (cudaSetDevice(dev) != cudaSuccess) { r.error = "cudaSetDevice failed"; return r; }

    const std::uint32_t T = fp64::thread_count(setup.device);
    double* d_out = nullptr;
    if (cudaMalloc(&d_out, static_cast<std::size_t>(T) * sizeof(double)) != cudaSuccess) {
        r.error = "cudaMalloc failed"; return r;
    }
    std::vector<double> host(T);

    // --- pre-flight exactness: tiny problem, compare every output ---
    time_launches(d_out, fp64::kPreflightThreads, fp64::kPreflightIters, 1);
    cudaMemcpy(host.data(), d_out, fp64::kPreflightThreads * sizeof(double),
               cudaMemcpyDeviceToHost);
    for (std::uint32_t t = 0; t < fp64::kPreflightThreads; ++t) {
        const double e = fp64::reference(t, fp64::kPreflightIters);
        if (!fp64::matches(host[t], e)) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t], e);
            r.error = buf; cudaFree(d_out); return r;
        }
    }

    // --- warmup + calibrate ---
    for (int w = 0; w < kWarmups; ++w) time_launches(d_out, T, fp64::kIters, 1);
    const double t_once = time_launches(d_out, T, fp64::kIters, 1);
    const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);

    // --- timed run ---
    const double secs = time_launches(d_out, T, fp64::kIters, reps);

    cudaMemcpy(host.data(), d_out, static_cast<std::size_t>(T) * sizeof(double),
               cudaMemcpyDeviceToHost);
    const cudaError_t err = cudaGetLastError();
    cudaFree(d_out);
    if (err != cudaSuccess) {
        r.error = std::string("CUDA error: ") + cudaGetErrorString(err); return r;
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
