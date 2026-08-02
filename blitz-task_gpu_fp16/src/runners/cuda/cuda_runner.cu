/**
 * @file cuda_runner.cu
 * @brief CUDA fp16 throughput runner — tensor-first with a packed-SIMD fallback.
 *
 * Runtime selection by compute capability:
 *   * sm_70+  -> nvcuda::wmma 16x16x16 f16->f32   path="tensor(wmma 16x16x16 f16->f32)"
 *   * sm_53+  -> __hfma2 half2 dependent chains    path="simd(half2 hfma2)"
 *   * else    -> supported=false
 *
 * The tensor kernel accumulates all-1/16 tiles so every C element equals
 * iters*16*(1/16)^2 exactly; the half2 kernel runs the shared contracting chain.
 * Both are host-launchable regardless of the arch actually built (bodies are
 * guarded by __CUDA_ARCH__), and timed with CUDA events over calibrated reps.
 */

#include "cuda_runner.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

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
 * @brief Tensor path: 16x16x16 f16->f32 wmma
 * @param out
 * @param warps
 * @param iters
 */
__global__ void fp16_wmma_kernel(float* out, unsigned warps, unsigned iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    const unsigned gt   = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned warp = gt / 32u;
    if (warp >= warps) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(a, __float2half(fp16::kTensorVal));
    wmma::fill_fragment(b, __float2half(fp16::kTensorVal));
    wmma::fill_fragment(c, 0.0f);

    for (unsigned i = 0; i < iters; ++i)
        wmma::mma_sync(c, a, b, c);

    // Every element is identical; store this lane's element 0.
    out[gt] = c.num_elements > 0 ? c.x[0] : 0.0f;
#else
    (void)out; (void)warps; (void)iters;
#endif
}

/**
 * @brief packed SIMD fallback: __hfma2 half2 chains.
 * @param out
 * @param n
 * @param iters
 * @param m
 * @param a
 */
__global__ void fp16_half2_kernel(float* out, unsigned n, unsigned iters, float m, float a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const __half2 mh = __float2half2_rn(m);
    const __half2 ah = __float2half2_rn(a);

    __half2 acc[fp16::kChains];
    #pragma unroll
    for (int k = 0; k < fp16::kChains; ++k) {
        const float s0 = 1.0f + float((t + 7u * k + 0u) & 255u) * (1.0f / 256.0f);
        const float s1 = 1.0f + float((t + 7u * k + 13u) & 255u) * (1.0f / 256.0f);
        acc[k] = __halves2half2(__float2half(s0), __float2half(s1));
    }
    for (unsigned i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < fp16::kUnroll; ++u) {
            #pragma unroll
            for (int k = 0; k < fp16::kChains; ++k)
                acc[k] = __hfma2(acc[k], mh, ah);
        }
    }
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < fp16::kChains; ++k)
        s += __low2float(acc[k]) + __high2float(acc[k]);
    out[t] = s;
#else
    (void)out; (void)n; (void)iters; (void)m; (void)a;
#endif
}

double time_wmma(float* d_out, unsigned warps, unsigned iters, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const unsigned threads = warps * 32u;
    const unsigned grid = (threads + fp16::kBlock - 1) / fp16::kBlock;
    cudaEventRecord(a, 0);
    for (int r = 0; r < reps; ++r)
        fp16_wmma_kernel<<<grid, fp16::kBlock>>>(d_out, warps, iters);
    cudaEventRecord(b, 0);
    cudaEventSynchronize(b);
    float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms > 0.0f ? ms / 1000.0 : 0.0;
}

double time_half2(float* d_out, unsigned n, unsigned iters, float m, float a, int reps) {
    cudaEvent_t ea, eb;
    cudaEventCreate(&ea); cudaEventCreate(&eb);
    const unsigned grid = (n + fp16::kBlock - 1) / fp16::kBlock;
    cudaEventRecord(ea, 0);
    for (int r = 0; r < reps; ++r)
        fp16_half2_kernel<<<grid, fp16::kBlock>>>(d_out, n, iters, m, a);
    cudaEventRecord(eb, 0);
    cudaEventSynchronize(eb);
    float ms = 0.0f; cudaEventElapsedTime(&ms, ea, eb);
    cudaEventDestroy(ea); cudaEventDestroy(eb);
    return ms > 0.0f ? ms / 1000.0 : 0.0;
}

} // namespace

RunResult run_gpu_fp16_cuda(const gpgpu::Setup& setup) {
    RunResult r;
    r.score_unit = "GFLOPS";
    r.path       = "unsupported(fp16)";

    const int dev = find_cuda_device(setup.device);
    if (dev < 0) { r.error = "no CUDA device matched " + setup.device.id(); return r; }
    if (cudaSetDevice(dev) != cudaSuccess) { r.error = "cudaSetDevice failed"; return r; }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        r.error = "cudaGetDeviceProperties failed"; return r;
    }
    const int cc = prop.major * 10 + prop.minor;
    const bool use_wmma = prop.major >= 7;
    if (!use_wmma && cc < 53) {
        r.supported = false;
        r.path = "unsupported(fp16 needs sm_53+)";
        return r;
    }
    r.path = use_wmma ? "tensor(wmma 16x16x16 f16->f32)" : "simd(half2 hfma2)";

    const std::uint32_t T = fp16::thread_count(setup.device);
    float* d_out = nullptr;
    if (cudaMalloc(&d_out, static_cast<std::size_t>(T) * sizeof(float)) != cudaSuccess) {
        r.error = "cudaMalloc failed"; return r;
    }
    std::vector<float> host(T);

    auto fail = [&](const std::string& msg) {
        r.error = msg; cudaFree(d_out); return r;
    };

    if (use_wmma) {
        // --- pre-flight exactness ---
        const unsigned pf_warps = fp16::kPreflightThreads / 32u;
        time_wmma(d_out, pf_warps, fp16::kTensorPreflightIters, 1);
        cudaMemcpy(host.data(), d_out, pf_warps * 32u * sizeof(float), cudaMemcpyDeviceToHost);
        for (std::uint32_t t = 0; t < pf_warps * 32u; ++t) {
            if (!fp16::tensor_ok(host[t], fp16::kTensorPreflightIters, 16)) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "pre-flight mismatch at t=%u: got=%g expected=%g",
                              t, host[t], fp16::tensor_out(fp16::kTensorPreflightIters, 16));
                return fail(buf);
            }
        }

        const unsigned warps = T / 32u;
        for (int w = 0; w < kWarmups; ++w) time_wmma(d_out, warps, fp16::kTensorIters, 1);
        const double t_once = time_wmma(d_out, warps, fp16::kTensorIters, 1);
        const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
        const double secs = time_wmma(d_out, warps, fp16::kTensorIters, reps);

        cudaMemcpy(host.data(), d_out, static_cast<std::size_t>(T) * sizeof(float),
                   cudaMemcpyDeviceToHost);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return fail(std::string("CUDA error: ") + cudaGetErrorString(err));

        bool ok = true;
        for (std::uint32_t s = 0; s < 64; ++s) {
            const std::uint32_t t = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(s) * (T - 1) / 63);
            if (!fp16::tensor_ok(host[t], fp16::kTensorIters, 16)) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "sample mismatch at t=%u: got=%g expected=%g",
                              t, host[t], fp16::tensor_out(fp16::kTensorIters, 16));
                r.error = buf; ok = false; break;
            }
        }
        r.work     = fp16::tensor_flops(warps, fp16::kTensorIters, reps, 16, 16, 16);
        r.measured = std::chrono::duration<double>{secs};
        r.timings.kernel_compute = r.measured;
        r.score    = score_giga(r.work, secs);
        r.correct  = ok;
    } else {
        // --- packed half2 fallback ---
        time_half2(d_out, fp16::kPreflightThreads, fp16::kPreflightIters,
                   fp16::kPreM, fp16::kPreA, 1);
        cudaMemcpy(host.data(), d_out, fp16::kPreflightThreads * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (std::uint32_t t = 0; t < fp16::kPreflightThreads; ++t) {
            const double e = fp16::simd_preflight_out(t, 2, fp16::kPreflightIters);
            if (!fp16::matches_exact(host[t], e)) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "pre-flight mismatch at t=%u: got=%g expected=%g",
                              t, host[t], e);
                return fail(buf);
            }
        }

        for (int w = 0; w < kWarmups; ++w)
            time_half2(d_out, T, fp16::kIters, fp16::kMainM, fp16::kMainA, 1);
        const double t_once = time_half2(d_out, T, fp16::kIters, fp16::kMainM, fp16::kMainA, 1);
        const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
        const double secs = time_half2(d_out, T, fp16::kIters, fp16::kMainM, fp16::kMainA, reps);

        cudaMemcpy(host.data(), d_out, static_cast<std::size_t>(T) * sizeof(float),
                   cudaMemcpyDeviceToHost);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return fail(std::string("CUDA error: ") + cudaGetErrorString(err));

        bool ok = true;
        for (std::uint32_t s = 0; s < 64; ++s) {
            const std::uint32_t t = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(s) * (T - 1) / 63);
            if (!fp16::simd_main_ok(host[t], 2)) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "sample out of envelope at t=%u: got=%g", t, host[t]);
                r.error = buf; ok = false; break;
            }
        }
        r.work     = fp16::simd_flops(T, fp16::kIters, reps, 2);
        r.measured = std::chrono::duration<double>{secs};
        r.timings.kernel_compute = r.measured;
        r.score    = score_giga(r.work, secs);
        r.correct  = ok;
    }

    cudaFree(d_out);
    return r;
}

} // namespace bench
