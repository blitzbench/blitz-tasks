/**
 * @file rocm_runner.hip
 * @brief ROCm/HIP fp16 throughput runner — raw matrix-core builtins + packed fallback.
 *
 * No rocWMMA dependency: the matrix path uses the __builtin_amdgcn_mfma_* (CDNA)
 * and __builtin_amdgcn_wmma_* (RDNA3/4) intrinsics directly, each compiled only
 * for the gfx targets that own it (per-__gfx*__ guards). At runtime we dispatch
 * by hipDeviceProp_t.gcnArchName. A packed __hfma2 half2 chain is compiled for
 * every arch as the universal fallback.
 *
 *   gfx908                 -> mfma_f32_32x32x8f16   (K=8)   "tensor(mfma 32x32x8 f16)"
 *   gfx90a / gfx94x        -> mfma_f32_16x16x16f16  (K=16)  "tensor(mfma 16x16x16 f16)"
 *   gfx11xx / gfx12xx w32  -> wmma_f32_16x16x16_f16 (K=16)  "tensor(wmma rdna3 16x16x16 f16)"
 *   any                    -> __hfma2 half2 chains          "simd(half2)"
 *
 * Every A/B element is the exact half 1/16 and the accumulator starts at 0, so
 * after `iters` MMAs every C element equals iters*K*(1/16)^2 exactly (uniform
 * matrix product — independent of the per-lane fragment layout). CANNOT be built
 * or run on the local box (partial/broken ROCm); needs AMD CI with GPU_TARGETS
 * covering the device arch.
 */

#include "rocm_runner.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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

using half4v  = _Float16 __attribute__((ext_vector_type(4)));
using half16v = _Float16 __attribute__((ext_vector_type(16)));
using float4v = float    __attribute__((ext_vector_type(4)));
using float8v = float    __attribute__((ext_vector_type(8)));
using float16v = float   __attribute__((ext_vector_type(16)));

constexpr _Float16 kTV = static_cast<_Float16>(0.0625f);  // 1/16

/**
 * @brief CDNA gfx908: 32x32x8 f16 (K=8), wave64
 */
__global__ void fp16_mfma_32x32x8_kernel(float* out, unsigned warps, unsigned iters) {
#if defined(__gfx908__)
    const unsigned gt = blockIdx.x * blockDim.x + threadIdx.x;
    if (gt / 64u >= warps) return;
    half4v a, b;
    for (int i = 0; i < 4; ++i) { a[i] = kTV; b[i] = kTV; }
    float16v c = {0};
    for (unsigned i = 0; i < iters; ++i)
        c = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
    out[gt] = c[0];
#else
    (void)out; (void)warps; (void)iters;
#endif
}

/**
 * @brief CDNA gfx90a / gfx94x: 16x16x16 f16 (K=16), wave64
 */
__global__ void fp16_mfma_16x16x16_kernel(float* out, unsigned warps, unsigned iters) {
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    const unsigned gt = blockIdx.x * blockDim.x + threadIdx.x;
    if (gt / 64u >= warps) return;
    half4v a, b;
    for (int i = 0; i < 4; ++i) { a[i] = kTV; b[i] = kTV; }
    float4v c = {0};
    for (unsigned i = 0; i < iters; ++i)
        c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    out[gt] = c[0];
#else
    (void)out; (void)warps; (void)iters;
#endif
}

/**
 * @brief RDNA3/4 gfx11xx/12xx: WMMA 16x16x16 f16 (K=16), wave32
 */
__global__ void fp16_wmma_w32_kernel(float* out, unsigned warps, unsigned iters) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned gt = blockIdx.x * blockDim.x + threadIdx.x;
    if (gt / 32u >= warps) return;
    half16v a, b;
    for (int i = 0; i < 16; ++i) { a[i] = kTV; b[i] = kTV; }
    float8v c = {0};
    for (unsigned i = 0; i < iters; ++i)
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    out[gt] = c[0];
#else
    (void)out; (void)warps; (void)iters;
#endif
}

/**
 * @brief universal packed fallback: __hfma2 half2 chains
 */
__global__ void fp16_half2_kernel(float* out, unsigned n, unsigned iters, float mf, float af) {
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const __half2 mh = __float2half2_rn(mf);
    const __half2 ah = __float2half2_rn(af);
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
}

enum class Path { Mfma32x32x8, Mfma16x16x16, WmmaW32, Half2 };

double time_tensor(Path p, float* d_out, unsigned warps, unsigned wave, unsigned iters, int reps) {
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    const unsigned threads = warps * wave;
    const unsigned grid = (threads + fp16::kBlock - 1) / fp16::kBlock;
    hipEventRecord(a, 0);
    for (int r = 0; r < reps; ++r) {
        switch (p) {
            case Path::Mfma32x32x8:
                hipLaunchKernelGGL(fp16_mfma_32x32x8_kernel, dim3(grid), dim3(fp16::kBlock), 0, 0,
                                   d_out, warps, iters); break;
            case Path::Mfma16x16x16:
                hipLaunchKernelGGL(fp16_mfma_16x16x16_kernel, dim3(grid), dim3(fp16::kBlock), 0, 0,
                                   d_out, warps, iters); break;
            case Path::WmmaW32:
                hipLaunchKernelGGL(fp16_wmma_w32_kernel, dim3(grid), dim3(fp16::kBlock), 0, 0,
                                   d_out, warps, iters); break;
            default: break;
        }
    }
    hipEventRecord(b, 0);
    hipEventSynchronize(b);
    float ms = 0.0f; hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    return ms > 0.0f ? ms / 1000.0 : 0.0;
}

double time_half2(float* d_out, unsigned n, unsigned iters, float m, float a, int reps) {
    hipEvent_t ea, eb;
    hipEventCreate(&ea); hipEventCreate(&eb);
    const unsigned grid = (n + fp16::kBlock - 1) / fp16::kBlock;
    hipEventRecord(ea, 0);
    for (int r = 0; r < reps; ++r)
        hipLaunchKernelGGL(fp16_half2_kernel, dim3(grid), dim3(fp16::kBlock), 0, 0,
                           d_out, n, iters, m, a);
    hipEventRecord(eb, 0);
    hipEventSynchronize(eb);
    float ms = 0.0f; hipEventElapsedTime(&ms, ea, eb);
    hipEventDestroy(ea); hipEventDestroy(eb);
    return ms > 0.0f ? ms / 1000.0 : 0.0;
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

} // namespace

RunResult run_gpu_fp16_rocm(const gpgpu::Setup& setup) {
    RunResult r;
    r.score_unit = "GFLOPS";
    r.path       = "unsupported(fp16)";

    const int dev = find_hip_device(setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        r.error = "hipGetDeviceProperties failed"; return r;
    }
    std::string arch = prop.gcnArchName;                 // e.g. "gfx90a:sramecc+:xnack-"
    const std::string base = arch.substr(0, arch.find(':'));

    Path path = Path::Half2;
    int  K = 0, M = 0, N = 0;
    unsigned wave = static_cast<unsigned>(prop.warpSize);
    if (base == "gfx908") {
        path = Path::Mfma32x32x8; K = 8;  M = 32; N = 32; wave = 64;
        r.path = "tensor(mfma 32x32x8 f16)";
    } else if (base == "gfx90a" || starts_with(base, "gfx94")) {
        path = Path::Mfma16x16x16; K = 16; M = 16; N = 16; wave = 64;
        r.path = "tensor(mfma 16x16x16 f16)";
    } else if (starts_with(base, "gfx11") || starts_with(base, "gfx12")) {
        path = Path::WmmaW32; K = 16; M = 16; N = 16; wave = 32;
        r.path = "tensor(wmma rdna3 16x16x16 f16)";
    } else {
        path = Path::Half2;
        r.path = "simd(half2)";
    }

    const std::uint32_t T = fp16::thread_count(setup.device);
    float* d_out = nullptr;
    if (hipMalloc(&d_out, static_cast<std::size_t>(T) * sizeof(float)) != hipSuccess) {
        r.error = "hipMalloc failed"; return r;
    }
    std::vector<float> host(T);
    auto fail = [&](const std::string& msg) { r.error = msg; hipFree(d_out); return r; };

    if (path != Path::Half2) {
        const unsigned pf_warps = fp16::kPreflightThreads / wave;
        time_tensor(path, d_out, pf_warps, wave, fp16::kTensorPreflightIters, 1);
        hipMemcpy(host.data(), d_out, pf_warps * wave * sizeof(float), hipMemcpyDeviceToHost);
        if (hipGetLastError() != hipSuccess)
            return fail("tensor kernel launch failed (build GPU_TARGETS must include " + base + ")");
        for (std::uint32_t t = 0; t < pf_warps * wave; ++t) {
            if (!fp16::tensor_ok(host[t], fp16::kTensorPreflightIters, K)) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "pre-flight mismatch at t=%u: got=%g expected=%g",
                              t, host[t], fp16::tensor_out(fp16::kTensorPreflightIters, K));
                return fail(buf);
            }
        }

        const unsigned warps = T / wave;
        for (int w = 0; w < kWarmups; ++w) time_tensor(path, d_out, warps, wave, fp16::kTensorIters, 1);
        const double t_once = time_tensor(path, d_out, warps, wave, fp16::kTensorIters, 1);
        const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
        const double secs = time_tensor(path, d_out, warps, wave, fp16::kTensorIters, reps);

        hipMemcpy(host.data(), d_out, static_cast<std::size_t>(T) * sizeof(float),
                  hipMemcpyDeviceToHost);
        if (hipGetLastError() != hipSuccess) return fail("HIP error during timed run");

        bool ok = true;
        for (std::uint32_t s = 0; s < 64; ++s) {
            const std::uint32_t t = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(s) * (T - 1) / 63);
            if (!fp16::tensor_ok(host[t], fp16::kTensorIters, K)) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "sample mismatch at t=%u: got=%g expected=%g",
                              t, host[t], fp16::tensor_out(fp16::kTensorIters, K));
                r.error = buf; ok = false; break;
            }
        }
        r.work     = fp16::tensor_flops(warps, fp16::kTensorIters, reps, M, N, K);
        r.measured = std::chrono::duration<double>{secs};
        r.timings.kernel_compute = r.measured;
        r.score    = score_giga(r.work, secs);
        r.correct  = ok;
    } else {
        time_half2(d_out, fp16::kPreflightThreads, fp16::kPreflightIters, fp16::kPreM, fp16::kPreA, 1);
        hipMemcpy(host.data(), d_out, fp16::kPreflightThreads * sizeof(float), hipMemcpyDeviceToHost);
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

        hipMemcpy(host.data(), d_out, static_cast<std::size_t>(T) * sizeof(float),
                  hipMemcpyDeviceToHost);
        if (hipGetLastError() != hipSuccess) return fail("HIP error during timed run");

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

    hipFree(d_out);
    return r;
}

} // namespace bench
