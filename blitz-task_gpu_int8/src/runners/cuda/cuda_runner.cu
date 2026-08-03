/**
 * @file cuda_runner.cu
 * @brief CUDA int8 throughput runner — SDK-required variant.
 *
 * Tensor/dot-first with a runtime fallback; the chosen path is recorded:
 *   sm >= 7.2 -> nvcuda::wmma 16x16x16 s8->s32   path "tensor(wmma s8->s32)"
 *   sm >= 6.1 -> __dp4a (signed 4x int8 dot)     path "dp4a"
 *   else      -> char4 manual 4-lane MAC         path "simd(char4)"
 *
 * The packed kernel emits __dp4a where the compiled arch supports it (>=610)
 * and the manual char4 MAC otherwise; both compute the identical signed
 * dot4+accumulate math as bench::i8::reference_packed, so one CPU reference
 * verifies both. Unit GOPS (TOPS = GOPS/1000). See kernel_params.hpp for the
 * exact math + ops accounting.
 */

#include "cuda_runner.hpp"

#include <cuda_runtime.h>
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
 * @brief Packed path: dp4a / char4 per-thread dependent MAC chains.
 *
 * @param out
 * @param n
 * @param iters
 */
__global__ void int8_packed_kernel(unsigned* out, unsigned n, unsigned iters) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const unsigned W = 0x04030201u;
    unsigned acc[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k)
        acc[k] = 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * ((unsigned)k + 1u);

    for (unsigned i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
#if __CUDA_ARCH__ >= 610
                acc[k] = (unsigned)__dp4a((int)acc[k], (int)W, (int)acc[k]);
#else
                int a = (int)acc[k];
                int dot = (int)(signed char)(a & 0xff)         * (int)(signed char)(W & 0xff)         +
                          (int)(signed char)((a >> 8)  & 0xff) * (int)(signed char)((W >> 8)  & 0xff) +
                          (int)(signed char)((a >> 16) & 0xff) * (int)(signed char)((W >> 16) & 0xff) +
                          (int)(signed char)((a >> 24) & 0xff) * (int)(signed char)((W >> 24) & 0xff);
                acc[k] = acc[k] + (unsigned)dot;
#endif
            }
        }
    }
    out[t] = acc[0] + acc[1] + acc[2] + acc[3];
}

/**
 * @brief Tensor path: wmma 16x16x16 s8->s32.
 *
 * Each warp accumulates C += A*B over `iters` MMAs with all-ones s8 A,B; every
 * C element ends at 16*iters. Every thread stores that value (all lanes agree).
 *
 * @param out
 * @param n
 * @param iters
 */
__global__ void int8_wmma_kernel(unsigned* out, unsigned n, unsigned iters) {
#if __CUDA_ARCH__ >= 720
    using namespace nvcuda;
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
    wmma::fill_fragment(a, (signed char)1);
    wmma::fill_fragment(b, (signed char)1);
    wmma::fill_fragment(c, 0);
    for (unsigned i = 0; i < iters; ++i)
        wmma::mma_sync(c, a, b, c);
    if (t < n) out[t] = (unsigned)c.x[0];
#else
    (void)out; (void)n; (void)iters;
#endif
}

enum class Path { Wmma, Dp4a, Char4 };

/**
 * @brief Device seconds for `reps` launches of `n` threads / `iters` iterations
 *
 * @param path
 * @param d_out
 * @param n
 * @param iters
 * @param reps
 * @return
 */
double time_launches(Path path, unsigned* d_out, unsigned n, unsigned iters, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const unsigned grid = (n + i8::kBlock - 1) / i8::kBlock;
    cudaEventRecord(a, 0);
    for (int r = 0; r < reps; ++r) {
        if (path == Path::Wmma) int8_wmma_kernel<<<grid, i8::kBlock>>>(d_out, n, iters);
        else                    int8_packed_kernel<<<grid, i8::kBlock>>>(d_out, n, iters);
    }
    cudaEventRecord(b, 0);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms > 0.0f ? ms / 1000.0 : 0.0;
}

} // namespace

RunResult run_gpu_int8_cuda(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(char4)";
    r.score_unit = "GOPS";

    const int dev = find_cuda_device(setup.device);
    if (dev < 0) { r.error = "no CUDA device matched " + setup.device.id(); return r; }
    if (cudaSetDevice(dev) != cudaSuccess) { r.error = "cudaSetDevice failed"; return r; }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    const int sm = prop.major * 10 + prop.minor;

    Path path = Path::Char4;
    if (sm >= 72)      { path = Path::Wmma;  r.path = "tensor(wmma s8->s32)"; }
    else if (sm >= 61) { path = Path::Dp4a;  r.path = "dp4a"; }
    else               { path = Path::Char4; r.path = "simd(char4)"; }

    const bool tensor = (path == Path::Wmma);

    const std::uint32_t T = i8::thread_count(setup.device);
    unsigned* d_out = nullptr;
    if (cudaMalloc(&d_out, static_cast<std::size_t>(T) * sizeof(unsigned)) != cudaSuccess) {
        r.error = "cudaMalloc failed"; return r;
    }
    std::vector<unsigned> host(T);

    auto verify = [&](std::uint32_t count, std::uint32_t iters, std::uint32_t stride) -> bool {
        for (std::uint32_t s = 0; s < count; ++s) {
            const std::uint32_t t = s * stride;
            const unsigned got = host[t];
            const unsigned exp = tensor ? (unsigned)i8::reference_tensor(iters)
                                        : i8::reference_packed(t, iters);
            if (got != exp) {
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                              "mismatch at t=%u: got=%u expected=%u", t, got, exp);
                r.error = buf; return false;
            }
        }
        return true;
    };

    // --- pre-flight exactness: tiny problem, compare every output ---
    time_launches(path, d_out, i8::kPreflightThreads, i8::kPreflightIters, 1);
    cudaMemcpy(host.data(), d_out, i8::kPreflightThreads * sizeof(unsigned),
               cudaMemcpyDeviceToHost);
    if (!verify(i8::kPreflightThreads, i8::kPreflightIters, 1)) { cudaFree(d_out); return r; }

    // --- warmup + calibrate ---
    for (int w = 0; w < kWarmups; ++w) time_launches(path, d_out, T, i8::kIters, 1);
    const double t_once = time_launches(path, d_out, T, i8::kIters, 1);
    const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);

    // --- timed run (device time backs the score; host time is kernel_launch) ---
    const auto h0  = std::chrono::steady_clock::now();
    const double secs = time_launches(path, d_out, T, i8::kIters, reps);
    const auto h1  = std::chrono::steady_clock::now();

    cudaMemcpy(host.data(), d_out, static_cast<std::size_t>(T) * sizeof(unsigned),
               cudaMemcpyDeviceToHost);
    const cudaError_t err = cudaGetLastError();
    cudaFree(d_out);
    if (err != cudaSuccess) {
        r.error = std::string("CUDA error: ") + cudaGetErrorString(err); return r;
    }

    // --- post-run sampled verification ---
    constexpr std::uint32_t kSamples = 64;
    const std::uint32_t stride = (T - 1) / (kSamples - 1);
    const bool ok = verify(kSamples, i8::kIters, stride);

    const std::uint32_t tiles = T / 32u;
    r.work     = tensor ? i8::ops_tensor(tiles, i8::kIters, reps)
                        : i8::ops_packed(T, i8::kIters, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.timings.kernel_launch  = std::chrono::duration<double>{h1 - h0};
    r.score    = score_giga(r.work, secs);
    r.correct  = ok;
    return r;
}

} // namespace bench
