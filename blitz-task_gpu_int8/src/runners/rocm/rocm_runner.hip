/**
 * @file rocm_runner.hip
 * @brief ROCm/HIP int8 throughput runner — SDK-required variant.
 *
 * Raw AMDGCN matrix/dot builtins (NO rocWMMA dependency), selected at runtime by
 * hipDeviceProp_t.gcnArchName; the chosen path is recorded:
 *   CDNA gfx90a/94x/950 -> __builtin_amdgcn_mfma_i32_16x16x16i8   "tensor(mfma s8->s32)"
 *   RDNA3   gfx11xx      -> __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 "tensor(wmma iu8)"
 *   else (dot-capable)   -> __builtin_amdgcn_sdot4                 "dot4(sdot4)"
 *   else                 -> char4 manual 4-lane MAC               "simd(char4)"
 *
 * The tensor kernels feed all-ones s8 A,B into a genuine 16x16x16 MMA, so every
 * C element equals 16*iters (bench::i8::reference_tensor) regardless of the
 * hardware lane layout. The dot/char4 packed kernel matches i8::reference_packed.
 * Unit GOPS. See kernel_params.hpp for math + ops accounting.
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

typedef int int32x4_t __attribute__((ext_vector_type(4)));
typedef int int32x8_t __attribute__((ext_vector_type(8)));

/**
 * @brief Packed path: sdot4 / char4 per-thread dependent MAC chains.
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
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sdot4)
                acc[k] = (unsigned)__builtin_amdgcn_sdot4((int)acc[k], (int)W, (int)acc[k], false);
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
 * @brief Tensor path: 16x16x16 s8->s32 MMA, all-ones inputs => C == 16*iters.
 *
 * @param out
 * @param n
 * @param iters
 */
__global__ void int8_mma_kernel(unsigned* out, unsigned n, unsigned iters) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || \
    defined(__gfx942__) || defined(__gfx950__)
    int32x4_t c = {0, 0, 0, 0};
    const int a = 0x01010101; // four s8 ones
    const int b = 0x01010101;
    for (unsigned i = 0; i < iters; ++i)
        c = __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, c, 0, 0, 0);
    if (t < n) out[t] = (unsigned)c[0];
#elif defined(__gfx11__)
    int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
    const int32x4_t a = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    const int32x4_t b = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    for (unsigned i = 0; i < iters; ++i)
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c, false);
    if (t < n) out[t] = (unsigned)c[0];
#else
    (void)iters;
    if (t < n) out[t] = 0u;
#endif
}

enum class Path { Mma, Packed };

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
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    const unsigned grid = (n + i8::kBlock - 1) / i8::kBlock;
    hipEventRecord(a, 0);
    for (int r = 0; r < reps; ++r) {
        if (path == Path::Mma)
            hipLaunchKernelGGL(int8_mma_kernel, dim3(grid), dim3(i8::kBlock), 0, 0, d_out, n, iters);
        else
            hipLaunchKernelGGL(int8_packed_kernel, dim3(grid), dim3(i8::kBlock), 0, 0, d_out, n, iters);
    }
    hipEventRecord(b, 0);
    hipEventSynchronize(b);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    return ms > 0.0f ? ms / 1000.0 : 0.0;
}

bool contains(const char* hay, const char* needle) {
    return std::string(hay).find(needle) != std::string::npos;
}

} // namespace

RunResult run_gpu_int8_rocm(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(char4)";
    r.score_unit = "GOPS";

    const int dev = find_hip_device(setup.device);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, dev);
    const char* arch = prop.gcnArchName;

    Path path = Path::Packed;
    if (contains(arch, "gfx90a") || contains(arch, "gfx940") || contains(arch, "gfx941") ||
        contains(arch, "gfx942") || contains(arch, "gfx950")) {
        path = Path::Mma; r.path = "tensor(mfma s8->s32)";
    } else if (contains(arch, "gfx11")) {
        path = Path::Mma; r.path = "tensor(wmma iu8)";
    } else if (contains(arch, "gfx9") || contains(arch, "gfx101") || contains(arch, "gfx103")) {
        path = Path::Packed; r.path = "dot4(sdot4)";
    } else {
        path = Path::Packed; r.path = "simd(char4)";
    }
    const bool tensor = (path == Path::Mma);
    const unsigned warp = prop.warpSize > 0 ? (unsigned)prop.warpSize : 64u;

    const std::uint32_t T = i8::thread_count(setup.device);
    unsigned* d_out = nullptr;
    if (hipMalloc(&d_out, static_cast<std::size_t>(T) * sizeof(unsigned)) != hipSuccess) {
        r.error = "hipMalloc failed"; return r;
    }
    std::vector<unsigned> host(T);

    auto verify = [&](std::uint32_t count, std::uint32_t iters, std::uint32_t stride) -> bool {
        for (std::uint32_t s = 0; s < count; ++s) {
            const std::uint32_t t = s * stride;
            const unsigned exp = tensor ? (unsigned)i8::reference_tensor(iters)
                                        : i8::reference_packed(t, iters);
            if (host[t] != exp) {
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                              "mismatch at t=%u: got=%u expected=%u", t, host[t], exp);
                r.error = buf; return false;
            }
        }
        return true;
    };

    // --- pre-flight exactness ---
    time_launches(path, d_out, i8::kPreflightThreads, i8::kPreflightIters, 1);
    hipMemcpy(host.data(), d_out, i8::kPreflightThreads * sizeof(unsigned), hipMemcpyDeviceToHost);
    if (!verify(i8::kPreflightThreads, i8::kPreflightIters, 1)) { hipFree(d_out); return r; }

    // --- warmup + calibrate ---
    for (int w = 0; w < kWarmups; ++w) time_launches(path, d_out, T, i8::kIters, 1);
    const double t_once = time_launches(path, d_out, T, i8::kIters, 1);
    const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);

    // --- timed run ---
    const double secs = time_launches(path, d_out, T, i8::kIters, reps);

    hipMemcpy(host.data(), d_out, static_cast<std::size_t>(T) * sizeof(unsigned), hipMemcpyDeviceToHost);
    const hipError_t err = hipGetLastError();
    hipFree(d_out);
    if (err != hipSuccess) {
        r.error = std::string("HIP error: ") + hipGetErrorString(err); return r;
    }

    // --- post-run sampled verification ---
    constexpr std::uint32_t kSamples = 64;
    const std::uint32_t stride = (T - 1) / (kSamples - 1);
    const bool ok = verify(kSamples, i8::kIters, stride);

    const std::uint32_t tiles = T / warp;
    r.work     = tensor ? i8::ops_tensor(tiles, i8::kIters, reps)
                        : i8::ops_packed(T, i8::kIters, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score    = score_giga(r.work, secs);
    r.correct  = ok;
    return r;
}

} // namespace bench
