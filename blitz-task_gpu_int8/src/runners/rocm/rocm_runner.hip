/**
 * @file rocm_runner.hip
 * @brief ROCm/HIP int8 throughput runner.
 *
 * Raw AMDGCN matrix/dot builtins (no rocWMMA dependency), selected at runtime by
 * hipDeviceProp_t.gcnArchName; the chosen path is recorded:
 *   CDNA1/2 gfx908/gfx90a -> mfma_i32_16x16x16i8          (K=16) "tensor(mfma 16x16x16 s8->s32)"
 *   CDNA3   gfx94x        -> mfma_i32_16x16x32_i8         (K=32) "tensor(mfma 16x16x32 s8->s32)"
 *   CDNA4   gfx950        -> mfma_i32_16x16x64_i8         (K=64) "tensor(mfma 16x16x64 s8->s32)"
 *   RDNA3   gfx11xx       -> wmma_i32_16x16x16_iu8_w32    (K=16) "tensor(wmma rdna3 iu8)"
 *   RDNA4   gfx12xx       -> wmma_i32_16x16x16_iu8 gfx12  (K=16) "tensor(wmma rdna4 iu8)"
 *   else (dot-capable)    -> __builtin_amdgcn_sdot4               "dot4(sdot4)"
 *   else                  -> char4 manual 4-lane MAC              "simd(char4)"
 *
 * The tensor kernels feed all-ones s8 A,B into a genuine 16x16xK MMA, so every
 * C element equals K*iters (bench::i8::reference_tensor) regardless of the
 * hardware lane layout. Each wave splits its MMAs over kMmaChains independent
 * accumulators so the matrix pipe never waits on the previous result. The
 * dot/char4 packed kernel matches i8::reference_packed. Unit GOPS. See
 * kernel_params.hpp for math + ops accounting.
 */

#include "rocm_runner.hpp"

#include <hip/hip_runtime.h>

#include "../../kernel_params.hpp"

#include <bench/config.hpp>
#include <gpgpu/vendor.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

namespace bench {

namespace {

typedef int int32x2_t __attribute__((ext_vector_type(2)));
typedef int int32x4_t __attribute__((ext_vector_type(4)));
typedef int int32x8_t __attribute__((ext_vector_type(8)));

constexpr unsigned kMmaChains = 4;
static_assert(i8::kIters % kMmaChains == 0 && i8::kPreflightIters % kMmaChains == 0,
              "tensor iteration counts must split evenly over the accumulator chains");

constexpr int kOnes = 0x01010101;  // four s8 ones

/**
 * @brief Runs `iters` MMAs split over kMmaChains independent accumulators.
 *
 * @param iters total MMAs of this wave, a multiple of kMmaChains
 * @param mma callable c' = A*B + c for one accumulator
 * @return the mean over all elements of the chain-summed accumulator, which equals
 *         the value every C element would hold after `iters` dependent MMAs
 */
template <typename Acc, typename Mma>
__device__ __forceinline__ unsigned run_mma_chains(unsigned iters, Mma mma) {
    constexpr int kElems = sizeof(Acc) / sizeof(int);
    Acc c[kMmaChains];
    #pragma unroll
    for (unsigned k = 0; k < kMmaChains; ++k) c[k] = Acc{};
    for (unsigned i = 0; i < iters / kMmaChains; ++i) {
        #pragma unroll
        for (unsigned k = 0; k < kMmaChains; ++k) c[k] = mma(c[k]);
    }
    unsigned s = 0;
    #pragma unroll
    for (unsigned k = 0; k < kMmaChains; ++k) {
        #pragma unroll
        for (int j = 0; j < kElems; ++j) s += static_cast<unsigned>(c[k][j]);
    }
    return s / static_cast<unsigned>(kElems);
}

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
 * @brief Tensor path: 16x16xK s8->s32 MMA, all-ones inputs => C == K*iters.
 *
 * @param out
 * @param n
 * @param iters
 */
__global__ __launch_bounds__(i8::kBlock) void int8_mma_kernel(unsigned* out, unsigned n, unsigned iters) {
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v = 0u;
#if defined(__gfx908__) || defined(__gfx90a__)
    v = run_mma_chains<int32x4_t>(iters, [](int32x4_t c) {
        return __builtin_amdgcn_mfma_i32_16x16x16i8(kOnes, kOnes, c, 0, 0, 0);
    });
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    const long ab = (static_cast<long>(kOnes) << 32) | static_cast<unsigned>(kOnes);
    v = run_mma_chains<int32x4_t>(iters, [ab](int32x4_t c) {
        return __builtin_amdgcn_mfma_i32_16x16x32_i8(ab, ab, c, 0, 0, 0);
    });
#elif defined(__gfx950__)
    const int32x4_t ab = {kOnes, kOnes, kOnes, kOnes};
    v = run_mma_chains<int32x4_t>(iters, [ab](int32x4_t c) {
        return __builtin_amdgcn_mfma_i32_16x16x64_i8(ab, ab, c, 0, 0, 0);
    });
#elif defined(__GFX11__)
    const int32x4_t ab = {kOnes, kOnes, kOnes, kOnes};
    v = run_mma_chains<int32x8_t>(iters, [ab](int32x8_t c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ab, true, ab, c, false);
    });
#elif defined(__GFX12__)
    const int32x2_t ab = {kOnes, kOnes};
    v = run_mma_chains<int32x8_t>(iters, [ab](int32x8_t c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, ab, true, ab, c, false);
    });
#else
    (void)iters;
#endif
    if (t < n) out[t] = v;
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


} // namespace

RunResult run_gpu_int8_rocm(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(char4)";
    r.score_unit = "GOPS";

    const int dev = gpgpu::vendor::find_hip_device(setup.device).value_or(-1);
    if (dev < 0) { r.error = "no HIP device matched " + setup.device.id(); return r; }
    if (hipSetDevice(dev) != hipSuccess) { r.error = "hipSetDevice failed"; return r; }

    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, dev);
    const char* arch = prop.gcnArchName;

    const std::string arch_s = arch;
    const std::string base = arch_s.substr(0, arch_s.find(':'));
    auto is = [&](std::initializer_list<const char*> names) {
        for (const char* n : names) if (base == n) return true;
        return false;
    };

    Path path = Path::Packed;
    int K = 16;
    if (is({"gfx908", "gfx90a"})) {
        path = Path::Mma; r.path = "tensor(mfma 16x16x16 s8->s32)";
    } else if (is({"gfx940", "gfx941", "gfx942"})) {
        path = Path::Mma; K = 32; r.path = "tensor(mfma 16x16x32 s8->s32)";
    } else if (base == "gfx950") {
        path = Path::Mma; K = 64; r.path = "tensor(mfma 16x16x64 s8->s32)";
    } else if (base.rfind("gfx11", 0) == 0) {
        path = Path::Mma; r.path = "tensor(wmma rdna3 iu8)";
    } else if (base.rfind("gfx12", 0) == 0) {
        path = Path::Mma; r.path = "tensor(wmma rdna4 iu8)";
    } else if (is({"gfx906", "gfx1011", "gfx1012"}) || base.rfind("gfx103", 0) == 0) {
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
            const unsigned exp = tensor ? (unsigned)i8::reference_tensor(iters, K)
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
    r.work     = tensor ? i8::ops_tensor(tiles, i8::kIters, reps, K)
                        : i8::ops_packed(T, i8::kIters, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score    = score_giga(r.work, secs);
    r.correct  = ok;
    return r;
}

} // namespace bench
