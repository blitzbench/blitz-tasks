/**
 * @file metal_runner.mm
 * @brief Metal fp16 throughput runner (macOS only) — simdgroup_matrix tensor path with
 *        a packed half4 fallback. Runtime-compiled MSL, selected by GPU family:
 *
 *   Apple7+ (@available macOS 11)  -> simdgroup_half8x8 + simdgroup_multiply_
 *                                     accumulate (K=8)  "tensor(simdgroup 8x8x8 f16->f32)"
 *   else                           -> half4 fma chains  "simd(half4 fma)"
 *
 * The tensor kernel accumulates all-1/16 8x8 tiles so every C element equals
 * iters*8*(1/16)^2 exactly. Timed with MTLCommandBuffer GPU{Start,End}Time.
 * CANNOT be built or run on the local (non-Apple) box; needs Apple-silicon CI.
 */

#if defined(__APPLE__)

#include "metal_runner.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../kernel_params.hpp"

#include <bench/config.hpp>
#include <bench/metal_match.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace bench {

namespace {

// Both kernels in one library. seed() mirrors kernel_params.hpp seed_val().
NSString* const kKernelSource = @
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
"inline half seed_h(uint t, uint k, uint l) {\n"
"    uint idx = (t + 7u*k + 13u*l) & 255u;\n"
"    return half(1.0f + float(idx) * (1.0f/256.0f));\n"
"}\n"
// --- tensor: simdgroup 8x8x8 f16 -> f32, one simdgroup (32 threads) per group ---
"kernel void fp16_simdgroup(device float* out [[buffer(0)]],\n"
"                           constant uint& warps [[buffer(1)]],\n"
"                           constant uint& iters [[buffer(2)]],\n"
"                           uint gid [[thread_position_in_grid]]) {\n"
"    threadgroup float tg[64];\n"
"    simdgroup_half8x8  a = simdgroup_half8x8(half(0.0625h));\n"
"    simdgroup_half8x8  b = simdgroup_half8x8(half(0.0625h));\n"
"    simdgroup_float8x8 c = simdgroup_float8x8(0.0f);\n"
"    for (uint i = 0; i < iters; ++i)\n"
"        simdgroup_multiply_accumulate(c, a, b, c);\n"
"    simdgroup_store(c, tg, 8);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    out[gid] = tg[0];\n"
"}\n"
// --- fallback: packed half4 fma chains ---
"kernel void fp16_half4(device float* out [[buffer(0)]],\n"
"                       constant uint& n     [[buffer(1)]],\n"
"                       constant uint& iters [[buffer(2)]],\n"
"                       constant float& mf   [[buffer(3)]],\n"
"                       constant float& af   [[buffer(4)]],\n"
"                       uint t [[thread_position_in_grid]]) {\n"
"    if (t >= n) return;\n"
"    half4 m = half4(half(mf)), a = half4(half(af));\n"
"    half4 acc[4];\n"
"    for (uint k = 0; k < 4; ++k)\n"
"        acc[k] = half4(seed_h(t,k,0), seed_h(t,k,1), seed_h(t,k,2), seed_h(t,k,3));\n"
"    for (uint i = 0; i < iters; ++i)\n"
"        for (int u = 0; u < 8; ++u)\n"
"            for (int k = 0; k < 4; ++k)\n"
"                acc[k] = fma(acc[k], m, a);\n"
"    float s = 0.0f;\n"
"    for (int k = 0; k < 4; ++k) { float4 f = float4(acc[k]); s += f.x+f.y+f.z+f.w; }\n"
"    out[t] = s;\n"
"}\n";

} // namespace

RunResult run_gpu_fp16_metal(const gpgpu::Setup& setup) {
    RunResult r;
    r.score_unit = "GFLOPS";
    r.path       = "simd(half4 fma)";

    @autoreleasepool {
        id<MTLDevice> device = find_metal_device(setup.device);
        if (!device) { r.error = "no Metal device matched " + setup.device.id(); return r; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { r.error = "newCommandQueue failed"; return r; }

        bool use_tensor = false;
        if (@available(macOS 11.0, *)) {
            use_tensor = [device supportsFamily:MTLGPUFamilyApple7];
        }
        r.path = use_tensor ? "tensor(simdgroup 8x8x8 f16->f32)" : "simd(half4 fma)";

        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kKernelSource options:nil error:&err];
        if (!lib) {
            r.error = std::string("newLibraryWithSource: ")
                    + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return r;
        }
        NSString* fname = use_tensor ? @"fp16_simdgroup" : @"fp16_half4";
        id<MTLFunction> fn = [lib newFunctionWithName:fname];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            r.error = std::string("newComputePipelineState: ")
                    + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return r;
        }

        const std::uint32_t T = fp16::thread_count(setup.device);
        id<MTLBuffer> buf = [device newBufferWithLength:T * sizeof(float)
                                               options:MTLResourceStorageModeShared];
        if (!buf) { r.error = "newBufferWithLength failed"; return r; }

        // Tensor path: one simdgroup (32 lanes) per threadgroup. Packed: kBlock.
        const NSUInteger tg = use_tensor
            ? 32u
            : std::min<NSUInteger>(fp16::kBlock, pso.maxTotalThreadsPerThreadgroup);

        auto time_launches = [&](std::uint32_t n, std::uint32_t iters, float mf, float af,
                                 int reps) -> double {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            for (int i = 0; i < reps; ++i) {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:buf offset:0 atIndex:0];
                [enc setBytes:&n length:sizeof(n) atIndex:1];
                [enc setBytes:&iters length:sizeof(iters) atIndex:2];
                if (!use_tensor) {
                    [enc setBytes:&mf length:sizeof(mf) atIndex:3];
                    [enc setBytes:&af length:sizeof(af) atIndex:4];
                }
                [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
            }
            [cmd commit];
            [cmd waitUntilCompleted];
            const double s = [cmd GPUEndTime] - [cmd GPUStartTime];
            return s > 0.0 ? s : 0.0;
        };

        std::vector<float> host(T);
        auto read_back = [&](std::uint32_t count) {
            std::memcpy(host.data(), [buf contents], count * sizeof(float));
        };

        bool ok = true;
        const std::uint32_t iters = use_tensor ? fp16::kTensorIters : fp16::kIters;

        // --- pre-flight ---
        time_launches(fp16::kPreflightThreads,
                      use_tensor ? fp16::kTensorPreflightIters : fp16::kPreflightIters,
                      fp16::kPreM, fp16::kPreA, 1);
        read_back(fp16::kPreflightThreads);
        for (std::uint32_t t = 0; t < fp16::kPreflightThreads && ok; ++t) {
            if (use_tensor) {
                if (!fp16::tensor_ok(host[t], fp16::kTensorPreflightIters, 8)) {
                    char b[160];
                    std::snprintf(b, sizeof(b), "pre-flight mismatch at t=%u: got=%g expected=%g",
                                  t, host[t], fp16::tensor_out(fp16::kTensorPreflightIters, 8));
                    r.error = b; ok = false;
                }
            } else {
                const double e = fp16::simd_preflight_out(t, 4, fp16::kPreflightIters);
                if (!fp16::matches_exact(host[t], e)) {
                    char b[160];
                    std::snprintf(b, sizeof(b), "pre-flight mismatch at t=%u: got=%g expected=%g",
                                  t, host[t], e);
                    r.error = b; ok = false;
                }
            }
        }

        double secs = 0.0;
        int reps = 0;
        if (ok) {
            for (int w = 0; w < kWarmups; ++w) time_launches(T, iters, fp16::kMainM, fp16::kMainA, 1);
            const double t_once = time_launches(T, iters, fp16::kMainM, fp16::kMainA, 1);
            reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
            secs = time_launches(T, iters, fp16::kMainM, fp16::kMainA, reps);
            read_back(T);
            for (std::uint32_t s = 0; s < 64 && ok; ++s) {
                const std::uint32_t t = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(s) * (T - 1) / 63);
                if (use_tensor) {
                    if (!fp16::tensor_ok(host[t], iters, 8)) {
                        char b[160];
                        std::snprintf(b, sizeof(b), "sample mismatch at t=%u: got=%g expected=%g",
                                      t, host[t], fp16::tensor_out(iters, 8));
                        r.error = b; ok = false;
                    }
                } else if (!fp16::simd_main_ok(host[t], 4)) {
                    char b[160];
                    std::snprintf(b, sizeof(b), "sample out of envelope at t=%u: got=%g", t, host[t]);
                    r.error = b; ok = false;
                }
            }
        }

        if (ok) {
            const std::uint32_t warps = T / 32u;
            r.work     = use_tensor ? fp16::tensor_flops(warps, iters, reps, 8, 8, 8)
                                    : fp16::simd_flops(T, iters, reps, 4);
            r.measured = std::chrono::duration<double>{secs};
            r.timings.kernel_compute = r.measured;
            r.score    = score_giga(r.work, secs);
            r.correct  = true;
        }
    }
    return r;
}

} // namespace bench

#endif // __APPLE__
