/**
 * @file metal_runner.mm
 * @brief Metal fp32 dense-FMA runner (macOS only). Uses the linked Metal system
 *        framework directly. Dependent fma() chains in registers; sum stored to a
 *        shared buffer for readback + verification. Timed with MTLCommandBuffer
 *        GPU{Start,End}Time over a calibrated rep count.
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

NSString* const kKernelSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void fp32_fma(device float* out [[buffer(0)]],\n"
"                     constant uint& n     [[buffer(1)]],\n"
"                     constant uint& iters [[buffer(2)]],\n"
"                     uint t [[thread_position_in_grid]]) {\n"
"    if (t >= n) return;\n"
"    float acc0 = 1.0f + float((t + 0u) & 1023u) * (1.0f / 2048.0f);\n"
"    float acc1 = 1.0f + float((t + 1u) & 1023u) * (1.0f / 2048.0f);\n"
"    float acc2 = 1.0f + float((t + 2u) & 1023u) * (1.0f / 2048.0f);\n"
"    float acc3 = 1.0f + float((t + 3u) & 1023u) * (1.0f / 2048.0f);\n"
"    const float m = 0.9999f, add = 0.0001f;\n"
"    for (uint i = 0; i < iters; ++i) {\n"
"        for (int u = 0; u < 8; ++u) {\n"
"            acc0 = fma(acc0, m, add);\n"
"            acc1 = fma(acc1, m, add);\n"
"            acc2 = fma(acc2, m, add);\n"
"            acc3 = fma(acc3, m, add);\n"
"        }\n"
"    }\n"
"    out[t] = acc0 + acc1 + acc2 + acc3;\n"
"}\n";

} // namespace

RunResult run_gpu_fp32_metal(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(fp32 fma)";
    r.score_unit = "GFLOPS";

    @autoreleasepool {
        id<MTLDevice> device = find_metal_device(setup.device);
        if (!device) { r.error = "no Metal device matched " + setup.device.id(); return r; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { r.error = "newCommandQueue failed"; return r; }

        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kKernelSource options:nil error:&err];
        if (!lib) {
            r.error = std::string("newLibraryWithSource: ")
                    + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return r;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"fp32_fma"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            r.error = std::string("newComputePipelineState: ")
                    + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return r;
        }

        const std::uint32_t T = fp32::thread_count(setup.device);
        id<MTLBuffer> buf = [device newBufferWithLength:T * sizeof(float)
                                               options:MTLResourceStorageModeShared];
        if (!buf) { r.error = "newBufferWithLength failed"; return r; }

        const NSUInteger tg = std::min<NSUInteger>(fp32::kBlock, pso.maxTotalThreadsPerThreadgroup);

        auto time_launches = [&](std::uint32_t n, std::uint32_t iters, int reps) -> double {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            for (int i = 0; i < reps; ++i) {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:buf offset:0 atIndex:0];
                [enc setBytes:&n length:sizeof(n) atIndex:1];
                [enc setBytes:&iters length:sizeof(iters) atIndex:2];
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

        // --- pre-flight exactness ---
        time_launches(fp32::kPreflightThreads, fp32::kPreflightIters, 1);
        read_back(fp32::kPreflightThreads);
        for (std::uint32_t t = 0; t < fp32::kPreflightThreads && ok; ++t) {
            const float e = fp32::reference(t, fp32::kPreflightIters);
            if (!fp32::matches(host[t], e)) {
                char b[160];
                std::snprintf(b, sizeof(b),
                              "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t], e);
                r.error = b; ok = false;
            }
        }

        double secs = 0.0;
        int reps = 0;
        if (ok) {
            for (int w = 0; w < kWarmups; ++w) time_launches(T, fp32::kIters, 1);
            const double t_once = time_launches(T, fp32::kIters, 1);
            reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
            secs = time_launches(T, fp32::kIters, reps);
            read_back(T);
            constexpr std::uint32_t kSamples = 64;
            for (std::uint32_t s = 0; s < kSamples && ok; ++s) {
                const std::uint32_t t = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(s) * (T - 1) / (kSamples - 1));
                const float e = fp32::reference(t, fp32::kIters);
                if (!fp32::matches(host[t], e)) {
                    char b[160];
                    std::snprintf(b, sizeof(b),
                                  "sample mismatch at t=%u: got=%g expected=%g", t, host[t], e);
                    r.error = b; ok = false;
                }
            }
        }

        if (ok) {
            r.work     = fp32::flops(T, fp32::kIters, reps);
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
