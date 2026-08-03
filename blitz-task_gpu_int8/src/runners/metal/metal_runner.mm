/**
 * @file metal_runner.mm
 * @brief Metal int8 throughput runner (macOS only). MSL has no simdgroup int8 matrix
 *        type, so this uses the char4->int manual MAC path (path "simd(char4)").
 *        Dependent MAC chains in registers; sum stored to a shared buffer for readback
 *        + verification. Timed with MTLCommandBuffer GPU{Start,End}Time. Unit GOPS.
 *
 * as_type<char4>(acc) reinterprets the 32-bit accumulator as four SIGNED int8
 * lanes, exactly matching bench::i8::reference_packed.
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
"kernel void int8_packed(device uint* out [[buffer(0)]],\n"
"                        constant uint& n     [[buffer(1)]],\n"
"                        constant uint& iters [[buffer(2)]],\n"
"                        uint t [[thread_position_in_grid]]) {\n"
"    if (t >= n) return;\n"
"    const uint W = 0x04030201u;\n"
"    char4 wb = as_type<char4>(W);\n"
"    uint acc0 = 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * 1u;\n"
"    uint acc1 = 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * 2u;\n"
"    uint acc2 = 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * 3u;\n"
"    uint acc3 = 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * 4u;\n"
"    for (uint i = 0; i < iters; ++i) {\n"
"        for (int u = 0; u < 8; ++u) {\n"
"            char4 b0 = as_type<char4>(acc0);\n"
"            char4 b1 = as_type<char4>(acc1);\n"
"            char4 b2 = as_type<char4>(acc2);\n"
"            char4 b3 = as_type<char4>(acc3);\n"
"            acc0 += uint(int(b0.x)*int(wb.x) + int(b0.y)*int(wb.y) + int(b0.z)*int(wb.z) + int(b0.w)*int(wb.w));\n"
"            acc1 += uint(int(b1.x)*int(wb.x) + int(b1.y)*int(wb.y) + int(b1.z)*int(wb.z) + int(b1.w)*int(wb.w));\n"
"            acc2 += uint(int(b2.x)*int(wb.x) + int(b2.y)*int(wb.y) + int(b2.z)*int(wb.z) + int(b2.w)*int(wb.w));\n"
"            acc3 += uint(int(b3.x)*int(wb.x) + int(b3.y)*int(wb.y) + int(b3.z)*int(wb.z) + int(b3.w)*int(wb.w));\n"
"        }\n"
"    }\n"
"    out[t] = acc0 + acc1 + acc2 + acc3;\n"
"}\n";

} // namespace

RunResult run_gpu_int8_metal(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(char4)";
    r.score_unit = "GOPS";

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
        id<MTLFunction> fn = [lib newFunctionWithName:@"int8_packed"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            r.error = std::string("newComputePipelineState: ")
                    + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return r;
        }

        const std::uint32_t T = i8::thread_count(setup.device);
        id<MTLBuffer> buf = [device newBufferWithLength:T * sizeof(std::uint32_t)
                                               options:MTLResourceStorageModeShared];
        if (!buf) { r.error = "newBufferWithLength failed"; return r; }

        const NSUInteger tg = std::min<NSUInteger>(i8::kBlock, pso.maxTotalThreadsPerThreadgroup);

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

        std::vector<std::uint32_t> host(T);
        auto read_back = [&](std::uint32_t count) {
            std::memcpy(host.data(), [buf contents], count * sizeof(std::uint32_t));
        };
        auto verify = [&](std::uint32_t count, std::uint32_t iters, std::uint32_t stride) -> bool {
            for (std::uint32_t s = 0; s < count; ++s) {
                const std::uint32_t t = s * stride;
                const std::uint32_t exp = i8::reference_packed(t, iters);
                if (host[t] != exp) {
                    char b[160];
                    std::snprintf(b, sizeof(b),
                                  "mismatch at t=%u: got=%u expected=%u", t, host[t], exp);
                    r.error = b; return false;
                }
            }
            return true;
        };

        // --- pre-flight exactness ---
        time_launches(i8::kPreflightThreads, i8::kPreflightIters, 1);
        read_back(i8::kPreflightThreads);
        if (!verify(i8::kPreflightThreads, i8::kPreflightIters, 1)) return r;

        // --- warmup + calibrate ---
        for (int w = 0; w < kWarmups; ++w) time_launches(T, i8::kIters, 1);
        const double t_once = time_launches(T, i8::kIters, 1);
        const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);

        // --- timed run ---
        const double secs = time_launches(T, i8::kIters, reps);
        read_back(T);
        constexpr std::uint32_t kSamples = 64;
        const std::uint32_t stride = (T - 1) / (kSamples - 1);
        const bool ok = verify(kSamples, i8::kIters, stride);

        r.work     = i8::ops_packed(T, i8::kIters, reps);
        r.measured = std::chrono::duration<double>{secs};
        r.timings.kernel_compute = r.measured;
        r.score    = score_giga(r.work, secs);
        r.correct  = ok;
    }
    return r;
}

} // namespace bench

#endif // __APPLE__
