/**
 * @file metal_runner.mm
 * @brief Metal device-local VRAM write runner (macOS only). Grid-stride uint4 stores
 *        into a PRIVATE buffer, value pattern(i) = i * 2654435761u. Timed via
 *        MTLCommandBuffer GPU{Start,End}Time; verification windows blit to a SHARED
 *        staging buffer. work = reps * S (writes only).
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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bench {

namespace {

NSString* const kKernelSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void vram_write(device uint4* data [[buffer(0)]],\n"
"                       constant uint& n_vec4 [[buffer(1)]],\n"
"                       uint gid   [[thread_position_in_grid]],\n"
"                       uint gsize [[threads_per_grid]]) {\n"
"    const uint M = 2654435761u;\n"
"    for (uint v = gid; v < n_vec4; v += gsize) {\n"
"        uint base = v * 4u;\n"
"        data[v] = uint4(base * M, (base + 1u) * M, (base + 2u) * M, (base + 3u) * M);\n"
"    }\n"
"}\n";

} // namespace

RunResult run_gpu_vram_write_metal(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(uint4 stores)";
    r.score_unit = "GB/s";

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
        id<MTLFunction> fn = [lib newFunctionWithName:@"vram_write"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            r.error = std::string("newComputePipelineState: ")
                    + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return r;
        }

        const std::size_t   S = vram::buffer_bytes(setup.device);
        const std::uint32_t N = static_cast<std::uint32_t>(S / 4);
        const std::uint32_t n_vec4 = N / 4;
        const std::uint32_t w = std::min<std::uint32_t>(vram::kWindowU32, N);
        const std::uint32_t T = vram::thread_count(setup.device);

        id<MTLBuffer> buf = [device newBufferWithLength:S options:MTLResourceStorageModePrivate];
        id<MTLBuffer> staging = [device newBufferWithLength:static_cast<NSUInteger>(w) * 4
                                                   options:MTLResourceStorageModeShared];
        if (!buf || !staging) { r.error = "newBufferWithLength failed"; return r; }

        const NSUInteger tg = std::min<NSUInteger>(vram::kBlock, pso.maxTotalThreadsPerThreadgroup);

        auto time_writes = [&](int reps) -> double {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            for (int i = 0; i < reps; ++i) {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:buf offset:0 atIndex:0];
                [enc setBytes:&n_vec4 length:sizeof(n_vec4) atIndex:1];
                [enc dispatchThreads:MTLSizeMake(T, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
            }
            [cmd commit];
            [cmd waitUntilCompleted];
            const double s = [cmd GPUEndTime] - [cmd GPUStartTime];
            return s > 0.0 ? s : 0.0;
        };

        for (int i = 0; i < kWarmups; ++i) time_writes(1);
        const double t_once = time_writes(1);
        const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

        const auto wall0 = std::chrono::steady_clock::now();
        const double secs = time_writes(reps);
        const auto wall1 = std::chrono::steady_clock::now();

        bool ok = true;
        std::string verr;
        auto check_window = [&](std::uint32_t start) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:buf
                    sourceOffset:static_cast<NSUInteger>(start) * 4
                        toBuffer:staging
               destinationOffset:0
                            size:static_cast<NSUInteger>(w) * 4];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            const auto* p = static_cast<const std::uint32_t*>([staging contents]);
            for (std::uint32_t i = 0; i < w && ok; ++i) {
                const std::uint32_t idx = start + i;
                if (p[i] != vram::pattern(idx)) {
                    char b[128];
                    std::snprintf(b, sizeof(b), "write mismatch at u32[%u]: got=%u expected=%u",
                                  idx, p[i], vram::pattern(idx));
                    verr = b; ok = false;
                }
            }
        };
        check_window(0);
        if (ok && N > w) check_window(N - w);

        r.work                   = static_cast<std::uint64_t>(reps) * S;
        r.measured               = std::chrono::duration<double>{secs};
        r.timings.kernel_compute = r.measured;
        r.timings.total          = wall1 - wall0;
        r.score                  = score_giga(r.work, secs);
        r.correct                = ok;
        if (!ok) r.error = verr;
    }
    return r;
}

} // namespace bench

#endif // __APPLE__
