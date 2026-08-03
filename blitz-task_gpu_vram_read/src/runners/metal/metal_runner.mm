/**
 * @file metal_runner.mm
 * @brief Metal device-local VRAM read runner (macOS only). A PRIVATE input buffer is
 *        prefilled (untimed) via a SHARED staging buffer. Grid-stride uint4 loads
 *        accumulate per-thread (u32 wraparound) and write one uint per thread to a
 *        PRIVATE out-buffer. Timed via MTLCommandBuffer GPU{Start,End}Time. work =
 *        reps * S. Verified against the closed form (out blitted to staging, summed).
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
"kernel void vram_read(device const uint4* data [[buffer(0)]],\n"
"                      constant uint& n_vec4    [[buffer(1)]],\n"
"                      device uint* out         [[buffer(2)]],\n"
"                      uint gid   [[thread_position_in_grid]],\n"
"                      uint gsize [[threads_per_grid]]) {\n"
"    uint acc = 0u;\n"
"    for (uint v = gid; v < n_vec4; v += gsize) {\n"
"        uint4 x = data[v];\n"
"        acc += x.x + x.y + x.z + x.w;\n"
"    }\n"
"    out[gid] = acc;\n"
"}\n";

} // namespace

RunResult run_gpu_vram_read_metal(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "simd(uint4 loads)";
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
        id<MTLFunction> fn = [lib newFunctionWithName:@"vram_read"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            r.error = std::string("newComputePipelineState: ")
                    + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return r;
        }

        const std::size_t   S = vram::buffer_bytes(setup.device);
        const std::uint32_t N = static_cast<std::uint32_t>(S / 4);
        const std::uint32_t n_vec4 = N / 4;
        const std::uint32_t T = vram::thread_count(setup.device);
        const std::size_t   out_bytes = static_cast<std::size_t>(T) * 4;

        id<MTLBuffer> in_buf  = [device newBufferWithLength:S options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out_buf = [device newBufferWithLength:out_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> staging = [device newBufferWithLength:S options:MTLResourceStorageModeShared];
        if (!in_buf || !out_buf || !staging) { r.error = "newBufferWithLength failed"; return r; }

        // --- prefill input via staging (untimed) ---
        {
            auto* p = static_cast<std::uint32_t*>([staging contents]);
            for (std::uint32_t i = 0; i < N; ++i) p[i] = vram::pattern(i);
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:staging sourceOffset:0 toBuffer:in_buf destinationOffset:0 size:S];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        const NSUInteger tg = std::min<NSUInteger>(vram::kBlock, pso.maxTotalThreadsPerThreadgroup);

        auto time_reads = [&](int reps) -> double {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            for (int i = 0; i < reps; ++i) {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:in_buf offset:0 atIndex:0];
                [enc setBytes:&n_vec4 length:sizeof(n_vec4) atIndex:1];
                [enc setBuffer:out_buf offset:0 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(T, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
            }
            [cmd commit];
            [cmd waitUntilCompleted];
            const double s = [cmd GPUEndTime] - [cmd GPUStartTime];
            return s > 0.0 ? s : 0.0;
        };

        for (int i = 0; i < kWarmups; ++i) time_reads(1);
        const double t_once = time_reads(1);
        const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

        const auto wall0 = std::chrono::steady_clock::now();
        const double secs = time_reads(reps);
        const auto wall1 = std::chrono::steady_clock::now();

        // --- verify: exact closed-form checksum (out blitted to staging) ---
        bool ok = true;
        std::string verr;
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:out_buf sourceOffset:0 toBuffer:staging destinationOffset:0 size:out_bytes];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            const auto* p = static_cast<const std::uint32_t*>([staging contents]);
            std::uint32_t sum = 0u;
            for (std::uint32_t i = 0; i < T; ++i) sum += p[i];
            const std::uint32_t expected = vram::expected_sum(S);
            if (sum != expected) {
                char b[128];
                std::snprintf(b, sizeof(b), "read checksum mismatch: got=%u expected=%u", sum, expected);
                verr = b; ok = false;
            }
        }

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
