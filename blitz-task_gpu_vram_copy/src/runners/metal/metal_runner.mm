/**
 * @file metal_runner.mm
 * @brief Metal device-local VRAM copy runner (macOS only). Two PRIVATE buffers; the
 *        source is filled once (untimed) through a SHARED staging buffer. The timed
 *        loop records `reps` blit copyFromBuffer:src toBuffer:dst commands in one
 *        command buffer, timed via MTLCommandBuffer GPU{Start,End}Time.
 *        work = 2 * reps * S (a copy reads and writes the buffer).
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
#include <cstring>
#include <string>
#include <vector>

namespace bench {

RunResult run_gpu_vram_copy_metal(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "device copy (blit private->private)";
    r.score_unit = "GB/s";

    @autoreleasepool {
        id<MTLDevice> device = find_metal_device(setup.device);
        if (!device) { r.error = "no Metal device matched " + setup.device.id(); return r; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { r.error = "newCommandQueue failed"; return r; }

        const std::size_t   S = vram::buffer_bytes(setup.device);
        const std::uint32_t N = static_cast<std::uint32_t>(S / 4);

        id<MTLBuffer> src = [device newBufferWithLength:S options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dst = [device newBufferWithLength:S options:MTLResourceStorageModePrivate];
        id<MTLBuffer> staging = [device newBufferWithLength:S options:MTLResourceStorageModeShared];
        if (!src || !dst || !staging) { r.error = "newBufferWithLength failed"; return r; }

        // --- fill source via staging (untimed) ---
        {
            auto* p = static_cast<std::uint32_t*>([staging contents]);
            for (std::uint32_t i = 0; i < N; ++i) p[i] = vram::pattern(i);
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:staging sourceOffset:0 toBuffer:src destinationOffset:0 size:S];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        auto time_copies = [&](int reps) -> double {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            for (int i = 0; i < reps; ++i)
                [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:S];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            const double s = [cmd GPUEndTime] - [cmd GPUStartTime];
            return s > 0.0 ? s : 0.0;
        };

        for (int w = 0; w < kWarmups; ++w) time_copies(1);
        const double t_once = time_copies(1);
        const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

        const auto wall0 = std::chrono::steady_clock::now();
        const double secs = time_copies(reps);
        const auto wall1 = std::chrono::steady_clock::now();

        // --- verify: first + last 1 MiB window of dst (via staging) ---
        bool ok = true;
        std::string verr;
        const std::uint32_t w = std::min<std::uint32_t>(vram::kWindowU32, N);
        auto check_window = [&](std::uint32_t start) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:dst
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
                    std::snprintf(b, sizeof(b), "copy mismatch at u32[%u]: got=%u expected=%u",
                                  idx, p[i], vram::pattern(idx));
                    verr = b; ok = false;
                }
            }
        };
        check_window(0);
        if (ok && N > w) check_window(N - w);

        r.work                  = 2ull * static_cast<std::uint64_t>(reps) * S;
        r.measured              = std::chrono::duration<double>{secs};
        r.timings.copy_h2d      = r.measured;
        r.timings.copy_h2d_size = static_cast<std::size_t>(reps) * S;
        r.timings.total         = wall1 - wall0;
        r.score                 = score_giga(r.work, secs);
        r.correct               = ok;
        if (!ok) r.error = verr;
    }
    return r;
}

} // namespace bench

#endif // __APPLE__
