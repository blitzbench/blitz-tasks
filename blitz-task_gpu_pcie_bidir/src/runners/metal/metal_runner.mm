/**
 * @file metal_runner.mm
 * @brief Metal concurrent bidirectional transfer runner (macOS only).
 *
 * Two MTLCommandQueues run at the same time: one blits shared -> private
 * (the "upload" direction), the other blits private -> shared (the "download"
 * direction). Score is aggregate wall throughput over a single host clock window
 * bracketing commit -> waitUntilCompleted on both command buffers. Per-direction
 * GPU{Start,End}Time spans feed the timings block only.
 *
 * NOTE: Apple GPUs use unified memory — there is no host<->device interconnect,
 * so both blits read and write the SAME DRAM. The aggregate figure is honest
 * DRAM copy bandwidth under concurrent access, NOT a PCIe-style link rate. The
 * path string states this. Untestable locally (no Apple hardware) — CI only.
 */

#if defined(__APPLE__)

#include "metal_runner.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../bidir_params.hpp"

#include <bench/config.hpp>
#include <bench/metal_match.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace bench {

RunResult run_gpu_bidir_metal(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "2 command queues (unified memory: aggregate DRAM bandwidth)";
    r.score_unit = "GB/s";

    @autoreleasepool {
        id<MTLDevice> device = find_metal_device(setup.device);
        if (!device) { r.error = "no Metal device matched " + setup.device.id(); return r; }

        id<MTLCommandQueue> qUp = [device newCommandQueue];
        id<MTLCommandQueue> qDn = [device newCommandQueue];
        if (!qUp || !qDn) { r.error = "newCommandQueue failed"; return r; }

        const std::size_t S = bidir::buffer_bytes(setup.device);
        const std::size_t N = bidir::elem_count(S);

        id<MTLBuffer> up_src = [device newBufferWithLength:S options:MTLResourceStorageModeShared];
        id<MTLBuffer> up_dst = [device newBufferWithLength:S options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dn_src = [device newBufferWithLength:S options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dn_dst = [device newBufferWithLength:S options:MTLResourceStorageModeShared];
        if (!up_src || !up_dst || !dn_src || !dn_dst) { r.error = "newBufferWithLength failed"; return r; }

        bidir::fill_pattern(static_cast<std::uint32_t*>([up_src contents]), N);

        // Seed the private download source with the pattern (pre-window).
        {
            id<MTLCommandBuffer> cmd = [qUp commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:up_src sourceOffset:0 toBuffer:dn_src destinationOffset:0 size:S];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        auto encode = [&](id<MTLCommandQueue> q, id<MTLBuffer> src, id<MTLBuffer> dst,
                          int reps) -> id<MTLCommandBuffer> {
            id<MTLCommandBuffer> cmd = [q commandBuffer];
            for (int i = 0; i < reps; ++i) {
                id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:S];
                [blit endEncoding];
            }
            return cmd;
        };

        auto run_window = [&](int reps, double& h2d_secs, double& d2h_secs) -> double {
            id<MTLCommandBuffer> cUp = encode(qUp, up_src, up_dst, reps);
            id<MTLCommandBuffer> cDn = encode(qDn, dn_src, dn_dst, reps);
            const auto t0 = std::chrono::steady_clock::now();
            [cUp commit];
            [cDn commit];
            [cUp waitUntilCompleted];
            [cDn waitUntilCompleted];
            const auto t1 = std::chrono::steady_clock::now();
            const double su = [cUp GPUEndTime] - [cUp GPUStartTime];
            const double sd = [cDn GPUEndTime] - [cDn GPUStartTime];
            h2d_secs = su > 0.0 ? su : 0.0;
            d2h_secs = sd > 0.0 ? sd : 0.0;
            return std::chrono::duration<double>(t1 - t0).count();
        };

        double dummy_h = 0.0, dummy_d = 0.0;
        for (int w = 0; w < kWarmups; ++w) run_window(1, dummy_h, dummy_d);
        const double t_once = run_window(1, dummy_h, dummy_d);
        const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

        double h2d_secs = 0.0, d2h_secs = 0.0;
        const double wall = run_window(reps, h2d_secs, d2h_secs);

        // --- verify both directions once, outside the timed window ---
        const bool dn_ok = bidir::verify_sample(static_cast<std::uint32_t*>([dn_dst contents]), N);
        std::vector<std::uint32_t> back(N);
        {
            id<MTLBuffer> rb = [device newBufferWithLength:S options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmd = [qUp commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:up_dst sourceOffset:0 toBuffer:rb destinationOffset:0 size:S];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            std::memcpy(back.data(), [rb contents], S);
        }
        const bool up_ok = bidir::verify_sample(back.data(), N);
        const bool ok = up_ok && dn_ok;
        if (!ok) r.error = up_ok ? "download verification failed" : "upload verification failed";

        const std::uint64_t work = 2ull * static_cast<std::uint64_t>(reps) * S;
        r.work     = work;
        r.measured = std::chrono::duration<double>{wall};
        r.score    = score_giga(work, wall);
        r.correct  = ok;
        r.timings.copy_h2d      = std::chrono::duration<double>{h2d_secs};
        r.timings.copy_h2d_size = static_cast<std::size_t>(reps) * S;
        r.timings.copy_d2h      = std::chrono::duration<double>{d2h_secs};
        r.timings.copy_d2h_size = static_cast<std::size_t>(reps) * S;
        r.timings.total         = r.measured;
    }
    return r;
}

} // namespace bench

#endif // __APPLE__
