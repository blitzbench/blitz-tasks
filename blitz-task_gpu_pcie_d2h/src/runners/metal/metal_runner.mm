/**
 * @file metal_runner.mm
 * @brief Metal device-to-host transfer runner (macOS only).
 *
 * A storageModePrivate source (prefilled once from a shared staging buffer) is
 * blit-copied into a storageModeShared destination with an MTLBlitCommandEncoder,
 * timed with MTLCommandBuffer GPU{Start,End}Time. Apple GPUs share DRAM with the
 * CPU, so this measures a DRAM↔DRAM blit — not a discrete interconnect; the path
 * says so. The received shared destination is sample-checked once. Mirror of h2d.
 */

#if defined(__APPLE__)

#include "metal_runner.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <bench/config.hpp>
#include <bench/metal_match.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace bench {

namespace {

constexpr std::uint64_t kMiB  = 1024ull * 1024ull;
constexpr std::uint64_t k2GiB = 2ull * 1024ull * 1024ull * 1024ull;

std::size_t transfer_bytes(const gpgpu::Device& d) {
    std::uint64_t mem = d.memory().value_or(k2GiB);
    std::uint64_t s   = std::min<std::uint64_t>(256ull * kMiB, mem / 8ull);
    s &= ~std::uint64_t(3);
    if (s < 4) s = 4;
    return static_cast<std::size_t>(s);
}

inline std::uint32_t pattern_at(std::size_t i) {
    return static_cast<std::uint32_t>(i) * 2654435761u;
}

bool sample_ok(const std::uint32_t* host, std::size_t count) {
    constexpr std::size_t kSamples = 64;
    if (count == 0) return false;
    for (std::size_t s = 0; s < kSamples; ++s) {
        std::size_t i = (count == 1) ? 0 : (s * (count - 1) / (kSamples - 1));
        if (host[i] != pattern_at(i)) return false;
    }
    return true;
}

} // namespace

RunResult run_gpu_d2h_metal(const gpgpu::Setup& setup) {
    RunResult r;
    r.path       = "private->shared blit (unified mem)";
    r.score_unit = "GB/s";

    @autoreleasepool {
        id<MTLDevice> device = find_metal_device(setup.device);
        if (!device) { r.error = "no Metal device matched " + setup.device.id(); return r; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { r.error = "newCommandQueue failed"; return r; }

        const std::size_t S     = transfer_bytes(setup.device);
        const std::size_t count = S / sizeof(std::uint32_t);

        id<MTLBuffer> src = [device newBufferWithLength:S options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dst = [device newBufferWithLength:S options:MTLResourceStorageModeShared];
        if (!src || !dst) { r.error = "newBufferWithLength failed"; return r; }

        // Prefill the private source from a shared staging buffer (untimed).
        {
            id<MTLBuffer> staging = [device newBufferWithLength:S options:MTLResourceStorageModeShared];
            std::uint32_t* p = static_cast<std::uint32_t*>([staging contents]);
            for (std::size_t i = 0; i < count; ++i) p[i] = pattern_at(i);
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
            [enc copyFromBuffer:staging sourceOffset:0 toBuffer:src destinationOffset:0 size:S];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        auto time_copies = [&](int reps) -> double {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
            for (int i = 0; i < reps; ++i)
                [enc copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:S];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            const double s = [cmd GPUEndTime] - [cmd GPUStartTime];
            return s > 0.0 ? s : 0.0;
        };

        for (int w = 0; w < kWarmups; ++w) time_copies(1);
        const double t_once = time_copies(1);
        const int    reps   = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

        const auto wall0 = std::chrono::steady_clock::now();
        double secs = time_copies(reps);
        const auto wall1 = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(wall1 - wall0).count();
        if (secs <= 0.0) secs = wall;

        // --- verify once: the received shared destination ---
        std::vector<std::uint32_t> host(count);
        std::memcpy(host.data(), [dst contents], S);
        const bool ok = sample_ok(host.data(), count);

        r.work                  = static_cast<std::uint64_t>(reps) * S;
        r.measured              = std::chrono::duration<double>{secs};
        r.timings.copy_d2h      = r.measured;
        r.timings.copy_d2h_size = S;
        r.timings.total         = std::chrono::duration<double>{wall};
        r.score                 = score_giga(r.work, secs);
        r.correct               = ok;
        if (!ok) r.error = "verification failed";
    }
    return r;
}

} // namespace bench

#endif // __APPLE__
