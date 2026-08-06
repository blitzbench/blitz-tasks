#pragma once

/**
 * @file metal_match.hpp
 * @brief Metal device matching. Obj-C++ ONLY — include from a .mm TU. Extracted from
 *        the example Metal runner's find_matching_device().
 */

#import <Metal/Metal.h>

#include <cstdio>
#include <string>

#include <gpgpu/setup.hpp>

namespace bench {

// Match a gpgpu::Device to an MTLDevice by "metal-%llx" registryID. Falls back
// to the first enumerated device (mirrors the example) so a single-GPU Mac
// still runs when ids drift.
inline id<MTLDevice> find_metal_device(const gpgpu::Device& target) {
    NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
    if (!devs) return nil;
    for (id<MTLDevice> d in devs) {
        char id_buf[40];
        std::snprintf(id_buf, sizeof(id_buf), "metal-%llx",
                      (unsigned long long)[d registryID]);
        if (std::string(id_buf) == target.id()) return d;
    }
    return devs.count > 0 ? devs[0] : nil;
}

} // namespace bench
