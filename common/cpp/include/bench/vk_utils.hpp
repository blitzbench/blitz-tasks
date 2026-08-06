#pragma once

/**
 * @file vk_utils.hpp
 * @brief Vulkan device matching + buffer / queue / timestamp utilities. Included ONLY
 *        from a Vulkan runner TU that carries the Vulkan headers (with prototypes:
 *        the Vulkan::Vulkan loader provides them). Extracted / generalized from the
 *        example Vulkan runner.
 */

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <gpgpu/setup.hpp>

namespace bench {

// Stable Vulkan device id string, matching what the gpgpu Vulkan backend emits.
inline std::string vk_format_id(const VkPhysicalDeviceProperties& p) {
    const auto& u = p.pipelineCacheUUID;
    char buf[80];
    std::snprintf(buf, sizeof(buf),
                  "vk-%04x:%04x-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  p.vendorID, p.deviceID,
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

inline VkPhysicalDevice find_vk_device(VkInstance inst, const gpgpu::Device& target) {
    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (n == 0) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> phys(n);
    vkEnumeratePhysicalDevices(inst, &n, phys.data());
    for (auto pd : phys) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (vk_format_id(props) == target.id()) return pd;
    }
    return VK_NULL_HANDLE;
}

// First queue family holding ALL bits in `flags`, or UINT32_MAX.
inline std::uint32_t find_queue_family(VkPhysicalDevice pd, VkQueueFlags flags) {
    std::uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    if (n == 0) return UINT32_MAX;
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());
    for (std::uint32_t k = 0; k < n; ++k) {
        if ((fams[k].queueFlags & flags) == flags) return k;
    }
    return UINT32_MAX;
}

// A queue family with `flags` set but WITHOUT any bit in `without` — useful to
// find a transfer-only or dedicated queue. Falls back to UINT32_MAX.
inline std::uint32_t find_queue_family_excluding(VkPhysicalDevice pd,
                                                 VkQueueFlags     flags,
                                                 VkQueueFlags     without) {
    std::uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    if (n == 0) return UINT32_MAX;
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());
    for (std::uint32_t k = 0; k < n; ++k) {
        if ((fams[k].queueFlags & flags) == flags && !(fams[k].queueFlags & without)) return k;
    }
    return UINT32_MAX;
}

inline std::uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties& mp,
                                      std::uint32_t          allowed,
                                      VkMemoryPropertyFlags  wanted) {
    for (std::uint32_t k = 0; k < mp.memoryTypeCount; ++k) {
        if ((allowed & (1u << k)) &&
            (mp.memoryTypes[k].propertyFlags & wanted) == wanted) return k;
    }
    return UINT32_MAX;
}

struct VkBufferAlloc {
    VkBuffer       buf{VK_NULL_HANDLE};
    VkDeviceMemory mem{VK_NULL_HANDLE};
};

// Create a buffer of `bytes` with `usage`, backed by a memory type carrying
// `memflags`. Returns false (and cleans up) on any failure.
inline bool create_buffer(VkDevice                                dev,
                          const VkPhysicalDeviceMemoryProperties& mp,
                          VkDeviceSize                            bytes,
                          VkBufferUsageFlags                      usage,
                          VkMemoryPropertyFlags                   memflags,
                          VkBufferAlloc&                          out) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size        = bytes;
    bi.usage       = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, nullptr, &out.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, out.buf, &req);
    const std::uint32_t mt = find_memory_type(mp, req.memoryTypeBits, memflags);
    if (mt == UINT32_MAX) return false;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(dev, &ai, nullptr, &out.mem) != VK_SUCCESS) return false;
    return vkBindBufferMemory(dev, out.buf, out.mem, 0) == VK_SUCCESS;
}

inline void destroy_buffer(VkDevice dev, VkBufferAlloc& b) {
    if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
    b = {};
}

// Two-slot timestamp query pool (index 0 = begin, 1 = end). Returns
// VK_NULL_HANDLE if the device can't do compute-queue timestamps.
inline VkQueryPool create_timestamp_pool(VkDevice dev, std::uint32_t count = 2) {
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = count;
    VkQueryPool qp = VK_NULL_HANDLE;
    vkCreateQueryPool(dev, &qpci, nullptr, &qp);
    return qp;
}

// Read a begin/end timestamp pair and convert to seconds using the device's
// timestampPeriod (ns per tick). Returns false if the delta is unusable.
inline bool read_timestamp_span(VkDevice      dev,
                                VkQueryPool   qp,
                                float         timestamp_period_ns,
                                double&       out_seconds) {
    std::uint64_t ts[2] = {0, 0};
    if (vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(ts), ts, sizeof(std::uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
        return false;
    if (ts[1] <= ts[0]) return false;
    out_seconds = (ts[1] - ts[0]) * static_cast<double>(timestamp_period_ns) / 1.0e9;
    return true;
}

} // namespace bench
