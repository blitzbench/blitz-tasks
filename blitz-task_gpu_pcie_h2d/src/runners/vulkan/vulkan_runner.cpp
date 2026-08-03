/**
 * @file vulkan_runner.cpp
 * @brief Vulkan host-to-device transfer runner — SDK-required variant.
 *
 * A HOST_VISIBLE|HOST_COHERENT staging buffer (filled once, untimed) is copied
 * into a DEVICE_LOCAL destination with vkCmdCopyBuffer. The rep loop is bracketed
 * by timestamp queries (scaled by limits.timestampPeriod; the compute queue's
 * timestampValidBits are checked). On a UMA / software device with no
 * DEVICE_LOCAL-only heap the copy still runs and the path notes it. Verified once
 * by copying the destination back into a host-visible readback buffer.
 */

#include "vulkan_runner.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <bench/config.hpp>
#include <bench/vk_utils.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace bench {

namespace {

constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t k2GiB = 2ull * 1024ull * 1024ull * 1024ull;

std::size_t transfer_bytes(const gpgpu::Device& d) {
  std::uint64_t mem = d.memory().value_or(k2GiB);
  std::uint64_t s = std::min<std::uint64_t>(256ull * kMiB, mem / 8ull);
  s &= ~std::uint64_t(3);
  if (s < 4) s = 4;
  return static_cast<std::size_t>(s);
}

inline std::uint32_t pattern_at(std::size_t i) { return static_cast<std::uint32_t>(i) * 2654435761u; }

bool sample_ok(const std::uint32_t* host, std::size_t count) {
  constexpr std::size_t kSamples = 64;
  if (count == 0) return false;
  for (std::size_t s = 0; s < kSamples; ++s) {
    std::size_t i = (count == 1) ? 0 : (s * (count - 1) / (kSamples - 1));
    if (host[i] != pattern_at(i)) return false;
  }
  return true;
}

/**
 * @brief True if the device exposes a DEVICE_LOCAL heap that is NOT host-visible.
 *
 * That is a discrete-style upload path rather than a UMA shared heap.
 *
 * @param mp
 * @return
 */
bool has_discrete_heap(const VkPhysicalDeviceMemoryProperties& mp) {
  for (std::uint32_t k = 0; k < mp.memoryTypeCount; ++k) {
    const auto f = mp.memoryTypes[k].propertyFlags;
    if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) return true;
  }
  return false;
}

}  // namespace

RunResult run_gpu_h2d_vulkan(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "staging->device copy";
  r.score_unit = "GB/s";

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_data_transfer_h2d";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst = VK_NULL_HANDLE;
  if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
    r.error = "vkCreateInstance failed";
    return r;
  }

  VkPhysicalDevice pd = find_vk_device(inst, setup.device);
  if (!pd) {
    vkDestroyInstance(inst, nullptr);
    r.error = "no Vulkan device matched " + setup.device.id();
    return r;
  }

  VkPhysicalDeviceProperties pd_props{};
  vkGetPhysicalDeviceProperties(pd, &pd_props);
  const float ts_period_ns = pd_props.limits.timestampPeriod;

  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  if (!has_discrete_heap(mp)) r.path = "staging->device copy (UMA)";

  const std::uint32_t qfam = find_queue_family(pd, VK_QUEUE_COMPUTE_BIT);
  if (qfam == UINT32_MAX) {
    vkDestroyInstance(inst, nullptr);
    r.error = "no compute/transfer queue family";
    return r;
  }

  // Does this queue family produce meaningful timestamps?
  bool ts_ok = pd_props.limits.timestampComputeAndGraphics != 0;
  {
    std::uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp.data());
    if (qfam >= nqf || qfp[qfam].timestampValidBits == 0) ts_ok = false;
  }

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice dev = VK_NULL_HANDLE;
  if (vkCreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS) {
    vkDestroyInstance(inst, nullptr);
    r.error = "vkCreateDevice failed";
    return r;
  }
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(dev, qfam, 0, &queue);

  const std::size_t S = transfer_bytes(setup.device);
  const std::size_t count = S / sizeof(std::uint32_t);
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(S);

  VkBufferAlloc staging{}, dst{}, readback{};
  auto teardown = [&](VkQueryPool qp, VkCommandPool cp) {
    if (qp) vkDestroyQueryPool(dev, qp, nullptr);
    if (cp) vkDestroyCommandPool(dev, cp, nullptr);
    destroy_buffer(dev, readback);
    destroy_buffer(dev, dst);
    destroy_buffer(dev, staging);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
  };

  if (!create_buffer(dev, mp, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging) ||
      !create_buffer(dev, mp, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dst) ||
      !create_buffer(dev, mp, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, readback)) {
    teardown(VK_NULL_HANDLE, VK_NULL_HANDLE);
    r.error = "buffer alloc failed";
    return r;
  }

  // Fill the staging buffer once (untimed).
  {
    void* m = nullptr;
    vkMapMemory(dev, staging.mem, 0, bytes, 0, &m);
    std::uint32_t* p = static_cast<std::uint32_t*>(m);
    for (std::size_t i = 0; i < count; ++i) p[i] = pattern_at(i);
    vkUnmapMemory(dev, staging.mem);
  }

  VkQueryPool qpool = ts_ok ? create_timestamp_pool(dev, 2) : VK_NULL_HANDLE;

  VkCommandPoolCreateInfo cpoolci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpoolci.queueFamilyIndex = qfam;
  cpoolci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool cpool = VK_NULL_HANDLE;
  vkCreateCommandPool(dev, &cpoolci, nullptr, &cpool);
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = cpool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(dev, &cbai, &cb);

  auto submit_wait = [&]() {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
  };

  auto time_copies = [&](int reps) -> double {
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    if (qpool) {
      vkCmdResetQueryPool(cb, qpool, 0, 2);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    }
    VkBufferCopy region{0, 0, bytes};
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    for (int i = 0; i < reps; ++i) {
      vkCmdCopyBuffer(cb, staging.buf, dst.buf, 1, &region);
      if (i + 1 < reps)
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr,
                             0, nullptr);
    }
    if (qpool) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
    vkEndCommandBuffer(cb);
    submit_wait();
    double secs = 0.0;
    if (qpool) read_timestamp_span(dev, qpool, ts_period_ns, secs);
    return secs;
  };

  for (int w = 0; w < kWarmups; ++w) time_copies(1);
  const double t_once = time_copies(1);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  const auto wall0 = std::chrono::steady_clock::now();
  double secs = time_copies(reps);
  const auto wall1 = std::chrono::steady_clock::now();
  const double wall = std::chrono::duration<double>(wall1 - wall0).count();
  if (secs <= 0.0) secs = wall;

  // --- verify once: copy dst → readback, then sample ---
  vkResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bi);
  VkBufferCopy region{0, 0, bytes};
  vkCmdCopyBuffer(cb, dst.buf, readback.buf, 1, &region);
  vkEndCommandBuffer(cb);
  submit_wait();

  std::vector<std::uint32_t> host(count);
  {
    void* m = nullptr;
    vkMapMemory(dev, readback.mem, 0, bytes, 0, &m);
    std::memcpy(host.data(), m, S);
    vkUnmapMemory(dev, readback.mem);
  }
  const bool ok = sample_ok(host.data(), count);

  teardown(qpool, cpool);

  r.work = static_cast<std::uint64_t>(reps) * S;
  r.measured = std::chrono::duration<double>{secs};
  r.timings.copy_h2d = r.measured;
  r.timings.copy_h2d_size = S;
  r.timings.total = std::chrono::duration<double>{wall};
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  if (!ok) r.error = "verification failed";
  return r;
}

}  // namespace bench
