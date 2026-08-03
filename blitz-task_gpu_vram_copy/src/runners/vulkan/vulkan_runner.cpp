/**
 * @file vulkan_runner.cpp
 * @brief Vulkan device-local VRAM copy runner. Two DEVICE_LOCAL buffers; the source is
 *        filled once (untimed) through a HOST_VISIBLE staging buffer. The timed loop
 *        records `reps` full-buffer vkCmdCopyBuffer(src->dst) commands, serialized by a
 *        transfer memory barrier, bracketed by timestamp queries.
 *        work = 2 * reps * S (a copy reads and writes the buffer).
 */

#include "vulkan_runner.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <bench/config.hpp>
#include <bench/vk_utils.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../kernel_params.hpp"

namespace bench {

RunResult run_gpu_vram_copy_vulkan(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "device copy (vkCmdCopyBuffer)";
  r.score_unit = "GB/s";

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_vram_copy";
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
  const bool ts_ok = pd_props.limits.timestampComputeAndGraphics != 0;

  const std::uint32_t qfam = find_queue_family(pd, VK_QUEUE_COMPUTE_BIT);
  if (qfam == UINT32_MAX) {
    vkDestroyInstance(inst, nullptr);
    r.error = "no compute queue family";
    return r;
  }

  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);

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

  const std::size_t S = vram::buffer_bytes(setup.device);
  const std::uint32_t N = static_cast<std::uint32_t>(S / 4);
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(S);

  VkBufferAlloc src{}, dst{}, staging{};
  const VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bool alloc_ok = create_buffer(dev, mp, bytes, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src) &&
                  create_buffer(dev, mp, bytes, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dst) &&
                  create_buffer(dev, mp, bytes, dev_usage,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);

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

  VkQueryPool qpool = ts_ok ? create_timestamp_pool(dev, 2) : VK_NULL_HANDLE;

  auto submit_wait = [&]() {
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
  };
  auto begin_cb = [&]() {
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
  };

  auto cleanup = [&]() {
    if (qpool) vkDestroyQueryPool(dev, qpool, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    destroy_buffer(dev, src);
    destroy_buffer(dev, dst);
    destroy_buffer(dev, staging);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
  };

  if (!alloc_ok) {
    r.error = "buffer alloc failed";
    cleanup();
    return r;
  }

  // --- fill source via staging (untimed) ---
  {
    void* mapped = nullptr;
    vkMapMemory(dev, staging.mem, 0, bytes, 0, &mapped);
    auto* p = static_cast<std::uint32_t*>(mapped);
    for (std::uint32_t i = 0; i < N; ++i) p[i] = vram::pattern(i);
    vkUnmapMemory(dev, staging.mem);
    begin_cb();
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cb, staging.buf, src.buf, 1, &region);
    submit_wait();
  }

  auto time_copies = [&](int reps) -> double {
    begin_cb();
    if (qpool) {
      vkCmdResetQueryPool(cb, qpool, 0, 2);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    }
    VkBufferCopy region{0, 0, bytes};
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    for (int i = 0; i < reps; ++i) {
      vkCmdCopyBuffer(cb, src.buf, dst.buf, 1, &region);
      if (i + 1 < reps)
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr,
                             0, nullptr);
    }
    if (qpool) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
    submit_wait();
    double secs = 0.0;
    if (qpool) read_timestamp_span(dev, qpool, ts_period_ns, secs);
    return secs;
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
  std::vector<std::uint32_t> chk(w);
  auto check_window = [&](std::uint32_t start) {
    begin_cb();
    VkBufferCopy region{static_cast<VkDeviceSize>(start) * 4, 0, static_cast<VkDeviceSize>(w) * 4};
    vkCmdCopyBuffer(cb, dst.buf, staging.buf, 1, &region);
    submit_wait();
    void* mapped = nullptr;
    vkMapMemory(dev, staging.mem, 0, static_cast<VkDeviceSize>(w) * 4, 0, &mapped);
    std::memcpy(chk.data(), mapped, static_cast<std::size_t>(w) * 4);
    vkUnmapMemory(dev, staging.mem);
    for (std::uint32_t i = 0; i < w && ok; ++i) {
      const std::uint32_t idx = start + i;
      if (chk[i] != vram::pattern(idx)) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "copy mismatch at u32[%u]: got=%u expected=%u", idx, chk[i],
                      vram::pattern(idx));
        verr = buf;
        ok = false;
      }
    }
  };
  check_window(0);
  if (ok && N > w) check_window(N - w);

  if (secs <= 0.0 && ok) {
    verr = "no usable device timestamp";
    ok = false;
  }

  cleanup();

  r.work = 2ull * static_cast<std::uint64_t>(reps) * S;
  r.measured = std::chrono::duration<double>{secs};
  r.timings.copy_h2d = r.measured;
  r.timings.copy_h2d_size = static_cast<std::size_t>(reps) * S;
  r.timings.total = wall1 - wall0;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  if (!ok) r.error = verr;
  return r;
}

}  // namespace bench
