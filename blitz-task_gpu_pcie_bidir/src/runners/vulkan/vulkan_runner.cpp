/**
 * @file vulkan_runner.cpp
 * @brief Vulkan concurrent bidirectional transfer runner.
 *
 * Prefers two queues from different families so the two directions can run on
 * independent engines: a dedicated transfer-only family (TRANSFER without
 * COMPUTE/GRAPHICS) for one direction plus the compute/graphics family for the
 * other. If no dedicated transfer family exists (e.g. lavapipe, single-queue
 * parts) it falls back to submitting both command buffers to a single queue and
 * says so in `path`. Upload = HOST_VISIBLE staging -> DEVICE_LOCAL via
 * vkCmdCopyBuffer; download = DEVICE_LOCAL -> HOST_VISIBLE staging. Score is
 * aggregate wall throughput over submit -> vkQueueWaitIdle on both queues.
 * Per-direction timestamp spans feed the timings block only.
 */

#include "vulkan_runner.hpp"

#include <vulkan/vulkan.h>

#include <bench/config.hpp>
#include <bench/vk_utils.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../../bidir_params.hpp"

namespace bench {

namespace {

std::uint32_t timestamp_valid_bits(const gpgpu::vendor::VulkanInstanceFns& vki, VkPhysicalDevice pd, std::uint32_t fam) {
  std::uint32_t n = 0;
  vki.vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
  if (fam >= n) return 0;
  std::vector<VkQueueFamilyProperties> fams(n);
  vki.vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());
  return fams[fam].timestampValidBits;
}

}  // namespace

RunResult run_gpu_bidir_vulkan(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "single-queue (no concurrent engines)";
  r.score_unit = "GB/s";

  const gpgpu::vendor::VulkanFns* vk_fns = gpgpu::vendor::vulkan();
  if (!vk_fns) {
    r.error = "Vulkan loader unavailable";
    return r;
  }
  const gpgpu::vendor::VulkanFns& vk = *vk_fns;
  gpgpu::vendor::VulkanInstanceFns vki{};
  gpgpu::vendor::VulkanDeviceFns vkd{};

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_data_transfer_bidirectional";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst = VK_NULL_HANDLE;
  if (vk.vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
    r.error = "vkCreateInstance failed";
    return r;
  }
  if (!gpgpu::vendor::load_instance_fns(vk, inst, vki)) {
    if (vki.vkDestroyInstance) vki.vkDestroyInstance(inst, nullptr);
    r.error = "Vulkan instance entry points unavailable";
    return r;
  }

  VkPhysicalDevice pd = find_vk_device(vki, inst, setup.device);
  if (!pd) {
    vki.vkDestroyInstance(inst, nullptr);
    r.error = "no Vulkan device matched " + setup.device.id();
    return r;
  }

  VkPhysicalDeviceProperties pd_props{};
  vki.vkGetPhysicalDeviceProperties(pd, &pd_props);
  const float ts_period_ns = pd_props.limits.timestampPeriod;

  VkPhysicalDeviceMemoryProperties mp{};
  vki.vkGetPhysicalDeviceMemoryProperties(pd, &mp);

  // --- queue topology ---
  std::uint32_t famBase = find_queue_family(vki, pd, VK_QUEUE_COMPUTE_BIT);
  if (famBase == UINT32_MAX) famBase = find_queue_family(vki, pd, VK_QUEUE_GRAPHICS_BIT);
  if (famBase == UINT32_MAX) famBase = find_queue_family(vki, pd, VK_QUEUE_TRANSFER_BIT);
  if (famBase == UINT32_MAX) {
    vki.vkDestroyInstance(inst, nullptr);
    r.error = "no transfer-capable queue family";
    return r;
  }
  const std::uint32_t famXfer =
      find_queue_family_excluding(vki, pd, VK_QUEUE_TRANSFER_BIT, VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT);

  std::uint32_t famUp = famBase, famDn = famBase;
  bool two_queues = false;
  if (famXfer != UINT32_MAX && famXfer != famBase) {
    famUp = famXfer;  // dedicated transfer engine handles the upload
    famDn = famBase;  // compute/graphics engine handles the download
    two_queues = true;
    r.path = "2 queues (transfer + compute families)";
  }

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qcis[2]{};
  std::uint32_t n_qci = 1;
  qcis[0] = VkDeviceQueueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qcis[0].queueFamilyIndex = famBase;
  qcis[0].queueCount = 1;
  qcis[0].pQueuePriorities = &prio;
  if (two_queues) {
    qcis[1] = VkDeviceQueueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qcis[1].queueFamilyIndex = famXfer;
    qcis[1].queueCount = 1;
    qcis[1].pQueuePriorities = &prio;
    n_qci = 2;
  }
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = n_qci;
  dci.pQueueCreateInfos = qcis;
  VkDevice dev = VK_NULL_HANDLE;
  if (vki.vkCreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS) {
    vki.vkDestroyInstance(inst, nullptr);
    r.error = "vkCreateDevice failed";
    return r;
  }
  if (!gpgpu::vendor::load_device_fns(vki, dev, vkd)) {
    if (vkd.vkDestroyDevice) vkd.vkDestroyDevice(dev, nullptr);
    vki.vkDestroyInstance(inst, nullptr);
    r.error = "Vulkan device entry points unavailable";
    return r;
  }
  VkQueue queueUp = VK_NULL_HANDLE, queueDn = VK_NULL_HANDLE;
  vkd.vkGetDeviceQueue(dev, famUp, 0, &queueUp);
  vkd.vkGetDeviceQueue(dev, famDn, 0, &queueDn);

  const std::size_t S = bidir::buffer_bytes(setup.device);
  const std::size_t N = bidir::elem_count(S);
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(S);

  const VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VkBufferAlloc up_stage{}, up_dev{}, dn_dev{}, dn_stage{};
  bool alloc_ok = create_buffer(vkd, dev, mp, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_flags, up_stage) &&
                  create_buffer(vkd, dev, mp, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, up_dev) &&
                  create_buffer(vkd, dev, mp, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dn_dev) &&
                  create_buffer(vkd, dev, mp, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_flags, dn_stage);

  auto teardown = [&]() {
    destroy_buffer(vkd, dev, up_stage);
    destroy_buffer(vkd, dev, up_dev);
    destroy_buffer(vkd, dev, dn_dev);
    destroy_buffer(vkd, dev, dn_stage);
    vkd.vkDestroyDevice(dev, nullptr);
    vki.vkDestroyInstance(inst, nullptr);
  };
  if (!alloc_ok) {
    r.error = "buffer allocation failed";
    teardown();
    return r;
  }

  // Fill the upload staging buffer with the pattern.
  {
    void* mapped = nullptr;
    vkd.vkMapMemory(dev, up_stage.mem, 0, bytes, 0, &mapped);
    bidir::fill_pattern(static_cast<std::uint32_t*>(mapped), N);
    vkd.vkUnmapMemory(dev, up_stage.mem);
  }

  // Command pools: one per family actually used.
  auto make_pool = [&](std::uint32_t fam) {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.queueFamilyIndex = fam;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool p = VK_NULL_HANDLE;
    vkd.vkCreateCommandPool(dev, &ci, nullptr, &p);
    return p;
  };
  VkCommandPool poolUp = make_pool(famUp);
  VkCommandPool poolDn = (famDn == famUp) ? poolUp : make_pool(famDn);

  auto alloc_cb = [&](VkCommandPool pool) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkd.vkAllocateCommandBuffers(dev, &ai, &cb);
    return cb;
  };
  VkCommandBuffer cbUp = alloc_cb(poolUp);
  VkCommandBuffer cbDn = alloc_cb(poolDn);

  // Per-direction timestamp pools (best effort).
  const bool tsUp = timestamp_valid_bits(vki, pd, famUp) > 0 && ts_period_ns > 0.f;
  const bool tsDn = timestamp_valid_bits(vki, pd, famDn) > 0 && ts_period_ns > 0.f;
  VkQueryPool qpUp = tsUp ? create_timestamp_pool(vkd, dev, 2) : VK_NULL_HANDLE;
  VkQueryPool qpDn = tsDn ? create_timestamp_pool(vkd, dev, 2) : VK_NULL_HANDLE;

  auto record = [&](VkCommandBuffer cb, VkQueryPool qp, VkBuffer src, VkBuffer dst, int reps) {
    vkd.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkd.vkBeginCommandBuffer(cb, &bi);
    if (qp) {
      vkd.vkCmdResetQueryPool(cb, qp, 0, 2);
      vkd.vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    }
    VkBufferCopy region{0, 0, bytes};
    VkBufferMemoryBarrier mb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.buffer = dst;
    mb.offset = 0;
    mb.size = bytes;
    for (int i = 0; i < reps; ++i) {
      vkd.vkCmdCopyBuffer(cb, src, dst, 1, &region);
      if (i + 1 < reps)
        vkd.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &mb,
                                 0, nullptr);
    }
    if (qp) vkd.vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    vkd.vkEndCommandBuffer(cb);
  };

  auto submit = [&](VkQueue q, VkCommandBuffer cb) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkd.vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
  };

  // Seed the device download source with the pattern (pre-window, one-off).
  record(cbUp, VK_NULL_HANDLE, up_stage.buf, dn_dev.buf, 1);
  submit(queueUp, cbUp);
  vkd.vkQueueWaitIdle(queueUp);

  auto run_window = [&](int reps, double& h2d_secs, double& d2h_secs) -> double {
    record(cbUp, qpUp, up_stage.buf, up_dev.buf, reps);  // host -> device
    record(cbDn, qpDn, dn_dev.buf, dn_stage.buf, reps);  // device -> host
    const auto t0 = std::chrono::steady_clock::now();
    submit(queueUp, cbUp);
    submit(queueDn, cbDn);
    vkd.vkQueueWaitIdle(queueUp);
    if (queueDn != queueUp) vkd.vkQueueWaitIdle(queueDn);
    const auto t1 = std::chrono::steady_clock::now();
    h2d_secs = 0.0;
    d2h_secs = 0.0;
    if (qpUp) read_timestamp_span(vkd, dev, qpUp, ts_period_ns, h2d_secs);
    if (qpDn) read_timestamp_span(vkd, dev, qpDn, ts_period_ns, d2h_secs);
    return std::chrono::duration<double>(t1 - t0).count();
  };

  double dummy_h = 0.0, dummy_d = 0.0;
  for (int w = 0; w < kWarmups; ++w) run_window(1, dummy_h, dummy_d);
  const double t_once = run_window(1, dummy_h, dummy_d);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  double h2d_secs = 0.0, d2h_secs = 0.0;
  const double wall = run_window(reps, h2d_secs, d2h_secs);

  // --- verify both directions once, outside the timed window ---
  bool dn_ok = false, up_ok = false;
  {
    void* mapped = nullptr;
    vkd.vkMapMemory(dev, dn_stage.mem, 0, bytes, 0, &mapped);
    dn_ok = bidir::verify_sample(static_cast<std::uint32_t*>(mapped), N);
    vkd.vkUnmapMemory(dev, dn_stage.mem);
  }
  // Reuse dn_stage as a host-visible readback for the upload destination.
  {
    record(cbUp, VK_NULL_HANDLE, up_dev.buf, dn_stage.buf, 1);
    submit(queueUp, cbUp);
    vkd.vkQueueWaitIdle(queueUp);
    void* mapped = nullptr;
    vkd.vkMapMemory(dev, dn_stage.mem, 0, bytes, 0, &mapped);
    up_ok = bidir::verify_sample(static_cast<std::uint32_t*>(mapped), N);
    vkd.vkUnmapMemory(dev, dn_stage.mem);
  }
  const bool ok = up_ok && dn_ok;
  if (!ok) r.error = up_ok ? "download verification failed" : "upload verification failed";

  if (qpUp) vkd.vkDestroyQueryPool(dev, qpUp, nullptr);
  if (qpDn) vkd.vkDestroyQueryPool(dev, qpDn, nullptr);
  if (poolDn != poolUp) vkd.vkDestroyCommandPool(dev, poolDn, nullptr);
  vkd.vkDestroyCommandPool(dev, poolUp, nullptr);
  teardown();

  const std::uint64_t work = 2ull * static_cast<std::uint64_t>(reps) * S;
  r.work = work;
  r.measured = std::chrono::duration<double>{wall};
  r.score = score_giga(work, wall);
  r.correct = ok;
  r.timings.copy_h2d = std::chrono::duration<double>{h2d_secs};
  r.timings.copy_h2d_size = static_cast<std::size_t>(reps) * S;
  r.timings.copy_d2h = std::chrono::duration<double>{d2h_secs};
  r.timings.copy_d2h_size = static_cast<std::size_t>(reps) * S;
  r.timings.total = r.measured;
  return r;
}

}  // namespace bench
