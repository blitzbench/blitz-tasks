/**
 * @file vulkan_runner.cpp
 * @brief Vulkan device-local VRAM read runner. A DEVICE_LOCAL input buffer is prefilled
 *        (untimed) with the pattern via a HOST_VISIBLE staging buffer. The shared
 *        vram_read.comp (grid-stride uvec4 loads) accumulates per-thread and writes one
 *        uint per thread to a device out-buffer. The timed run records `reps` dispatches
 *        bracketed by timestamp queries. work = reps * S. Verified against the closed
 *        form (per-thread outputs copied out to staging and summed).
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
#include "vram_read.spv.inl"  // gpu_vram_read_shader::k_vram_read_spv_bytes / _len

namespace bench {

namespace {

struct PushConstants {
  std::uint32_t n_vec4;
};

}  // namespace

RunResult run_gpu_vram_read_vulkan(const gpgpu::Setup& setup) {
  using namespace gpu_vram_read_shader;

  RunResult r;
  r.path = "simd(uvec4 loads)";
  r.score_unit = "GB/s";

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_vram_read";
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
  const std::uint32_t n_vec4 = N / 4;
  const std::uint32_t groups = (vram::thread_count(setup.device) + vram::kBlock - 1) / vram::kBlock;
  const std::uint32_t T_launch = groups * vram::kBlock;
  const VkDeviceSize out_bytes = static_cast<VkDeviceSize>(T_launch) * 4;

  VkBufferAlloc in_buf{}, out_buf{}, staging{};
  bool alloc_ok =
      create_buffer(dev, mp, S, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, in_buf) &&
      create_buffer(dev, mp, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_buf) &&
      create_buffer(dev, mp, S, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);

  // Pipeline (2 storage bindings + push constants).
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = k_vram_read_spv_bytes_len;
  smci.pCode = reinterpret_cast<const std::uint32_t*>(k_vram_read_spv_bytes);
  VkShaderModule shader = VK_NULL_HANDLE;
  vkCreateShaderModule(dev, &smci, nullptr, &shader);

  VkDescriptorSetLayoutBinding binds[2]{};
  binds[0].binding = 0;
  binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binds[0].descriptorCount = 1;
  binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  binds[1].binding = 1;
  binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binds[1].descriptorCount = 1;
  binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 2;
  dslci.pBindings = binds;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl = VK_NULL_HANDLE;
  vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

  VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ssci.module = shader;
  ssci.pName = "main";
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = ssci;
  cpci.layout = pl;
  VkPipeline pipeline = VK_NULL_HANDLE;
  vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool dpool = VK_NULL_HANDLE;
  vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool);
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dpool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet dset = VK_NULL_HANDLE;
  vkAllocateDescriptorSets(dev, &dsai, &dset);
  VkDescriptorBufferInfo dbi_in{in_buf.buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo dbi_out{out_buf.buf, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet wr[2]{};
  wr[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wr[0].dstSet = dset;
  wr[0].dstBinding = 0;
  wr[0].descriptorCount = 1;
  wr[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr[0].pBufferInfo = &dbi_in;
  wr[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wr[1].dstSet = dset;
  wr[1].dstBinding = 1;
  wr[1].descriptorCount = 1;
  wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr[1].pBufferInfo = &dbi_out;
  vkUpdateDescriptorSets(dev, 2, wr, 0, nullptr);

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

  auto begin_cb = [&]() {
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
  };
  auto submit_wait = [&]() {
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
  };

  auto cleanup = [&]() {
    if (qpool) vkDestroyQueryPool(dev, qpool, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDescriptorPool(dev, dpool, nullptr);
    vkDestroyPipeline(dev, pipeline, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
    destroy_buffer(dev, in_buf);
    destroy_buffer(dev, out_buf);
    destroy_buffer(dev, staging);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
  };

  if (!alloc_ok) {
    r.error = "buffer alloc failed";
    cleanup();
    return r;
  }

  // --- prefill input buffer via staging (untimed) ---
  {
    void* mapped = nullptr;
    vkMapMemory(dev, staging.mem, 0, S, 0, &mapped);
    auto* p = static_cast<std::uint32_t*>(mapped);
    for (std::uint32_t i = 0; i < N; ++i) p[i] = vram::pattern(i);
    vkUnmapMemory(dev, staging.mem);
    begin_cb();
    VkBufferCopy region{0, 0, static_cast<VkDeviceSize>(S)};
    vkCmdCopyBuffer(cb, staging.buf, in_buf.buf, 1, &region);
    submit_wait();
  }

  auto time_reads = [&](int reps) -> double {
    begin_cb();
    if (qpool) {
      vkCmdResetQueryPool(cb, qpool, 0, 2);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
    PushConstants pc{n_vec4};
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    for (int i = 0; i < reps; ++i) {
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, groups, 1, 1);
      if (i + 1 < reps)
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                             0, nullptr, 0, nullptr);
    }
    if (qpool) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
    submit_wait();
    double secs = 0.0;
    if (qpool) read_timestamp_span(dev, qpool, ts_period_ns, secs);
    return secs;
  };

  for (int i = 0; i < kWarmups; ++i) time_reads(1);
  const double t_once = time_reads(1);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  const auto wall0 = std::chrono::steady_clock::now();
  const double secs = time_reads(reps);
  const auto wall1 = std::chrono::steady_clock::now();

  // --- verify: exact closed-form checksum (out_buf -> staging) ---
  bool ok = true;
  std::string verr;
  {
    begin_cb();
    VkBufferCopy region{0, 0, out_bytes};
    vkCmdCopyBuffer(cb, out_buf.buf, staging.buf, 1, &region);
    submit_wait();
    void* mapped = nullptr;
    vkMapMemory(dev, staging.mem, 0, out_bytes, 0, &mapped);
    const auto* p = static_cast<const std::uint32_t*>(mapped);
    std::uint32_t sum = 0u;
    for (std::uint32_t i = 0; i < T_launch; ++i) sum += p[i];
    vkUnmapMemory(dev, staging.mem);
    const std::uint32_t expected = vram::expected_sum(S);
    if (sum != expected) {
      char b[128];
      std::snprintf(b, sizeof(b), "read checksum mismatch: got=%u expected=%u", sum, expected);
      verr = b;
      ok = false;
    }
  }
  if (secs <= 0.0 && ok) {
    verr = "no usable device timestamp";
    ok = false;
  }

  cleanup();

  r.work = static_cast<std::uint64_t>(reps) * S;
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.timings.total = wall1 - wall0;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  if (!ok) r.error = verr;
  return r;
}

}  // namespace bench
