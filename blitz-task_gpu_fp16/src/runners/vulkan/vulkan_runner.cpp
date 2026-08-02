/**
 * @file vulkan_runner.cpp
 * @brief Vulkan compute fp16 throughput runner — cooperative-matrix tensor path with a
 *        packed-SIMD fallback.
 *
 * At configure time the shaders/ build embeds fp16_coopmat.spv only when glslang
 * can compile KHR_cooperative_matrix (GPU_FP16_COOPMAT_SPV_AVAILABLE). At runtime
 * we additionally require the device to expose VK_KHR_cooperative_matrix and a
 * 16x16x16 {f16,f16,f32} subgroup configuration; otherwise we run the packed
 * f16vec4 kernel. RunResult.path records which ran.
 *
 *   tensor:  "tensor(coopmat 16x16x16 f16->f32)"
 *   packed:  "simd(f16vec4 fma)"
 *
 * The device is created (when using coopmat) with CooperativeMatrix +
 * VulkanMemoryModel + shaderFloat16 enabled and API version 1.3 requested.
 */

#include "vulkan_runner.hpp"

#include <vulkan/vulkan.h>

#include "fp16_packed.spv.inl"  // gpu_fp16_shader::k_fp16_packed_spv_bytes / _len
#if defined(GPU_FP16_COOPMAT_SPV_AVAILABLE)
#include "fp16_coopmat.spv.inl"  // gpu_fp16_shader::k_fp16_coopmat_spv_bytes / _len
#endif

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

namespace {

struct PushConstants {
  std::uint32_t n;
  std::uint32_t iters;
  float mf;
  float af;
};

bool device_has_extension(VkPhysicalDevice pd, const char* name) {
  std::uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> exts(n);
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, exts.data());
  for (const auto& e : exts)
    if (std::strcmp(e.extensionName, name) == 0) return true;
  return false;
}

/**
 * @brief Query whether a 16x16x16 {f16,f16,f32} subgroup coopmat config exists.
 * @param inst
 * @param pd
 * @return
 */
bool coopmat_16x16x16_supported(VkInstance inst, VkPhysicalDevice pd) {
#if defined(VK_KHR_cooperative_matrix)
  auto pfn = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (!pfn) return false;
  std::uint32_t n = 0;
  if (pfn(pd, &n, nullptr) != VK_SUCCESS || n == 0) return false;
  std::vector<VkCooperativeMatrixPropertiesKHR> props(n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
  if (pfn(pd, &n, props.data()) != VK_SUCCESS) return false;
  for (const auto& p : props) {
    if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16 && p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
        p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.scope == VK_SCOPE_SUBGROUP_KHR)
      return true;
  }
#else
  (void)inst;
  (void)pd;
#endif
  return false;
}

std::uint32_t subgroup_size(VkPhysicalDevice pd) {
  VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &sg;
  vkGetPhysicalDeviceProperties2(pd, &p2);
  return sg.subgroupSize ? sg.subgroupSize : 32u;
}

}  // namespace

RunResult run_gpu_fp16_vulkan(const gpgpu::Setup& setup) {
  using namespace gpu_fp16_shader;

  RunResult r;
  r.score_unit = "GFLOPS";
  r.path = "simd(f16vec4 fma)";

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_fp16";
  app.apiVersion = VK_API_VERSION_1_3;
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
  const std::uint32_t sg = subgroup_size(pd);

  // Decide path.
  bool use_coopmat = false;
#if defined(GPU_FP16_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
  use_coopmat =
      device_has_extension(pd, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && coopmat_16x16x16_supported(inst, pd);
#endif
  r.path = use_coopmat ? "tensor(coopmat 16x16x16 f16->f32)" : "simd(f16vec4 fma)";

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

  // Feature chain: shaderFloat16 + vulkanMemoryModel (+ cooperativeMatrix).
  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  v12.shaderFloat16 = VK_TRUE;
  v12.vulkanMemoryModel = VK_TRUE;
  v12.vulkanMemoryModelDeviceScope = VK_TRUE;
  VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feat2.pNext = &v12;

  std::vector<const char*> dev_exts;
#if defined(VK_KHR_cooperative_matrix)
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  if (use_coopmat) {
    coopFeat.cooperativeMatrix = VK_TRUE;
    v12.pNext = &coopFeat;
    dev_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  }
#endif

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &feat2;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = static_cast<std::uint32_t>(dev_exts.size());
  dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
  VkDevice dev = VK_NULL_HANDLE;
  if (vkCreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS) {
    vkDestroyInstance(inst, nullptr);
    r.error = "vkCreateDevice failed";
    return r;
  }
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(dev, qfam, 0, &queue);

  const std::uint32_t T = fp16::thread_count(setup.device);
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(T) * sizeof(float);

  VkBufferAlloc out{};
  if (!create_buffer(dev, mp, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out)) {
    destroy_buffer(dev, out);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    r.error = "buffer alloc failed";
    return r;
  }

  // Choose SPIR-V.
  const unsigned char* spv = k_fp16_packed_spv_bytes;
  unsigned int spv_len = k_fp16_packed_spv_bytes_len;
#if defined(GPU_FP16_COOPMAT_SPV_AVAILABLE)
  if (use_coopmat) {
    spv = k_fp16_coopmat_spv_bytes;
    spv_len = k_fp16_coopmat_spv_bytes_len;
  }
#endif

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv_len;
  smci.pCode = reinterpret_cast<const std::uint32_t*>(spv);
  VkShaderModule shader = VK_NULL_HANDLE;
  vkCreateShaderModule(dev, &smci, nullptr, &shader);

  VkDescriptorSetLayoutBinding bind{};
  bind.binding = 0;
  bind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bind.descriptorCount = 1;
  bind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 1;
  dslci.pBindings = &bind;
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
  const bool pipeline_ok = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) == VK_SUCCESS;

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
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
  VkDescriptorBufferInfo dbi{out.buf, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wr.dstSet = dset;
  wr.dstBinding = 0;
  wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr.pBufferInfo = &dbi;
  vkUpdateDescriptorSets(dev, 1, &wr, 0, nullptr);

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

  auto time_launches = [&](std::uint32_t n, std::uint32_t iters, float mf, float af, int reps) -> double {
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    if (qpool) {
      vkCmdResetQueryPool(cb, qpool, 0, 2);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
    PushConstants pc{n, iters, mf, af};
    const std::uint32_t groups = (T + fp16::kBlock - 1) / fp16::kBlock;
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
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    double secs = 0.0;
    if (qpool) read_timestamp_span(dev, qpool, ts_period_ns, secs);
    return secs;
  };

  auto read_back = [&](std::uint32_t count, std::vector<float>& host) {
    void* mapped = nullptr;
    vkMapMemory(dev, out.mem, 0, count * sizeof(float), 0, &mapped);
    std::memcpy(host.data(), mapped, count * sizeof(float));
    vkUnmapMemory(dev, out.mem);
  };

  std::vector<float> host(T);
  bool ok = pipeline_ok;
  if (!pipeline_ok) r.error = "vkCreateComputePipelines failed";
  double secs = 0.0;
  int reps = 0;

  const std::uint32_t iters = use_coopmat ? fp16::kTensorIters : fp16::kIters;
  const std::uint32_t warps = T / (sg ? sg : 32u);

  // --- pre-flight ---
  if (ok) {
    time_launches(fp16::kPreflightThreads, use_coopmat ? fp16::kTensorPreflightIters : fp16::kPreflightIters,
                  fp16::kPreM, fp16::kPreA, 1);
    read_back(fp16::kPreflightThreads, host);
    for (std::uint32_t t = 0; t < fp16::kPreflightThreads && ok; ++t) {
      if (use_coopmat) {
        if (!fp16::tensor_ok(host[t], fp16::kTensorPreflightIters, 16)) {
          char b[160];
          std::snprintf(b, sizeof(b), "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t],
                        fp16::tensor_out(fp16::kTensorPreflightIters, 16));
          r.error = b;
          ok = false;
        }
      } else {
        const double e = fp16::simd_preflight_out(t, 4, fp16::kPreflightIters);
        if (!fp16::matches_exact(host[t], e)) {
          char b[160];
          std::snprintf(b, sizeof(b), "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t], e);
          r.error = b;
          ok = false;
        }
      }
    }
  }

  // --- warmup + calibrate + timed run ---
  if (ok) {
    for (int w = 0; w < kWarmups; ++w) time_launches(T, iters, fp16::kMainM, fp16::kMainA, 1);
    const double t_once = time_launches(T, iters, fp16::kMainM, fp16::kMainA, 1);
    reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
    secs = time_launches(T, iters, fp16::kMainM, fp16::kMainA, reps);
    read_back(T, host);
    for (std::uint32_t s = 0; s < 64 && ok; ++s) {
      const std::uint32_t t = static_cast<std::uint32_t>(static_cast<std::uint64_t>(s) * (T - 1) / 63);
      if (use_coopmat) {
        if (!fp16::tensor_ok(host[t], iters, 16)) {
          char b[160];
          std::snprintf(b, sizeof(b), "sample mismatch at t=%u: got=%g expected=%g", t, host[t],
                        fp16::tensor_out(iters, 16));
          r.error = b;
          ok = false;
        }
      } else if (!fp16::simd_main_ok(host[t], 4)) {
        char b[160];
        std::snprintf(b, sizeof(b), "sample out of envelope at t=%u: got=%g", t, host[t]);
        r.error = b;
        ok = false;
      }
    }
  }

  if (qpool) vkDestroyQueryPool(dev, qpool, nullptr);
  vkDestroyCommandPool(dev, cpool, nullptr);
  vkDestroyDescriptorPool(dev, dpool, nullptr);
  if (pipeline) vkDestroyPipeline(dev, pipeline, nullptr);
  vkDestroyPipelineLayout(dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  if (shader) vkDestroyShaderModule(dev, shader, nullptr);
  destroy_buffer(dev, out);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);

  if (ok) {
    r.work = use_coopmat ? fp16::tensor_flops(warps, iters, reps, 16, 16, 16) : fp16::simd_flops(T, iters, reps, 4);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = true;
  }
  return r;
}

}  // namespace bench
