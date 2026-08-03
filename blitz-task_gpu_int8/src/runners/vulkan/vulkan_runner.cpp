/**
 * @file vulkan_runner.cpp
 * @brief Vulkan compute int8 throughput runner — SDK-required variant.
 *
 * Tensor-first with a runtime fallback; the chosen path is recorded:
 *   VK_KHR_cooperative_matrix with a {s8,s8,s32,16,16,16,subgroup} config, the
 *   coopmat feature, shaderInt8, and the coopmat SPIR-V present at build
 *        -> int8_coopmat  path "tensor(coopmat s8->s32)"
 *   else shaderInt8 available
 *        -> int8_packed   path "simd(i8vec4)"
 *   else -> supported=false, "unsupported(no shaderInt8)"
 *
 * The coopmat kernel feeds all-ones s8 A,B into a 16x16x16 MMA, so every C
 * element equals 16*iters (bench::i8::reference_tensor). The packed kernel
 * matches bench::i8::reference_packed. Both write a uint32 storage buffer.
 * Unit GOPS. See kernel_params.hpp for math + ops accounting.
 *
 * CI NOTE: no cooperative-matrix hardware on the dev box (and glslang absent, so
 * this TU is configure-skipped locally). The coopmat device/feature plumbing is
 * validated on NVIDIA/AMD CI; on any gap the runner falls back to the packed
 * path.
 */

#include "vulkan_runner.hpp"

#include <vulkan/vulkan.h>

// Packed fallback always embedded; coopmat only when the build-time probe passed.
#include "int8_packed.spv.inl"  // gpu_int8_shader::k_int8_packed_spv_bytes / _len
#if defined(__has_include)
#if __has_include("int8_coopmat.spv.inl")
#include "int8_coopmat.spv.inl"  // gpu_int8_shader::k_int8_coopmat_spv_bytes / _len
#define GPU_INT8_COOPMAT_SPV_AVAILABLE 1
#endif
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
};

// Does the device expose a 16x16x16 {s8,s8,s32} subgroup cooperative-matrix?
#if defined(GPU_INT8_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
bool has_int8_coopmat_config(VkInstance inst, VkPhysicalDevice pd) {
  auto fn = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (!fn) return false;
  std::uint32_t n = 0;
  if (fn(pd, &n, nullptr) != VK_SUCCESS || n == 0) return false;
  std::vector<VkCooperativeMatrixPropertiesKHR> props(n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
  if (fn(pd, &n, props.data()) != VK_SUCCESS) return false;
  for (const auto& p : props) {
    if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16 && p.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
        p.BType == VK_COMPONENT_TYPE_SINT8_KHR && p.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
        p.ResultType == VK_COMPONENT_TYPE_SINT32_KHR && p.scope == VK_SCOPE_SUBGROUP_KHR)
      return true;
  }
  return false;
}
#endif

bool device_has_extension(VkPhysicalDevice pd, const char* name) {
  std::uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
  if (n == 0) return false;
  std::vector<VkExtensionProperties> ext(n);
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, ext.data());
  for (const auto& e : ext)
    if (std::strcmp(e.extensionName, name) == 0) return true;
  return false;
}

}  // namespace

RunResult run_gpu_int8_vulkan(const gpgpu::Setup& setup) {
  using namespace gpu_int8_shader;

  RunResult r;
  r.path = "simd(i8vec4)";
  r.score_unit = "GOPS";

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_int8";
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

  // --- feature probe: shaderInt8 (required by both int8 kernels) ---
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feat2.pNext = &f12;
  vkGetPhysicalDeviceFeatures2(pd, &feat2);
  const bool has_int8 = f12.shaderInt8 == VK_TRUE;

  if (!has_int8) {
    vkDestroyInstance(inst, nullptr);
    r.supported = false;
    r.path = "unsupported(no shaderInt8)";
    return r;
  }

  // --- decide path ---
  bool use_coopmat = false;
#if defined(GPU_INT8_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
  if (device_has_extension(pd, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && has_int8_coopmat_config(inst, pd)) {
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    q.pNext = &cmf;
    vkGetPhysicalDeviceFeatures2(pd, &q);
    use_coopmat = (cmf.cooperativeMatrix == VK_TRUE);
  }
#endif

  const std::uint32_t local_x = use_coopmat ? 32u : i8::kBlock;
  const bool tensor = use_coopmat;
  const unsigned char* spirv = k_int8_packed_spv_bytes;
  std::size_t spirv_len = k_int8_packed_spv_bytes_len;
  r.path = "simd(i8vec4)";
#if defined(GPU_INT8_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
  if (use_coopmat) {
    spirv = k_int8_coopmat_spv_bytes;
    spirv_len = k_int8_coopmat_spv_bytes_len;
    r.path = "tensor(coopmat s8->s32)";
  }
#endif

  const std::uint32_t qfam = find_queue_family(pd, VK_QUEUE_COMPUTE_BIT);
  if (qfam == UINT32_MAX) {
    vkDestroyInstance(inst, nullptr);
    r.error = "no compute queue family";
    return r;
  }

  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);

  // --- device with the required feature chain ---
  VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  en12.shaderInt8 = VK_TRUE;
  void* chain = &en12;

#if defined(GPU_INT8_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR en_cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceVulkanMemoryModelFeatures en_vmm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
  if (use_coopmat) {
    en_cm.cooperativeMatrix = VK_TRUE;
    en_vmm.vulkanMemoryModel = VK_TRUE;
    en12.pNext = &en_cm;
    en_cm.pNext = &en_vmm;
  }
#endif

  std::vector<const char*> dev_exts;
#if defined(GPU_INT8_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
  if (use_coopmat) dev_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#endif

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = chain;
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

  const std::uint32_t T = i8::thread_count(setup.device);
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(T) * sizeof(std::uint32_t);

  VkBufferAlloc out{};
  if (!create_buffer(dev, mp, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out)) {
    destroy_buffer(dev, out);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    r.error = "buffer alloc failed";
    return r;
  }

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spirv_len;
  smci.pCode = reinterpret_cast<const std::uint32_t*>(spirv);
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
  // Pin the subgroup size for the coopmat kernel (assumes size 32 is offered).
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rssci{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
  rssci.requiredSubgroupSize = 32;
  if (tensor) ssci.pNext = &rssci;
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = ssci;
  cpci.layout = pl;
  VkPipeline pipeline = VK_NULL_HANDLE;
  vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);

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

  auto time_launches = [&](std::uint32_t n, std::uint32_t iters, int reps) -> double {
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
    PushConstants pc{n, iters};
    const std::uint32_t groups = (n + local_x - 1) / local_x;
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

  auto read_back = [&](std::uint32_t count, std::vector<std::uint32_t>& host) {
    void* mapped = nullptr;
    vkMapMemory(dev, out.mem, 0, count * sizeof(std::uint32_t), 0, &mapped);
    std::memcpy(host.data(), mapped, count * sizeof(std::uint32_t));
    vkUnmapMemory(dev, out.mem);
  };

  std::vector<std::uint32_t> host(T);
  bool ok = true;

  auto verify = [&](std::uint32_t count, std::uint32_t iters, std::uint32_t stride) -> bool {
    for (std::uint32_t s = 0; s < count; ++s) {
      const std::uint32_t t = s * stride;
      const std::uint32_t exp = tensor ? (std::uint32_t)i8::reference_tensor(iters) : i8::reference_packed(t, iters);
      if (host[t] != exp) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "mismatch at t=%u: got=%u expected=%u", t, host[t], exp);
        r.error = buf;
        return false;
      }
    }
    return true;
  };

  // --- pre-flight exactness ---
  time_launches(i8::kPreflightThreads, i8::kPreflightIters, 1);
  read_back(i8::kPreflightThreads, host);
  ok = verify(i8::kPreflightThreads, i8::kPreflightIters, 1);

  double secs = 0.0;
  int reps = 0;
  if (ok) {
    for (int w = 0; w < kWarmups; ++w) time_launches(T, i8::kIters, 1);
    const double t_once = time_launches(T, i8::kIters, 1);
    reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
    secs = time_launches(T, i8::kIters, reps);
    read_back(T, host);
    constexpr std::uint32_t kSamples = 64;
    const std::uint32_t stride = (T - 1) / (kSamples - 1);
    ok = verify(kSamples, i8::kIters, stride);
  }

  if (qpool) vkDestroyQueryPool(dev, qpool, nullptr);
  vkDestroyCommandPool(dev, cpool, nullptr);
  vkDestroyDescriptorPool(dev, dpool, nullptr);
  vkDestroyPipeline(dev, pipeline, nullptr);
  vkDestroyPipelineLayout(dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  vkDestroyShaderModule(dev, shader, nullptr);
  destroy_buffer(dev, out);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);

  if (ok) {
    const std::uint32_t tiles = T / local_x;
    r.work = tensor ? i8::ops_tensor(tiles, i8::kIters, reps) : i8::ops_packed(T, i8::kIters, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = true;
  }
  return r;
}

}  // namespace bench
