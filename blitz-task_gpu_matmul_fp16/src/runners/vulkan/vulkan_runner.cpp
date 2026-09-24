/**
 * @file vulkan_runner.cpp
 * @brief Vulkan fp16 GEMM runner — cooperative-matrix path with a shared-memory tiled
 *        fallback.
 *
 * Vulkan has no vendor BLAS to call, so this runner is what the task reports wherever
 * cuBLASLt, hipBLASLt and MPS are all out of reach. At configure time the shaders/ build
 * embeds gemm_coopmat.spv only when glslang can compile KHR_cooperative_matrix
 * (GPU_MATMUL_FP16_COOPMAT_SPV_AVAILABLE). At runtime we additionally require the device
 * to expose VK_KHR_cooperative_matrix and a 16x16x16 {f16,f16,f32} subgroup
 * configuration; otherwise the tiled kernel runs. RunResult.path records which:
 *
 *   coopmat: "tiled(coopmat 16x16x16 f16->f32)"
 *   fallback: "tiled(shared 16x16 f16->f32)"
 *
 * Operands never cross the bus: gemm_fill.comp writes A and B from the coordinate hash
 * once, when the context is built. Only kVerifyRows rows of C come back, and the
 * sampled elements are checked against bench::gemm::reference_element.
 *
 * The device is created with shaderFloat16 + storageBuffer16BitAccess (+
 * cooperativeMatrix) and API version 1.3 requested.
 */

#include "vulkan_runner.hpp"

#include <vulkan/vulkan.h>

#include "gemm_fill.spv.inl"   // gpu_matmul_fp16_shader::k_gemm_fill_spv_bytes / _len
#include "gemm_tiled.spv.inl"  // gpu_matmul_fp16_shader::k_gemm_tiled_spv_bytes / _len
#if defined(GPU_MATMUL_FP16_COOPMAT_SPV_AVAILABLE)
#include "gemm_coopmat.spv.inl"  // gpu_matmul_fp16_shader::k_gemm_coopmat_spv_bytes / _len
#endif

#include <bench/config.hpp>
#include <bench/gemm/params.hpp>
#include <bench/vk_utils.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace bench {

namespace {

// Output tile edge, and therefore the workgroup footprint of every kernel here.
constexpr std::uint32_t kTile = 16u;

struct PushConstants {
  std::uint32_t n;
  std::uint32_t seed;
};

bool device_has_extension(const gpgpu::vendor::VulkanInstanceFns& vki, VkPhysicalDevice pd, const char* name) {
  std::uint32_t n = 0;
  vki.vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> exts(n);
  vki.vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, exts.data());
  for (const auto& e : exts)
    if (std::strcmp(e.extensionName, name) == 0) return true;
  return false;
}

/**
 * @brief Query whether a 16x16x16 {f16,f16,f32} subgroup coopmat config exists.
 * @param vk
 * @param inst
 * @param pd
 * @return
 */
bool coopmat_16x16x16_supported(const gpgpu::vendor::VulkanFns& vk, VkInstance inst, VkPhysicalDevice pd) {
#if defined(VK_KHR_cooperative_matrix)
  auto pfn = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      vk.vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
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
  (void)vk;
  (void)inst;
  (void)pd;
#endif
  return false;
}

std::uint32_t subgroup_size(const gpgpu::vendor::VulkanInstanceFns& vki, VkPhysicalDevice pd) {
  VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &sg;
  vki.vkGetPhysicalDeviceProperties2(pd, &p2);
  return sg.subgroupSize ? sg.subgroupSize : 32u;
}

// Whether the device can hold and multiply fp16 values in a storage buffer at all.
bool supports_fp16_storage(const gpgpu::vendor::VulkanInstanceFns& vki, VkPhysicalDevice pd) {
  VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  v11.pNext = &v12;
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  f2.pNext = &v11;
  vki.vkGetPhysicalDeviceFeatures2(pd, &f2);
  return v11.storageBuffer16BitAccess == VK_TRUE && v12.shaderFloat16 == VK_TRUE;
}

/**
 * @class VulkanGemmContext
 * @brief Instance, device, operand buffers and pipelines for one (device, size) pair.
 */
struct VulkanGemmContext : gemm::Context {
  std::string device_id;
  std::uint32_t n{0};
  std::uint32_t seed{0};
  bool coopmat{false};

  gpgpu::vendor::VulkanInstanceFns vki{};
  gpgpu::vendor::VulkanDeviceFns vkd{};
  VkInstance inst{VK_NULL_HANDLE};
  VkDevice dev{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  std::uint32_t qfam{0};
  float ts_period_ns{0.0f};
  bool ts_ok{false};

  VkBufferAlloc a{}, b{}, c{}, staging{};
  VkShaderModule fill_shader{VK_NULL_HANDLE}, gemm_shader{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkPipeline fill_pipe{VK_NULL_HANDLE}, gemm_pipe{VK_NULL_HANDLE};
  VkDescriptorPool dpool{VK_NULL_HANDLE};
  VkDescriptorSet dset{VK_NULL_HANDLE};
  VkQueryPool qpool{VK_NULL_HANDLE};
  VkCommandPool cpool{VK_NULL_HANDLE};
  VkCommandBuffer cb{VK_NULL_HANDLE};

  ~VulkanGemmContext() override { destroy(); }

  void destroy() {
    if (dev) {
      if (qpool) vkd.vkDestroyQueryPool(dev, qpool, nullptr);
      if (cpool) vkd.vkDestroyCommandPool(dev, cpool, nullptr);
      if (dpool) vkd.vkDestroyDescriptorPool(dev, dpool, nullptr);
      if (fill_pipe) vkd.vkDestroyPipeline(dev, fill_pipe, nullptr);
      if (gemm_pipe) vkd.vkDestroyPipeline(dev, gemm_pipe, nullptr);
      if (layout) vkd.vkDestroyPipelineLayout(dev, layout, nullptr);
      if (dsl) vkd.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
      if (fill_shader) vkd.vkDestroyShaderModule(dev, fill_shader, nullptr);
      if (gemm_shader) vkd.vkDestroyShaderModule(dev, gemm_shader, nullptr);
      destroy_buffer(vkd, dev, a);
      destroy_buffer(vkd, dev, b);
      destroy_buffer(vkd, dev, c);
      destroy_buffer(vkd, dev, staging);
      vkd.vkDestroyDevice(dev, nullptr);
      dev = VK_NULL_HANDLE;
    }
    if (inst) {
      vki.vkDestroyInstance(inst, nullptr);
      inst = VK_NULL_HANDLE;
    }
  }

  /**
   * @brief Record and submit @p reps multiplies, returning the device-side seconds.
   *
   * @param reps
   * @return 0 when the device cannot report timestamps.
   */
  double time_gemm(int reps) {
    vkd.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkd.vkBeginCommandBuffer(cb, &bi);
    if (qpool) {
      vkd.vkCmdResetQueryPool(cb, qpool, 0, 2);
      vkd.vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    }
    vkd.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gemm_pipe);
    vkd.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
    const PushConstants pc{n, seed};
    vkd.vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    const std::uint32_t groups = n / kTile;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    for (int i = 0; i < reps; ++i) {
      vkd.vkCmdDispatch(cb, groups, groups, 1);
      if (i + 1 < reps)
        vkd.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                                 0, nullptr, 0, nullptr);
    }
    if (qpool) vkd.vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
    vkd.vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkd.vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkd.vkQueueWaitIdle(queue);
    double secs = 0.0;
    if (qpool) read_timestamp_span(vkd, dev, qpool, ts_period_ns, secs);
    return secs;
  }

  // Run the fill kernel once so A and B hold their hashed operands.
  void fill_operands() {
    vkd.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkd.vkBeginCommandBuffer(cb, &bi);
    vkd.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipe);
    vkd.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
    const PushConstants pc{n, seed};
    vkd.vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    const std::uint32_t groups = n / kTile;
    vkd.vkCmdDispatch(cb, groups, groups, 1);
    vkd.vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkd.vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkd.vkQueueWaitIdle(queue);
  }

  /**
   * @brief Copy the first gemm::kVerifyRows rows of C into @p host.
   *
   * @param host
   */
  void read_verify_rows(std::vector<float>& host) {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(gemm::kVerifyRows) * n * sizeof(float);
    vkd.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkd.vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy region{0, 0, bytes};
    vkd.vkCmdCopyBuffer(cb, c.buf, staging.buf, 1, &region);
    vkd.vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkd.vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkd.vkQueueWaitIdle(queue);

    void* mapped = nullptr;
    vkd.vkMapMemory(dev, staging.mem, 0, bytes, 0, &mapped);
    std::memcpy(host.data(), mapped, static_cast<std::size_t>(bytes));
    vkd.vkUnmapMemory(dev, staging.mem);
  }
};

/**
 * @brief Build a context for @p setup at problem size @p n.
 *
 * @param vk
 * @param setup
 * @param n
 * @param seed
 * @param out
 * @param error
 * @return false with @p error populated on any failure; @p out is then unusable.
 */
bool build_context(const gpgpu::vendor::VulkanFns& vk, const gpgpu::Setup& setup, std::uint32_t n,
                   std::uint32_t seed, VulkanGemmContext& out, std::string& error) {
  using namespace gpu_matmul_fp16_shader;
  gpgpu::vendor::VulkanInstanceFns& vki = out.vki;
  gpgpu::vendor::VulkanDeviceFns& vkd = out.vkd;

  out.device_id = setup.device.id();
  out.n = n;
  out.seed = seed;

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_matmul_fp16";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  if (vk.vkCreateInstance(&ici, nullptr, &out.inst) != VK_SUCCESS) {
    error = "vkCreateInstance failed";
    return false;
  }
  if (!gpgpu::vendor::load_instance_fns(vk, out.inst, vki)) {
    if (vki.vkDestroyInstance) vki.vkDestroyInstance(out.inst, nullptr);
    out.inst = VK_NULL_HANDLE;
    error = "Vulkan instance entry points unavailable";
    return false;
  }

  VkPhysicalDevice pd = find_vk_device(vki, out.inst, setup.device);
  if (!pd) {
    error = "no Vulkan device matched " + setup.device.id();
    return false;
  }
  if (!supports_fp16_storage(vki, pd)) {
    error = "device lacks shaderFloat16 / storageBuffer16BitAccess";
    return false;
  }

  VkPhysicalDeviceProperties pd_props{};
  vki.vkGetPhysicalDeviceProperties(pd, &pd_props);
  out.ts_period_ns = pd_props.limits.timestampPeriod;
  out.ts_ok = pd_props.limits.timestampComputeAndGraphics != 0;
  const std::uint32_t sg = subgroup_size(vki, pd);

#if defined(GPU_MATMUL_FP16_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
  out.coopmat = device_has_extension(vki, pd, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) &&
                coopmat_16x16x16_supported(vk, out.inst, pd);
#endif

  out.qfam = find_queue_family(vki, pd, VK_QUEUE_COMPUTE_BIT);
  if (out.qfam == UINT32_MAX) {
    error = "no compute queue family";
    return false;
  }

  VkPhysicalDeviceMemoryProperties mp{};
  vki.vkGetPhysicalDeviceMemoryProperties(pd, &mp);

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = out.qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  v11.storageBuffer16BitAccess = VK_TRUE;
  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  v12.shaderFloat16 = VK_TRUE;
  v12.vulkanMemoryModel = VK_TRUE;
  v12.vulkanMemoryModelDeviceScope = VK_TRUE;
  v11.pNext = &v12;
  VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feat2.pNext = &v11;

  std::vector<const char*> dev_exts;
#if defined(VK_KHR_cooperative_matrix)
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_feat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  if (out.coopmat) {
    coop_feat.cooperativeMatrix = VK_TRUE;
    v12.pNext = &coop_feat;
    dev_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  }
#endif

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &feat2;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = static_cast<std::uint32_t>(dev_exts.size());
  dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
  if (vki.vkCreateDevice(pd, &dci, nullptr, &out.dev) != VK_SUCCESS) {
    error = "vkCreateDevice failed";
    return false;
  }
  if (!gpgpu::vendor::load_device_fns(vki, out.dev, vkd)) {
    if (vkd.vkDestroyDevice) vkd.vkDestroyDevice(out.dev, nullptr);
    out.dev = VK_NULL_HANDLE;
    error = "Vulkan device entry points unavailable";
    return false;
  }
  vkd.vkGetDeviceQueue(out.dev, out.qfam, 0, &out.queue);

  const VkDeviceSize elems = static_cast<VkDeviceSize>(n) * n;
  const VkDeviceSize verify_bytes = static_cast<VkDeviceSize>(gemm::kVerifyRows) * n * sizeof(float);
  const VkMemoryPropertyFlags local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkMemoryPropertyFlags host_vis =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (!create_buffer(vkd, out.dev, mp, elems * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, local, out.a) ||
      !create_buffer(vkd, out.dev, mp, elems * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, local, out.b) ||
      !create_buffer(vkd, out.dev, mp, elems * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     local, out.c) ||
      !create_buffer(vkd, out.dev, mp, verify_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_vis, out.staging)) {
    error = "operand allocation failed at n=" + std::to_string(n);
    return false;
  }

  auto make_module = [&](const unsigned char* spv, unsigned int len) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = len;
    smci.pCode = reinterpret_cast<const std::uint32_t*>(spv);
    VkShaderModule m = VK_NULL_HANDLE;
    vkd.vkCreateShaderModule(out.dev, &smci, nullptr, &m);
    return m;
  };
  out.fill_shader = make_module(k_gemm_fill_spv_bytes, k_gemm_fill_spv_bytes_len);
#if defined(GPU_MATMUL_FP16_COOPMAT_SPV_AVAILABLE)
  out.gemm_shader = out.coopmat ? make_module(k_gemm_coopmat_spv_bytes, k_gemm_coopmat_spv_bytes_len)
                                : make_module(k_gemm_tiled_spv_bytes, k_gemm_tiled_spv_bytes_len);
#else
  out.gemm_shader = make_module(k_gemm_tiled_spv_bytes, k_gemm_tiled_spv_bytes_len);
#endif
  if (!out.fill_shader || !out.gemm_shader) {
    error = "vkCreateShaderModule failed";
    return false;
  }

  VkDescriptorSetLayoutBinding binds[3]{};
  for (std::uint32_t i = 0; i < 3; ++i) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 3;
  dslci.pBindings = binds;
  vkd.vkCreateDescriptorSetLayout(out.dev, &dslci, nullptr, &out.dsl);

  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &out.dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  vkd.vkCreatePipelineLayout(out.dev, &plci, nullptr, &out.layout);

  // The coopmat kernel takes its workgroup width from the device's subgroup size, so one
  // workgroup is exactly one subgroup on every vendor.
  const VkSpecializationMapEntry sme{0, 0, sizeof(std::uint32_t)};
  VkSpecializationInfo spec{};
  spec.mapEntryCount = 1;
  spec.pMapEntries = &sme;
  spec.dataSize = sizeof(std::uint32_t);
  spec.pData = &sg;

  auto make_pipeline = [&](VkShaderModule module, bool specialized) {
    VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = module;
    ssci.pName = "main";
    if (specialized) ssci.pSpecializationInfo = &spec;
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci;
    cpci.layout = out.layout;
    VkPipeline p = VK_NULL_HANDLE;
    vkd.vkCreateComputePipelines(out.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p);
    return p;
  };
  out.fill_pipe = make_pipeline(out.fill_shader, false);
  out.gemm_pipe = make_pipeline(out.gemm_shader, out.coopmat);
  if (!out.fill_pipe || !out.gemm_pipe) {
    error = "vkCreateComputePipelines failed";
    return false;
  }

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  vkd.vkCreateDescriptorPool(out.dev, &dpci, nullptr, &out.dpool);
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = out.dpool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &out.dsl;
  vkd.vkAllocateDescriptorSets(out.dev, &dsai, &out.dset);

  const VkDescriptorBufferInfo dbi[3] = {
      {out.a.buf, 0, VK_WHOLE_SIZE}, {out.b.buf, 0, VK_WHOLE_SIZE}, {out.c.buf, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet writes[3]{};
  for (std::uint32_t i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = out.dset;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &dbi[i];
  }
  vkd.vkUpdateDescriptorSets(out.dev, 3, writes, 0, nullptr);

  out.qpool = out.ts_ok ? create_timestamp_pool(vkd, out.dev, 2) : VK_NULL_HANDLE;

  VkCommandPoolCreateInfo cpoolci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpoolci.queueFamilyIndex = out.qfam;
  cpoolci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkd.vkCreateCommandPool(out.dev, &cpoolci, nullptr, &out.cpool);
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = out.cpool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  vkd.vkAllocateCommandBuffers(out.dev, &cbai, &out.cb);

  out.fill_operands();
  return true;
}

}  // namespace

RunResult run_gpu_matmul_fp16_vulkan(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                     const gemm::RunParams& params) {
  RunResult r;
  r.score_unit = "GFLOPS";
  r.path = "unsupported(fp16 gemm)";

  const gpgpu::vendor::VulkanFns* vk_fns = gpgpu::vendor::vulkan();
  if (!vk_fns) {
    r.error = "Vulkan loader unavailable";
    return r;
  }
  const gpgpu::vendor::VulkanFns& vk = *vk_fns;

  auto* gemm_ctx = dynamic_cast<VulkanGemmContext*>(ctx.get());
  if (gemm_ctx && (gemm_ctx->device_id != setup.device.id() || gemm_ctx->seed != params.seed)) gemm_ctx = nullptr;

  // Size the problem on the first round; later rounds reuse what this settled on. The
  // bottom rung is timed first and extrapolated, so a slow device is never asked to run
  // a multiply it cannot finish, and the ladder steps down again if allocation fails.
  if (!gemm_ctx) {
    const std::uint32_t memory_limit_n = gemm::choose_size(gemm::Precision::Fp16, setup.device, params.cap_bytes);
    std::string error;

    auto build_at = [&](std::uint32_t n) -> std::unique_ptr<VulkanGemmContext> {
      while (n) {
        auto fresh = std::make_unique<VulkanGemmContext>();
        if (build_context(vk, setup, n, params.seed, *fresh, error)) return fresh;
        n = gemm::step_down(n);
      }
      return nullptr;
    };

    auto probe_ctx = build_at(gemm::kLadder[0]);
    if (!probe_ctx) {
      r.error = error;
      return r;
    }
    const double t_probe = probe_ctx->time_gemm(1);
    const std::uint32_t target = gemm::largest_within_time(gemm::kLadder[0], t_probe, memory_limit_n);

    if (target > gemm::kLadder[0]) {
      probe_ctx.reset();
      probe_ctx = build_at(target);
      if (!probe_ctx) {
        r.error = error;
        return r;
      }
    }
    ctx = std::move(probe_ctx);
    gemm_ctx = static_cast<VulkanGemmContext*>(ctx.get());
  }

  r.path = gemm_ctx->coopmat ? "tiled(coopmat 16x16x16 f16->f32)" : "tiled(shared 16x16 f16->f32)";

  for (int w = 0; w < kWarmups; ++w) gemm_ctx->time_gemm(1);
  const double t_once = gemm_ctx->time_gemm(1);
  if (t_once <= 0.0) {
    r.error = "device reported no timestamp span";
    return r;
  }
  const int reps =
      params.pinned_reps > 0 ? params.pinned_reps : calibrate_repeats(t_once, kGemmTargetSeconds, kGemmRepCap);
  const double secs = gemm_ctx->time_gemm(reps);
  if (secs <= 0.0) {
    r.error = "device reported no timestamp span";
    return r;
  }

  std::vector<float> host(static_cast<std::size_t>(gemm::kVerifyRows) * gemm_ctx->n);
  gemm_ctx->read_verify_rows(host);

  bool ok = true;
  for (std::uint32_t s = 0; s < gemm::kVerifySamples && ok; ++s) {
    const std::uint32_t row = gemm::sample_row(s);
    const std::uint32_t col = gemm::sample_col(s, gemm_ctx->n);
    const double got = host[static_cast<std::size_t>(row) * gemm_ctx->n + col];
    const double expected = gemm::reference_element(row, col, gemm_ctx->n, gemm_ctx->seed);
    if (!gemm::element_ok(got, expected, gemm_ctx->n)) {
      char b[160];
      std::snprintf(b, sizeof(b), "sample mismatch at (%u,%u): got=%g expected=%g", row, col, got, expected);
      r.error = b;
      ok = false;
    }
  }

  r.work = gemm::gemm_flops(gemm_ctx->n, reps);
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  r.info = {{"gemm_n", std::to_string(gemm_ctx->n)},
            {"reps", std::to_string(reps)},
            {"blas_library", "none"},
            {"math_mode", "fp16 inputs, fp32 accumulate"}};
  return r;
}

}  // namespace bench
