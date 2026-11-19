/**
 * @file vulkan_runner.cpp
 * @brief Vulkan compute int8 throughput runner.
 *
 * Tensor-first with a runtime fallback; the chosen path is recorded:
 *   VK_KHR_cooperative_matrix with a subgroup {s8,s8,s32} configuration, the
 *   coopmat feature, the Vulkan memory model, shaderInt8, and the coopmat SPIR-V
 *   present at build
 *        -> int8_coopmat  path "tensor(coopmat MxNxK s8->s32)"
 *   else shaderInt8 available
 *        -> int8_packed   path "simd(i8vec4)"
 *   else -> supported=false, "unsupported(no shaderInt8)"
 *
 * The coopmat shape comes from the device's list (bench::best_coopmat_shape). The
 * kernel feeds all-ones s8 A,B into the MMA, so every C element equals K*iters
 * (bench::i8::reference_tensor). Some drivers list cooperative matrices on hardware
 * without matrix units and emulate them, so when both kernels are available each
 * is timed once and the faster one is benchmarked. The packed kernel matches
 * bench::i8::reference_packed. Both write a uint32 storage buffer. Unit GOPS. See
 * kernel_params.hpp for math + ops accounting.
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
  std::int32_t one;  // A/B element value of the coopmat kernel, ignored by the packed kernel
};

bool device_has_extension(const gpgpu::vendor::VulkanInstanceFns& vki, VkPhysicalDevice pd, const char* name) {
  std::uint32_t n = 0;
  vki.vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
  if (n == 0) return false;
  std::vector<VkExtensionProperties> ext(n);
  vki.vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, ext.data());
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

  const gpgpu::vendor::VulkanFns* vk_fns = gpgpu::vendor::vulkan();
  if (!vk_fns) {
    r.error = "Vulkan loader unavailable";
    return r;
  }
  const gpgpu::vendor::VulkanFns& vk = *vk_fns;
  gpgpu::vendor::VulkanInstanceFns vki{};
  gpgpu::vendor::VulkanDeviceFns vkd{};

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_int8";
  app.apiVersion = VK_API_VERSION_1_3;
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

  VkPhysicalDevice pd = gpgpu::vendor::find_vulkan_device(vki, inst, setup.device);
  if (!pd) {
    vki.vkDestroyInstance(inst, nullptr);
    r.error = "no Vulkan device matched " + setup.device.id();
    return r;
  }

  VkPhysicalDeviceProperties pd_props{};
  vki.vkGetPhysicalDeviceProperties(pd, &pd_props);
  const float ts_period_ns = pd_props.limits.timestampPeriod;
  const bool ts_ok = pd_props.limits.timestampComputeAndGraphics != 0;

  // --- feature probe: shaderInt8 (both kernels), memory model (coopmat kernel) ---
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feat2.pNext = &f12;
  vki.vkGetPhysicalDeviceFeatures2(pd, &feat2);

  if (f12.shaderInt8 != VK_TRUE) {
    vki.vkDestroyInstance(inst, nullptr);
    r.supported = false;
    r.path = "unsupported(no shaderInt8)";
    return r;
  }

  CoopmatShape shape{};
#if defined(GPU_INT8_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
  if (f12.vulkanMemoryModel == VK_TRUE && device_has_extension(vki, pd, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    q.pNext = &cmf;
    vki.vkGetPhysicalDeviceFeatures2(pd, &q);
    if (cmf.cooperativeMatrix == VK_TRUE)
      shape = best_coopmat_shape(vk, inst, pd, VK_COMPONENT_TYPE_SINT8_KHR, VK_COMPONENT_TYPE_SINT32_KHR);
  }
#endif
  const bool have_coopmat = static_cast<bool>(shape);

  const std::uint32_t qfam = find_queue_family(vki, pd, VK_QUEUE_COMPUTE_BIT);
  if (qfam == UINT32_MAX) {
    vki.vkDestroyInstance(inst, nullptr);
    r.error = "no compute queue family";
    return r;
  }

  VkPhysicalDeviceMemoryProperties mp{};
  vki.vkGetPhysicalDeviceMemoryProperties(pd, &mp);

  // --- device with the required feature chain ---
  VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  en12.shaderInt8 = VK_TRUE;
  std::vector<const char*> dev_exts;
#if defined(GPU_INT8_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR en_cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  if (have_coopmat) {
    en12.vulkanMemoryModel = VK_TRUE;
    en_cm.cooperativeMatrix = VK_TRUE;
    en12.pNext = &en_cm;
    dev_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  }
#endif

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &en12;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = static_cast<std::uint32_t>(dev_exts.size());
  dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
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
  VkQueue queue = VK_NULL_HANDLE;
  vkd.vkGetDeviceQueue(dev, qfam, 0, &queue);

  const std::uint32_t T = i8::thread_count(setup.device);
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(T) * sizeof(std::uint32_t);

  VkBufferAlloc out{};
  VkBufferAlloc count{};
  if (!create_buffer(vkd, dev, mp, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out) ||
      !create_buffer(vkd, dev, mp, sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, count)) {
    destroy_buffer(vkd, dev, count);
    destroy_buffer(vkd, dev, out);
    vkd.vkDestroyDevice(dev, nullptr);
    vki.vkDestroyInstance(inst, nullptr);
    r.error = "buffer alloc failed";
    return r;
  }

  // Both kernels share one layout; the packed kernel ignores the subgroup counter at binding 1.
  VkDescriptorSetLayoutBinding binds[2]{};
  for (std::uint32_t i = 0; i < 2; ++i) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 2;
  dslci.pBindings = binds;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  vkd.vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl = VK_NULL_HANDLE;
  vkd.vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

  std::vector<VkShaderModule> modules;
  auto make_pipeline = [&](const unsigned char* spv, std::size_t len, const VkSpecializationInfo* spec) -> VkPipeline {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = len;
    smci.pCode = reinterpret_cast<const std::uint32_t*>(spv);
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vkd.vkCreateShaderModule(dev, &smci, nullptr, &shader) != VK_SUCCESS) return VK_NULL_HANDLE;
    modules.push_back(shader);
    VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = shader;
    ssci.pName = "main";
    ssci.pSpecializationInfo = spec;
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci;
    cpci.layout = pl;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkd.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    return pipeline;
  };

  const VkPipeline packed_pipe = make_pipeline(k_int8_packed_spv_bytes, k_int8_packed_spv_bytes_len, nullptr);
  VkPipeline coop_pipe = VK_NULL_HANDLE;
#if defined(GPU_INT8_COOPMAT_SPV_AVAILABLE) && defined(VK_KHR_cooperative_matrix)
  const std::uint32_t mnk[3] = {shape.m, shape.n, shape.k};
  const VkSpecializationMapEntry sme[3] = {{0, 0, sizeof(std::uint32_t)},
                                           {1, sizeof(std::uint32_t), sizeof(std::uint32_t)},
                                           {2, 2 * sizeof(std::uint32_t), sizeof(std::uint32_t)}};
  const VkSpecializationInfo spec{3, sme, sizeof(mnk), mnk};
  if (have_coopmat) coop_pipe = make_pipeline(k_int8_coopmat_spv_bytes, k_int8_coopmat_spv_bytes_len, &spec);
#endif

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool dpool = VK_NULL_HANDLE;
  vkd.vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool);
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dpool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet dset = VK_NULL_HANDLE;
  vkd.vkAllocateDescriptorSets(dev, &dsai, &dset);
  VkDescriptorBufferInfo dbi[2] = {{out.buf, 0, VK_WHOLE_SIZE}, {count.buf, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet wr[2]{};
  for (std::uint32_t i = 0; i < 2; ++i) {
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = dset;
    wr[i].dstBinding = i;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &dbi[i];
  }
  vkd.vkUpdateDescriptorSets(dev, 2, wr, 0, nullptr);

  VkQueryPool qpool = ts_ok ? create_timestamp_pool(vkd, dev, 2) : VK_NULL_HANDLE;

  VkCommandPoolCreateInfo cpoolci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpoolci.queueFamilyIndex = qfam;
  cpoolci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool cpool = VK_NULL_HANDLE;
  vkd.vkCreateCommandPool(dev, &cpoolci, nullptr, &cpool);
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = cpool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  vkd.vkAllocateCommandBuffers(dev, &cbai, &cb);

  void* count_map = nullptr;
  vkd.vkMapMemory(dev, count.mem, 0, sizeof(std::uint32_t), 0, &count_map);
  auto subgroups_ran = [&]() { return *static_cast<volatile std::uint32_t*>(count_map); };

  // Runs `reps` dispatches covering `n` invocations; the subgroup counter is reset first.
  auto time_launches = [&](VkPipeline pipe, std::uint32_t n, std::uint32_t iters, int reps) -> double {
    *static_cast<volatile std::uint32_t*>(count_map) = 0;
    vkd.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkd.vkBeginCommandBuffer(cb, &bi);
    if (qpool) {
      vkd.vkCmdResetQueryPool(cb, qpool, 0, 2);
      vkd.vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    }
    vkd.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkd.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
    PushConstants pc{n, iters, 1};
    const std::uint32_t groups = (n + i8::kBlock - 1) / i8::kBlock;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    for (int i = 0; i < reps; ++i) {
      vkd.vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkd.vkCmdDispatch(cb, groups, 1, 1);
      if (i + 1 < reps)
        vkd.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                                 0, nullptr, 0, nullptr);
    }
    if (qpool) vkd.vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
    VkMemoryBarrier to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkd.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
                             nullptr, 0, nullptr);
    vkd.vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkd.vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkd.vkQueueWaitIdle(queue);
    double secs = 0.0;
    if (qpool) read_timestamp_span(vkd, dev, qpool, ts_period_ns, secs);
    return secs;
  };

  auto read_back = [&](std::uint32_t n, std::vector<std::uint32_t>& host) {
    void* mapped = nullptr;
    vkd.vkMapMemory(dev, out.mem, 0, n * sizeof(std::uint32_t), 0, &mapped);
    std::memcpy(host.data(), mapped, n * sizeof(std::uint32_t));
    vkd.vkUnmapMemory(dev, out.mem);
  };

  std::vector<std::uint32_t> host(T);
  std::string err;

  auto verify = [&](bool tensor, std::uint32_t n, std::uint32_t iters, std::uint32_t stride) -> bool {
    for (std::uint32_t s = 0; s < n; ++s) {
      const std::uint32_t t = s * stride;
      const std::uint32_t exp = tensor
                                    ? static_cast<std::uint32_t>(i8::reference_tensor(iters, static_cast<int>(shape.k)))
                                    : i8::reference_packed(t, iters);
      if (host[t] != exp) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "mismatch at t=%u: got=%u expected=%u", t, host[t], exp);
        err = buf;
        return false;
      }
    }
    return true;
  };

  auto preflight = [&](bool tensor) -> bool {
    time_launches(tensor ? coop_pipe : packed_pipe, i8::kPreflightThreads, i8::kPreflightIters, 1);
    read_back(i8::kPreflightThreads, host);
    return verify(tensor, i8::kPreflightThreads, i8::kPreflightIters, 1);
  };

  // Work of the last time_launches call on the main configuration.
  auto work_of = [&](bool tensor, int reps) -> std::uint64_t {
    return tensor ? static_cast<std::uint64_t>(subgroups_ran()) * i8::kIters * shape.flops()
                  : i8::ops_packed(T, i8::kIters, reps);
  };

  const bool packed_ok = packed_pipe != VK_NULL_HANDLE && preflight(false);
  const std::string packed_err = packed_pipe ? err : "vkCreateComputePipelines failed";
  err.clear();
  const bool coop_ok = coop_pipe != VK_NULL_HANDLE && preflight(true);

  // One timed launch of each candidate after warm-up; the faster one is benchmarked.
  bool tensor = coop_ok;
  if (coop_ok && packed_ok) {
    double rate[2] = {0.0, 0.0};
    for (int k = 0; k < 2; ++k) {
      const bool t = k == 1;
      const VkPipeline pipe = t ? coop_pipe : packed_pipe;
      for (int w = 0; w < kWarmups; ++w) time_launches(pipe, T, i8::kIters, 1);
      const double secs = time_launches(pipe, T, i8::kIters, 1);
      rate[k] = secs > 0.0 ? static_cast<double>(work_of(t, 1)) / secs : 0.0;
    }
    tensor = rate[1] >= rate[0];
  }

  bool ok = tensor || packed_ok;
  if (!ok) r.error = packed_err.empty() ? err : packed_err;
  r.path = tensor ? "tensor(coopmat " + shape.str() + " s8->s32)" : "simd(i8vec4)";

  double secs = 0.0;
  int reps = 0;
  std::uint64_t work = 0;
  if (ok) {
    const VkPipeline pipe = tensor ? coop_pipe : packed_pipe;
    for (int w = 0; w < kWarmups; ++w) time_launches(pipe, T, i8::kIters, 1);
    const double t_once = time_launches(pipe, T, i8::kIters, 1);
    reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
    secs = time_launches(pipe, T, i8::kIters, reps);
    work = work_of(tensor, reps);
    read_back(T, host);
    constexpr std::uint32_t kSamples = 64;
    const std::uint32_t stride = (T - 1) / (kSamples - 1);
    ok = verify(tensor, kSamples, i8::kIters, stride);
    if (!ok) r.error = err;
  }

  vkd.vkUnmapMemory(dev, count.mem);
  if (qpool) vkd.vkDestroyQueryPool(dev, qpool, nullptr);
  vkd.vkDestroyCommandPool(dev, cpool, nullptr);
  vkd.vkDestroyDescriptorPool(dev, dpool, nullptr);
  if (coop_pipe) vkd.vkDestroyPipeline(dev, coop_pipe, nullptr);
  if (packed_pipe) vkd.vkDestroyPipeline(dev, packed_pipe, nullptr);
  vkd.vkDestroyPipelineLayout(dev, pl, nullptr);
  vkd.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  for (VkShaderModule m : modules) vkd.vkDestroyShaderModule(dev, m, nullptr);
  destroy_buffer(vkd, dev, count);
  destroy_buffer(vkd, dev, out);
  vkd.vkDestroyDevice(dev, nullptr);
  vki.vkDestroyInstance(inst, nullptr);

  if (ok) {
    r.work = work;
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = true;
    if (tensor) r.info.emplace_back("coopmat_shape", shape.str());
  }
  return r;
}

}  // namespace bench
