/**
 * @file vulkan_runner.cpp
 * @brief Vulkan fp32 GEMM runner — shared-memory tiled multiply.
 *
 * Vulkan has no vendor BLAS to call, and KHR_cooperative_matrix is specified for
 * reduced-precision operands rather than fp32 inputs, so this runner always reports:
 *
 *   "tiled(shared 16x16 fp32)"
 *
 * It is what the task reports wherever cuBLASLt, rocBLAS and MPS are all out of reach.
 *
 * Operands never cross the bus: gemm_fill.comp writes A and B from the coordinate hash
 * once, when the context is built. Only kVerifyRows rows of C come back, and the
 * sampled elements are checked against bench::gemm::reference_element.
 */

#include "vulkan_runner.hpp"

#include <vulkan/vulkan.h>

#include "gemm_fill.spv.inl"   // gpu_matmul_fp32_shader::k_gemm_fill_spv_bytes / _len
#include "gemm_tiled.spv.inl"  // gpu_matmul_fp32_shader::k_gemm_tiled_spv_bytes / _len

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

/**
 * @class VulkanGemmContext
 * @brief Instance, device, operand buffers and pipelines for one (device, size) pair.
 */
struct VulkanGemmContext : gemm::Context {
  std::string device_id;
  std::uint32_t n{0};
  std::uint32_t seed{0};

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
      if (qpool) vkDestroyQueryPool(dev, qpool, nullptr);
      if (cpool) vkDestroyCommandPool(dev, cpool, nullptr);
      if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
      if (fill_pipe) vkDestroyPipeline(dev, fill_pipe, nullptr);
      if (gemm_pipe) vkDestroyPipeline(dev, gemm_pipe, nullptr);
      if (layout) vkDestroyPipelineLayout(dev, layout, nullptr);
      if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
      if (fill_shader) vkDestroyShaderModule(dev, fill_shader, nullptr);
      if (gemm_shader) vkDestroyShaderModule(dev, gemm_shader, nullptr);
      destroy_buffer(dev, a);
      destroy_buffer(dev, b);
      destroy_buffer(dev, c);
      destroy_buffer(dev, staging);
      vkDestroyDevice(dev, nullptr);
      dev = VK_NULL_HANDLE;
    }
    if (inst) {
      vkDestroyInstance(inst, nullptr);
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
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    if (qpool) {
      vkCmdResetQueryPool(cb, qpool, 0, 2);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gemm_pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
    const PushConstants pc{n, seed};
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    const std::uint32_t groups = n / kTile;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    for (int i = 0; i < reps; ++i) {
      vkCmdDispatch(cb, groups, groups, 1);
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
  }

  // Run the fill kernel once so A and B hold their hashed operands.
  void fill_operands() {
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
    const PushConstants pc{n, seed};
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    const std::uint32_t groups = n / kTile;
    vkCmdDispatch(cb, groups, groups, 1);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
  }

  /**
   * @brief Copy the first gemm::kVerifyRows rows of C into @p host.
   *
   * @param host
   */
  void read_verify_rows(std::vector<float>& host) {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(gemm::kVerifyRows) * n * sizeof(float);
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cb, c.buf, staging.buf, 1, &region);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    void* mapped = nullptr;
    vkMapMemory(dev, staging.mem, 0, bytes, 0, &mapped);
    std::memcpy(host.data(), mapped, static_cast<std::size_t>(bytes));
    vkUnmapMemory(dev, staging.mem);
  }
};

/**
 * @brief Build a context for @p setup at problem size @p n.
 *
 * @param setup
 * @param n
 * @param seed
 * @param out
 * @param error
 * @return false with @p error populated on any failure; @p out is then unusable.
 */
bool build_context(const gpgpu::Setup& setup, std::uint32_t n, std::uint32_t seed, VulkanGemmContext& out,
                   std::string& error) {
  using namespace gpu_matmul_fp32_shader;

  out.device_id = setup.device.id();
  out.n = n;
  out.seed = seed;

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpu_matmul_fp32";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  if (vkCreateInstance(&ici, nullptr, &out.inst) != VK_SUCCESS) {
    error = "vkCreateInstance failed";
    return false;
  }

  VkPhysicalDevice pd = find_vk_device(out.inst, setup.device);
  if (!pd) {
    error = "no Vulkan device matched " + setup.device.id();
    return false;
  }

  VkPhysicalDeviceProperties pd_props{};
  vkGetPhysicalDeviceProperties(pd, &pd_props);
  out.ts_period_ns = pd_props.limits.timestampPeriod;
  out.ts_ok = pd_props.limits.timestampComputeAndGraphics != 0;


  out.qfam = find_queue_family(pd, VK_QUEUE_COMPUTE_BIT);
  if (out.qfam == UINT32_MAX) {
    error = "no compute queue family";
    return false;
  }

  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = out.qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  v12.vulkanMemoryModel = VK_TRUE;
  v12.vulkanMemoryModelDeviceScope = VK_TRUE;
  VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feat2.pNext = &v12;

  const std::vector<const char*> dev_exts;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &feat2;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = static_cast<std::uint32_t>(dev_exts.size());
  dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
  if (vkCreateDevice(pd, &dci, nullptr, &out.dev) != VK_SUCCESS) {
    error = "vkCreateDevice failed";
    return false;
  }
  vkGetDeviceQueue(out.dev, out.qfam, 0, &out.queue);

  const VkDeviceSize elems = static_cast<VkDeviceSize>(n) * n;
  const VkDeviceSize verify_bytes = static_cast<VkDeviceSize>(gemm::kVerifyRows) * n * sizeof(float);
  const VkMemoryPropertyFlags local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkMemoryPropertyFlags host_vis =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (!create_buffer(out.dev, mp, elems * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, local, out.a) ||
      !create_buffer(out.dev, mp, elems * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, local, out.b) ||
      !create_buffer(out.dev, mp, elems * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     local, out.c) ||
      !create_buffer(out.dev, mp, verify_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_vis, out.staging)) {
    error = "operand allocation failed at n=" + std::to_string(n);
    return false;
  }

  auto make_module = [&](const unsigned char* spv, unsigned int len) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = len;
    smci.pCode = reinterpret_cast<const std::uint32_t*>(spv);
    VkShaderModule m = VK_NULL_HANDLE;
    vkCreateShaderModule(out.dev, &smci, nullptr, &m);
    return m;
  };
  out.fill_shader = make_module(k_gemm_fill_spv_bytes, k_gemm_fill_spv_bytes_len);
  out.gemm_shader = make_module(k_gemm_tiled_spv_bytes, k_gemm_tiled_spv_bytes_len);
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
  vkCreateDescriptorSetLayout(out.dev, &dslci, nullptr, &out.dsl);

  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &out.dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  vkCreatePipelineLayout(out.dev, &plci, nullptr, &out.layout);

  auto make_pipeline = [&](VkShaderModule module) {
    VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = module;
    ssci.pName = "main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci;
    cpci.layout = out.layout;
    VkPipeline p = VK_NULL_HANDLE;
    vkCreateComputePipelines(out.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p);
    return p;
  };
  out.fill_pipe = make_pipeline(out.fill_shader);
  out.gemm_pipe = make_pipeline(out.gemm_shader);
  if (!out.fill_pipe || !out.gemm_pipe) {
    error = "vkCreateComputePipelines failed";
    return false;
  }

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  vkCreateDescriptorPool(out.dev, &dpci, nullptr, &out.dpool);
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = out.dpool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &out.dsl;
  vkAllocateDescriptorSets(out.dev, &dsai, &out.dset);

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
  vkUpdateDescriptorSets(out.dev, 3, writes, 0, nullptr);

  out.qpool = out.ts_ok ? create_timestamp_pool(out.dev, 2) : VK_NULL_HANDLE;

  VkCommandPoolCreateInfo cpoolci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpoolci.queueFamilyIndex = out.qfam;
  cpoolci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkCreateCommandPool(out.dev, &cpoolci, nullptr, &out.cpool);
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = out.cpool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  vkAllocateCommandBuffers(out.dev, &cbai, &out.cb);

  out.fill_operands();
  return true;
}

}  // namespace

RunResult run_gpu_matmul_fp32_vulkan(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                     const gemm::RunParams& params) {
  RunResult r;
  r.score_unit = "GFLOPS";
  r.path = "unsupported(fp32 gemm)";

  auto* vk = dynamic_cast<VulkanGemmContext*>(ctx.get());
  if (vk && (vk->device_id != setup.device.id() || vk->seed != params.seed)) vk = nullptr;

  // Size the problem on the first round; later rounds reuse what this settled on. The
  // bottom rung is timed first and extrapolated, so a slow device is never asked to run
  // a multiply it cannot finish, and the ladder steps down again if allocation fails.
  if (!vk) {
    const std::uint32_t memory_limit_n = gemm::choose_size(gemm::Precision::Fp32, setup.device, params.cap_bytes);
    std::string error;

    auto build_at = [&](std::uint32_t n) -> std::unique_ptr<VulkanGemmContext> {
      while (n) {
        auto fresh = std::make_unique<VulkanGemmContext>();
        if (build_context(setup, n, params.seed, *fresh, error)) return fresh;
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
    vk = static_cast<VulkanGemmContext*>(ctx.get());
  }

  r.path = "tiled(shared 16x16 fp32)";

  for (int w = 0; w < kWarmups; ++w) vk->time_gemm(1);
  const double t_once = vk->time_gemm(1);
  if (t_once <= 0.0) {
    r.error = "device reported no timestamp span";
    return r;
  }
  const int reps =
      params.pinned_reps > 0 ? params.pinned_reps : calibrate_repeats(t_once, kGemmTargetSeconds, kGemmRepCap);
  const double secs = vk->time_gemm(reps);
  if (secs <= 0.0) {
    r.error = "device reported no timestamp span";
    return r;
  }

  std::vector<float> host(static_cast<std::size_t>(gemm::kVerifyRows) * vk->n);
  vk->read_verify_rows(host);

  bool ok = true;
  for (std::uint32_t s = 0; s < gemm::kVerifySamples && ok; ++s) {
    const std::uint32_t row = gemm::sample_row(s);
    const std::uint32_t col = gemm::sample_col(s, vk->n);
    const double got = host[static_cast<std::size_t>(row) * vk->n + col];
    const double expected = gemm::reference_element(row, col, vk->n, vk->seed);
    if (!gemm::element_ok(got, expected, vk->n)) {
      char b[160];
      std::snprintf(b, sizeof(b), "sample mismatch at (%u,%u): got=%g expected=%g", row, col, got, expected);
      r.error = b;
      ok = false;
    }
  }

  r.work = gemm::gemm_flops(vk->n, reps);
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  r.info = {{"gemm_n", std::to_string(vk->n)},
            {"reps", std::to_string(reps)},
            {"blas_library", "none"},
            {"math_mode", "fp32 throughout"}};
  return r;
}

}  // namespace bench
