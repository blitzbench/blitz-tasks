/**
 * @file oneapi_runner.cpp
 * @brief Level Zero / oneAPI device-local VRAM read runner. A device USM input buffer is
 *        prefilled (untimed) with the pattern from host USM. The shared vram_read.comp
 *        (grid-stride uvec4 loads) accumulates per-thread and writes one uint per thread
 *        to a device out-buffer. Timed via kernel-timestamp events. work = reps * S.
 *        Verified against the closed form (per-thread outputs copied back and summed).
 */

#include "oneapi_runner.hpp"

#include <level_zero/ze_api.h>

#include <bench/config.hpp>
#include <bench/l0_utils.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

RunResult run_gpu_vram_read_oneapi(const gpgpu::Setup& setup) {
  using namespace gpu_vram_read_shader;

  RunResult r;
  r.path = "simd(uvec4 loads)";
  r.score_unit = "GB/s";

  if (zeInit(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS && zeInit(0) != ZE_RESULT_SUCCESS) {
    r.error = "zeInit failed";
    return r;
  }

  ze_driver_handle_t drv = nullptr;
  ze_device_handle_t dev = nullptr;
  if (!find_l0_device(setup.device, drv, dev)) {
    r.error = "no Level Zero device matched " + setup.device.id();
    return r;
  }

  ze_device_properties_t dev_props{};
  dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  zeDeviceGetProperties(dev, &dev_props);
  const std::uint64_t timer_res_ns = dev_props.timerResolution;

  ze_context_desc_t ctx_desc{};
  ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t ctx = nullptr;
  if (zeContextCreate(drv, &ctx_desc, &ctx) != ZE_RESULT_SUCCESS) {
    r.error = "zeContextCreate failed";
    return r;
  }

  std::string log;
  ze_result_t rc = ZE_RESULT_SUCCESS;
  ze_module_handle_t module =
      create_module_with_log(ctx, dev, k_vram_read_spv_bytes, k_vram_read_spv_bytes_len, "", log, rc);
  if (!module) {
    r.error = "zeModuleCreate failed: " + log;
    zeContextDestroy(ctx);
    return r;
  }

  ze_kernel_desc_t kdesc{};
  kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kdesc.pKernelName = "main";
  ze_kernel_handle_t kernel = nullptr;
  if (zeKernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
    r.error = "zeKernelCreate: no 'main' entry";
    zeModuleDestroy(module);
    zeContextDestroy(ctx);
    return r;
  }
  zeKernelSetGroupSize(kernel, vram::kBlock, 1, 1);

  const std::size_t S = vram::buffer_bytes(setup.device);
  const std::uint32_t N = static_cast<std::uint32_t>(S / 4);
  const std::uint32_t n_vec4 = N / 4;
  const std::uint32_t groups = (vram::thread_count(setup.device) + vram::kBlock - 1) / vram::kBlock;
  const std::uint32_t T_launch = groups * vram::kBlock;
  const std::size_t out_bytes = static_cast<std::size_t>(T_launch) * 4;

  ze_device_mem_alloc_desc_t mad{};
  mad.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  ze_host_mem_alloc_desc_t had{};
  had.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;

  void* d_in = nullptr;
  void* d_out = nullptr;
  void* h_stage = nullptr;
  zeMemAllocDevice(ctx, &mad, S, 16, dev, &d_in);
  zeMemAllocDevice(ctx, &mad, out_bytes, 16, dev, &d_out);
  zeMemAllocHost(ctx, &had, S, 16, &h_stage);

  ze_command_queue_desc_t cqd{};
  cqd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  cqd.ordinal = 0;
  cqd.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
  cqd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ze_command_queue_handle_t queue = nullptr;
  zeCommandQueueCreate(ctx, dev, &cqd, &queue);

  ze_command_list_desc_t cld{};
  cld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  cld.commandQueueGroupOrdinal = 0;
  ze_command_list_handle_t list = nullptr;
  zeCommandListCreate(ctx, dev, &cld, &list);

  ze_event_pool_desc_t epd{};
  epd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  epd.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
  epd.count = 2;
  ze_event_pool_handle_t epool = nullptr;
  zeEventPoolCreate(ctx, &epd, 1, &dev, &epool);
  auto make_event = [&](std::uint32_t i) {
    ze_event_desc_t ed{};
    ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    ed.index = i;
    ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    ze_event_handle_t e = nullptr;
    zeEventCreate(epool, &ed, &e);
    return e;
  };
  ze_event_handle_t ev_first = make_event(0);
  ze_event_handle_t ev_last = make_event(1);

  auto cleanup = [&]() {
    zeEventDestroy(ev_first);
    zeEventDestroy(ev_last);
    zeEventPoolDestroy(epool);
    if (d_in) zeMemFree(ctx, d_in);
    if (d_out) zeMemFree(ctx, d_out);
    if (h_stage) zeMemFree(ctx, h_stage);
    zeKernelDestroy(kernel);
    zeModuleDestroy(module);
    zeCommandListDestroy(list);
    zeCommandQueueDestroy(queue);
    zeContextDestroy(ctx);
  };

  if (!d_in || !d_out || !h_stage) {
    r.error = "L0 allocation failed";
    cleanup();
    return r;
  }

  zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &d_in);
  zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &d_out);

  // --- prefill input buffer (untimed) ---
  {
    auto* p = static_cast<std::uint32_t*>(h_stage);
    for (std::uint32_t i = 0; i < N; ++i) p[i] = vram::pattern(i);
    zeCommandListReset(list);
    zeCommandListAppendMemoryCopy(list, d_in, h_stage, S, nullptr, 0, nullptr);
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX);
  }

  auto time_reads = [&](int reps) -> double {
    zeCommandListReset(list);
    zeEventHostReset(ev_first);
    zeEventHostReset(ev_last);
    PushConstants pc{n_vec4};
    zeKernelSetArgumentValue(kernel, 2, sizeof(pc), &pc);
    ze_group_count_t gc{};
    gc.groupCountX = groups;
    gc.groupCountY = 1;
    gc.groupCountZ = 1;
    for (int i = 0; i < reps; ++i) {
      ze_event_handle_t sig = (i == 0) ? ev_first : (i == reps - 1 ? ev_last : nullptr);
      zeCommandListAppendLaunchKernel(list, kernel, &gc, sig, 0, nullptr);
      zeCommandListAppendBarrier(list, nullptr, 0, nullptr);
    }
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX);
    std::uint64_t s0 = 0, e0 = 0, s1 = 0, e1 = 0;
    l0_kernel_ticks(ev_first, s0, e0);
    l0_kernel_ticks(reps == 1 ? ev_first : ev_last, s1, e1);
    return (e1 > s0) ? l0_timestamp_seconds(e1 - s0, timer_res_ns) : 0.0;
  };

  for (int i = 0; i < kWarmups; ++i) time_reads(1);
  const double t_once = time_reads(1);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  const auto wall0 = std::chrono::steady_clock::now();
  const double secs = time_reads(reps);
  const auto wall1 = std::chrono::steady_clock::now();

  // --- verify: exact closed-form checksum ---
  bool ok = true;
  std::string verr;
  {
    std::vector<std::uint32_t> out(T_launch);
    zeCommandListReset(list);
    zeCommandListAppendMemoryCopy(list, out.data(), d_out, out_bytes, nullptr, 0, nullptr);
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX);
    std::uint32_t sum = 0u;
    for (std::uint32_t i = 0; i < T_launch; ++i) sum += out[i];
    const std::uint32_t expected = vram::expected_sum(S);
    if (sum != expected) {
      char b[128];
      std::snprintf(b, sizeof(b), "read checksum mismatch: got=%u expected=%u", sum, expected);
      verr = b;
      ok = false;
    }
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
