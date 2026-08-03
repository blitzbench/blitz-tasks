/**
 * @file oneapi_runner.cpp
 * @brief Level Zero / oneAPI device-local VRAM write runner. A device USM buffer is
 *        filled by the shared vram_write.comp (grid-stride uvec4 stores). Timed via
 *        kernel-timestamp events over a calibrated rep count; verification windows are
 *        copied back to host. work = reps * S (writes only).
 */

#include "oneapi_runner.hpp"

#include <level_zero/ze_api.h>

#include <algorithm>
#include <bench/config.hpp>
#include <bench/l0_utils.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../../kernel_params.hpp"
#include "vram_write.spv.inl"  // gpu_vram_write_shader::k_vram_write_spv_bytes / _len

namespace bench {

namespace {

struct PushConstants {
  std::uint32_t n_vec4;
};

}  // namespace

RunResult run_gpu_vram_write_oneapi(const gpgpu::Setup& setup) {
  using namespace gpu_vram_write_shader;

  RunResult r;
  r.path = "simd(uvec4 stores)";
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
      create_module_with_log(ctx, dev, k_vram_write_spv_bytes, k_vram_write_spv_bytes_len, "", log, rc);
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
  const std::uint32_t w = std::min<std::uint32_t>(vram::kWindowU32, N);

  ze_device_mem_alloc_desc_t mad{};
  mad.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void* d_buf = nullptr;
  zeMemAllocDevice(ctx, &mad, S, 16, dev, &d_buf);
  zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &d_buf);

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

  const std::uint32_t groups = (vram::thread_count(setup.device) + vram::kBlock - 1) / vram::kBlock;

  auto time_writes = [&](int reps) -> double {
    zeCommandListReset(list);
    zeEventHostReset(ev_first);
    zeEventHostReset(ev_last);
    PushConstants pc{n_vec4};
    zeKernelSetArgumentValue(kernel, 1, sizeof(pc), &pc);
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

  auto read_window = [&](std::uint32_t start, std::vector<std::uint32_t>& host) {
    zeCommandListReset(list);
    zeCommandListAppendMemoryCopy(list, host.data(),
                                  static_cast<std::uint8_t*>(d_buf) + static_cast<std::size_t>(start) * 4,
                                  static_cast<std::size_t>(w) * 4, nullptr, 0, nullptr);
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX);
  };

  for (int i = 0; i < kWarmups; ++i) time_writes(1);
  const double t_once = time_writes(1);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  const auto wall0 = std::chrono::steady_clock::now();
  const double secs = time_writes(reps);
  const auto wall1 = std::chrono::steady_clock::now();

  bool ok = true;
  std::string verr;
  std::vector<std::uint32_t> host(w);
  auto check_window = [&](std::uint32_t start) {
    read_window(start, host);
    for (std::uint32_t i = 0; i < w && ok; ++i) {
      const std::uint32_t idx = start + i;
      if (host[i] != vram::pattern(idx)) {
        char b[128];
        std::snprintf(b, sizeof(b), "write mismatch at u32[%u]: got=%u expected=%u", idx, host[i], vram::pattern(idx));
        verr = b;
        ok = false;
      }
    }
  };
  check_window(0);
  if (ok && N > w) check_window(N - w);

  zeEventDestroy(ev_first);
  zeEventDestroy(ev_last);
  zeEventPoolDestroy(epool);
  zeMemFree(ctx, d_buf);
  zeKernelDestroy(kernel);
  zeModuleDestroy(module);
  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeContextDestroy(ctx);

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
