/**
 * @file oneapi_runner.cpp
 * @brief Level Zero / oneAPI fp32 dense-FMA runner — SDK-required variant.
 *
 * Consumes the shared shader build's fp32_fma.spv.inl (GLSL-compiled SPIR-V,
 * entry point "main"). The push_constant block {n, iters} is passed as a single
 * kernel argument after the storage buffer. Timed via kernel-timestamp events
 * over a calibrated rep count. If the L0 compiler rejects the shader-flavor
 * SPIR-V the row reports a clean error.
 */

#include "oneapi_runner.hpp"

#include <level_zero/ze_api.h>

#include <bench/config.hpp>
#include <bench/l0_utils.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../../kernel_params.hpp"
#include "fp32_fma.spv.inl"  // gpu_fp32_shader::k_fp32_fma_spv_bytes / _len

namespace bench {

namespace {

struct PushConstants {
  std::uint32_t n;
  std::uint32_t iters;
};

}  // namespace

RunResult run_gpu_fp32_oneapi(const gpgpu::Setup& setup) {
  using namespace gpu_fp32_shader;

  RunResult r;
  r.path = "simd(fp32 fma)";
  r.score_unit = "GFLOPS";

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
      create_module_with_log(ctx, dev, k_fp32_fma_spv_bytes, k_fp32_fma_spv_bytes_len, "", log, rc);
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
  zeKernelSetGroupSize(kernel, fp32::kBlock, 1, 1);

  const std::uint32_t T = fp32::thread_count(setup.device);
  const std::size_t bytes = static_cast<std::size_t>(T) * sizeof(float);

  ze_device_mem_alloc_desc_t mad{};
  mad.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void* d_out = nullptr;
  zeMemAllocDevice(ctx, &mad, bytes, alignof(float), dev, &d_out);
  zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &d_out);

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

  auto time_launches = [&](std::uint32_t n, std::uint32_t iters, int reps) -> double {
    zeCommandListReset(list);
    zeEventHostReset(ev_first);
    zeEventHostReset(ev_last);
    PushConstants pc{n, iters};
    zeKernelSetArgumentValue(kernel, 1, sizeof(pc), &pc);
    ze_group_count_t groups{};
    groups.groupCountX = (n + fp32::kBlock - 1) / fp32::kBlock;
    groups.groupCountY = 1;
    groups.groupCountZ = 1;
    for (int i = 0; i < reps; ++i) {
      ze_event_handle_t sig = (i == 0) ? ev_first : (i == reps - 1 ? ev_last : nullptr);
      zeCommandListAppendLaunchKernel(list, kernel, &groups, sig, 0, nullptr);
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

  auto read_back = [&](std::uint32_t count, std::vector<float>& host) {
    zeCommandListReset(list);
    zeCommandListAppendMemoryCopy(list, host.data(), d_out, count * sizeof(float), nullptr, 0, nullptr);
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX);
  };

  std::vector<float> host(T);
  bool ok = true;

  // --- pre-flight exactness ---
  time_launches(fp32::kPreflightThreads, fp32::kPreflightIters, 1);
  read_back(fp32::kPreflightThreads, host);
  for (std::uint32_t t = 0; t < fp32::kPreflightThreads && ok; ++t) {
    const float e = fp32::reference(t, fp32::kPreflightIters);
    if (!fp32::matches(host[t], e)) {
      char buf[160];
      std::snprintf(buf, sizeof(buf), "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t], e);
      r.error = buf;
      ok = false;
    }
  }

  double secs = 0.0;
  int reps = 0;
  if (ok) {
    for (int w = 0; w < kWarmups; ++w) time_launches(T, fp32::kIters, 1);
    const double t_once = time_launches(T, fp32::kIters, 1);
    reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
    secs = time_launches(T, fp32::kIters, reps);
    read_back(T, host);
    constexpr std::uint32_t kSamples = 64;
    for (std::uint32_t s = 0; s < kSamples && ok; ++s) {
      const std::uint32_t t = static_cast<std::uint32_t>(static_cast<std::uint64_t>(s) * (T - 1) / (kSamples - 1));
      const float e = fp32::reference(t, fp32::kIters);
      if (!fp32::matches(host[t], e)) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "sample mismatch at t=%u: got=%g expected=%g", t, host[t], e);
        r.error = buf;
        ok = false;
      }
    }
  }

  zeEventDestroy(ev_first);
  zeEventDestroy(ev_last);
  zeEventPoolDestroy(epool);
  zeMemFree(ctx, d_out);
  zeKernelDestroy(kernel);
  zeModuleDestroy(module);
  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeContextDestroy(ctx);

  if (ok) {
    r.work = fp32::flops(T, fp32::kIters, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = true;
  }
  return r;
}

}  // namespace bench
