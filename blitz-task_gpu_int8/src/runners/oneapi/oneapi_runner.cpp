/**
 * @file oneapi_runner.cpp
 * @brief Level Zero / oneAPI int8 throughput runner — SDK-required variant.
 *
 * PACKED ONLY: glslang cannot emit the Kernel-flavour joint-matrix SPIR-V the
 * L0 compiler wants, so this consumes the shared packed shader (int8_packed,
 * i8vec4 MACs) and runs the char4/i8vec path (path "simd(i8vec, level-zero)").
 * Timed via kernel-timestamp events over a calibrated rep count. Unit GOPS.
 *
 * int8-capability SPIR-V acceptance varies by driver; a zeModuleCreate failure
 * is reported as supported=false (a clean "n/a" row), NOT an error.
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
#include "int8_packed.spv.inl"  // gpu_int8_shader::k_int8_packed_spv_bytes / _len

namespace bench {

namespace {

struct PushConstants {
  std::uint32_t n;
  std::uint32_t iters;
};

}  // namespace

RunResult run_gpu_int8_oneapi(const gpgpu::Setup& setup) {
  using namespace gpu_int8_shader;

  RunResult r;
  r.path = "simd(i8vec, level-zero)";
  r.score_unit = "GOPS";

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
      create_module_with_log(ctx, dev, k_int8_packed_spv_bytes, k_int8_packed_spv_bytes_len, "", log, rc);
  if (!module) {
    // int8 SPIR-V not accepted by this driver — clean unsupported row.
    r.supported = false;
    r.path = "unsupported(int8 SPIR-V rejected by L0)";
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
  zeKernelSetGroupSize(kernel, i8::kBlock, 1, 1);

  const std::uint32_t T = i8::thread_count(setup.device);
  const std::size_t bytes = static_cast<std::size_t>(T) * sizeof(std::uint32_t);

  ze_device_mem_alloc_desc_t mad{};
  mad.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void* d_out = nullptr;
  zeMemAllocDevice(ctx, &mad, bytes, alignof(std::uint32_t), dev, &d_out);
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
    groups.groupCountX = (n + i8::kBlock - 1) / i8::kBlock;
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

  auto read_back = [&](std::uint32_t count, std::vector<std::uint32_t>& host) {
    zeCommandListReset(list);
    zeCommandListAppendMemoryCopy(list, host.data(), d_out, count * sizeof(std::uint32_t), nullptr, 0, nullptr);
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX);
  };

  std::vector<std::uint32_t> host(T);
  bool ok = true;

  auto verify = [&](std::uint32_t count, std::uint32_t iters, std::uint32_t stride) -> bool {
    for (std::uint32_t s = 0; s < count; ++s) {
      const std::uint32_t t = s * stride;
      const std::uint32_t exp = i8::reference_packed(t, iters);
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
    r.work = i8::ops_packed(T, i8::kIters, reps);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = true;
  }
  return r;
}

}  // namespace bench
