/**
 * @file oneapi_runner.cpp
 * @brief Level Zero / oneAPI fp16 throughput runner — packed SIMD only.
 *
 * glslang cannot emit the Kernel-flavor cooperative-matrix SPIR-V the L0
 * compiler wants, so this runner deliberately runs only the packed f16vec4
 * kernel (shared fp16_packed.spv). We gate on the device advertising 16-bit
 * float ops (ZE_DEVICE_MODULE_FLAG_FP16); when absent, or when the L0 compiler
 * rejects the shader-flavor SPIR-V, the row is a clean unsupported one rather
 * than a hard error.  path = "simd(f16vec, level-zero)".
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
#include "fp16_packed.spv.inl"  // gpu_fp16_shader::k_fp16_packed_spv_bytes / _len

namespace bench {

namespace {

struct PushConstants {
  std::uint32_t n;
  std::uint32_t iters;
  float mf;
  float af;
};

}  // namespace

RunResult run_gpu_fp16_oneapi(const gpgpu::Setup& setup) {
  using namespace gpu_fp16_shader;

  RunResult r;
  r.score_unit = "GFLOPS";
  r.path = "simd(f16vec, level-zero)";

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

  // fp16 capability gate.
  ze_device_module_properties_t mod_props{};
  mod_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
  zeDeviceGetModuleProperties(dev, &mod_props);
  if (!(mod_props.flags & ZE_DEVICE_MODULE_FLAG_FP16)) {
    r.supported = false;
    r.path = "unsupported(no fp16, level-zero)";
    return r;
  }

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
      create_module_with_log(ctx, dev, k_fp16_packed_spv_bytes, k_fp16_packed_spv_bytes_len, "", log, rc);
  if (!module) {
    // The L0 compiler rejecting the GLSL-flavor SPIR-V is expected on some
    // stacks; treat as unsupported, not a hard failure.
    r.supported = false;
    r.path = "unsupported(L0 zeModuleCreate rejected SPIR-V)";
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
  zeKernelSetGroupSize(kernel, fp16::kBlock, 1, 1);

  const std::uint32_t T = fp16::thread_count(setup.device);
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

  auto time_launches = [&](std::uint32_t n, std::uint32_t iters, float mf, float af, int reps) -> double {
    zeCommandListReset(list);
    zeEventHostReset(ev_first);
    zeEventHostReset(ev_last);
    PushConstants pc{n, iters, mf, af};
    zeKernelSetArgumentValue(kernel, 1, sizeof(pc), &pc);
    ze_group_count_t groups{};
    groups.groupCountX = (n + fp16::kBlock - 1) / fp16::kBlock;
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

  // --- pre-flight exactness (m=1,a=1) ---
  time_launches(fp16::kPreflightThreads, fp16::kPreflightIters, fp16::kPreM, fp16::kPreA, 1);
  read_back(fp16::kPreflightThreads, host);
  for (std::uint32_t t = 0; t < fp16::kPreflightThreads && ok; ++t) {
    const double e = fp16::simd_preflight_out(t, 4, fp16::kPreflightIters);
    if (!fp16::matches_exact(host[t], e)) {
      char b[160];
      std::snprintf(b, sizeof(b), "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t], e);
      r.error = b;
      ok = false;
    }
  }

  double secs = 0.0;
  int reps = 0;
  if (ok) {
    for (int w = 0; w < kWarmups; ++w) time_launches(T, fp16::kIters, fp16::kMainM, fp16::kMainA, 1);
    const double t_once = time_launches(T, fp16::kIters, fp16::kMainM, fp16::kMainA, 1);
    reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
    secs = time_launches(T, fp16::kIters, fp16::kMainM, fp16::kMainA, reps);
    read_back(T, host);
    for (std::uint32_t s = 0; s < 64 && ok; ++s) {
      const std::uint32_t t = static_cast<std::uint32_t>(static_cast<std::uint64_t>(s) * (T - 1) / 63);
      if (!fp16::simd_main_ok(host[t], 4)) {
        char b[160];
        std::snprintf(b, sizeof(b), "sample out of envelope at t=%u: got=%g", t, host[t]);
        r.error = b;
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
    r.work = fp16::simd_flops(T, fp16::kIters, reps, 4);
    r.measured = std::chrono::duration<double>{secs};
    r.timings.kernel_compute = r.measured;
    r.score = score_giga(r.work, secs);
    r.correct = true;
  }
  return r;
}

}  // namespace bench
