/**
 * @file oneapi_runner.cpp
 * @brief Level Zero / oneAPI concurrent bidirectional transfer runner.
 *
 * Enumerates the device's command-queue groups. If a COPY-only ordinal exists
 * (COPY flag without COMPUTE) it drives the upload on that dedicated copy engine
 * and the download on ordinal 0; otherwise both directions use two queues on
 * ordinal 0. Upload = host USM -> device USM; download = device USM -> host USM,
 * each via zeCommandListAppendMemoryCopy. Score is aggregate wall throughput over
 * execute -> synchronize on both queues. Per-direction kernel-timestamp spans
 * feed the timings block only.
 */

#include "oneapi_runner.hpp"

#include <level_zero/ze_api.h>

#include <bench/config.hpp>
#include <bench/l0_utils.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../../bidir_params.hpp"

namespace bench {

RunResult run_gpu_bidir_oneapi(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "2 queues (ordinal 0)";
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

  // --- pick queue-group ordinals ---
  std::uint32_t n_groups = 0;
  zeDeviceGetCommandQueueGroupProperties(dev, &n_groups, nullptr);
  std::vector<ze_command_queue_group_properties_t> groups(n_groups);
  for (auto& g : groups) g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  if (n_groups) zeDeviceGetCommandQueueGroupProperties(dev, &n_groups, groups.data());

  std::uint32_t ordCompute = 0;  // ordinal 0 is guaranteed present
  std::uint32_t ordCopy = UINT32_MAX;
  for (std::uint32_t i = 0; i < n_groups; ++i) {
    const auto f = groups[i].flags;
    if ((f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && !(f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
      ordCopy = i;
      break;
    }
  }
  std::uint32_t ordUp = ordCompute, ordDn = ordCompute;
  if (ordCopy != UINT32_MAX) {
    ordUp = ordCopy;  // dedicated copy engine handles the upload
    ordDn = ordCompute;
    r.path = "2 queues (copy-only + compute ordinals)";
  }

  ze_context_desc_t ctx_desc{};
  ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t ctx = nullptr;
  if (zeContextCreate(drv, &ctx_desc, &ctx) != ZE_RESULT_SUCCESS) {
    r.error = "zeContextCreate failed";
    return r;
  }

  const std::size_t S = bidir::buffer_bytes(setup.device);
  const std::size_t N = bidir::elem_count(S);

  // Host + device USM allocations.
  ze_host_mem_alloc_desc_t had{};
  had.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  ze_device_mem_alloc_desc_t mad{};
  mad.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void* h_up = nullptr;
  void* h_dn = nullptr;
  void* d_up = nullptr;
  void* d_dn = nullptr;
  zeMemAllocHost(ctx, &had, S, 16, &h_up);
  zeMemAllocHost(ctx, &had, S, 16, &h_dn);
  zeMemAllocDevice(ctx, &mad, S, 16, dev, &d_up);
  zeMemAllocDevice(ctx, &mad, S, 16, dev, &d_dn);

  auto cleanup = [&]() {
    if (h_up) zeMemFree(ctx, h_up);
    if (h_dn) zeMemFree(ctx, h_dn);
    if (d_up) zeMemFree(ctx, d_up);
    if (d_dn) zeMemFree(ctx, d_dn);
    zeContextDestroy(ctx);
  };
  if (!h_up || !h_dn || !d_up || !d_dn) {
    r.error = "USM allocation failed";
    cleanup();
    return r;
  }

  bidir::fill_pattern(static_cast<std::uint32_t*>(h_up), N);

  // Queues + lists, one per direction/ordinal.
  auto make_queue = [&](std::uint32_t ord) {
    ze_command_queue_desc_t d{};
    d.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    d.ordinal = ord;
    d.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    d.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_handle_t q = nullptr;
    zeCommandQueueCreate(ctx, dev, &d, &q);
    return q;
  };
  auto make_list = [&](std::uint32_t ord) {
    ze_command_list_desc_t d{};
    d.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    d.commandQueueGroupOrdinal = ord;
    ze_command_list_handle_t l = nullptr;
    zeCommandListCreate(ctx, dev, &d, &l);
    return l;
  };
  ze_command_queue_handle_t qUp = make_queue(ordUp), qDn = make_queue(ordDn);
  ze_command_list_handle_t lUp = make_list(ordUp), lDn = make_list(ordDn);

  // Kernel-timestamp events, two per direction (first + last copy).
  ze_event_pool_desc_t epd{};
  epd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  epd.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
  epd.count = 4;
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
  ze_event_handle_t upF = make_event(0), upL = make_event(1);
  ze_event_handle_t dnF = make_event(2), dnL = make_event(3);

  // Seed the device download source with the pattern (pre-window).
  zeCommandListAppendMemoryCopy(lUp, d_dn, h_up, S, nullptr, 0, nullptr);
  zeCommandListClose(lUp);
  zeCommandQueueExecuteCommandLists(qUp, 1, &lUp, nullptr);
  zeCommandQueueSynchronize(qUp, UINT64_MAX);

  auto record = [&](ze_command_list_handle_t l, const void* src, void* dst, int reps, ze_event_handle_t evF,
                    ze_event_handle_t evL) {
    zeCommandListReset(l);
    zeEventHostReset(evF);
    zeEventHostReset(evL);
    for (int i = 0; i < reps; ++i) {
      ze_event_handle_t sig = (i == 0) ? evF : (i == reps - 1 ? evL : nullptr);
      zeCommandListAppendMemoryCopy(l, dst, src, S, sig, 0, nullptr);
      zeCommandListAppendBarrier(l, nullptr, 0, nullptr);
    }
    zeCommandListClose(l);
  };

  auto dir_secs = [&](ze_event_handle_t evF, ze_event_handle_t evL, int reps) -> double {
    std::uint64_t s0 = 0, e0 = 0, s1 = 0, e1 = 0;
    l0_kernel_ticks(evF, s0, e0);
    l0_kernel_ticks(reps == 1 ? evF : evL, s1, e1);
    return (e1 > s0) ? l0_timestamp_seconds(e1 - s0, timer_res_ns) : 0.0;
  };

  auto run_window = [&](int reps, double& h2d_secs, double& d2h_secs) -> double {
    record(lUp, h_up, d_up, reps, upF, upL);  // host -> device
    record(lDn, d_dn, h_dn, reps, dnF, dnL);  // device -> host
    const auto t0 = std::chrono::steady_clock::now();
    zeCommandQueueExecuteCommandLists(qUp, 1, &lUp, nullptr);
    zeCommandQueueExecuteCommandLists(qDn, 1, &lDn, nullptr);
    zeCommandQueueSynchronize(qUp, UINT64_MAX);
    zeCommandQueueSynchronize(qDn, UINT64_MAX);
    const auto t1 = std::chrono::steady_clock::now();
    h2d_secs = dir_secs(upF, upL, reps);
    d2h_secs = dir_secs(dnF, dnL, reps);
    return std::chrono::duration<double>(t1 - t0).count();
  };

  double dummy_h = 0.0, dummy_d = 0.0;
  for (int w = 0; w < kWarmups; ++w) run_window(1, dummy_h, dummy_d);
  const double t_once = run_window(1, dummy_h, dummy_d);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  double h2d_secs = 0.0, d2h_secs = 0.0;
  const double wall = run_window(reps, h2d_secs, d2h_secs);

  // --- verify both directions once, outside the timed window ---
  const bool dn_ok = bidir::verify_sample(static_cast<std::uint32_t*>(h_dn), N);
  std::vector<std::uint32_t> back(N);
  {
    zeCommandListReset(lUp);
    zeCommandListAppendMemoryCopy(lUp, back.data(), d_up, S, nullptr, 0, nullptr);
    zeCommandListClose(lUp);
    zeCommandQueueExecuteCommandLists(qUp, 1, &lUp, nullptr);
    zeCommandQueueSynchronize(qUp, UINT64_MAX);
  }
  const bool up_ok = bidir::verify_sample(back.data(), N);
  const bool ok = up_ok && dn_ok;
  if (!ok) r.error = up_ok ? "download verification failed" : "upload verification failed";

  zeEventDestroy(upF);
  zeEventDestroy(upL);
  zeEventDestroy(dnF);
  zeEventDestroy(dnL);
  zeEventPoolDestroy(epool);
  zeCommandListDestroy(lUp);
  zeCommandListDestroy(lDn);
  zeCommandQueueDestroy(qUp);
  zeCommandQueueDestroy(qDn);
  cleanup();

  const std::uint64_t work = 2ull * static_cast<std::uint64_t>(reps) * S;
  r.work = work;
  r.measured = std::chrono::duration<double>{wall};
  r.score = score_giga(work, wall);
  r.correct = ok;
  r.timings.copy_h2d = std::chrono::duration<double>{h2d_secs};
  r.timings.copy_h2d_size = static_cast<std::size_t>(reps) * S;
  r.timings.copy_d2h = std::chrono::duration<double>{d2h_secs};
  r.timings.copy_d2h_size = static_cast<std::size_t>(reps) * S;
  r.timings.total = r.measured;
  return r;
}

}  // namespace bench
