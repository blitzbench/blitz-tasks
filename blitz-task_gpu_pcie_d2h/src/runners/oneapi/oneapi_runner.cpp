/**
 * @file oneapi_runner.cpp
 * @brief Level Zero / oneAPI device-to-host transfer runner — SDK-required variant.
 *
 * A device allocation (zeMemAllocDevice, prefilled once) is copied into a host
 * allocation (zeMemAllocHost) with zeCommandListAppendMemoryCopy. The rep loop is
 * bracketed by kernel-timestamp events (scaled by the device timerResolution).
 * The received host allocation is sample-checked once. Mirror of the h2d runner.
 */

#include "oneapi_runner.hpp"

#include <level_zero/ze_api.h>

#include <algorithm>
#include <bench/config.hpp>
#include <bench/l0_utils.hpp>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace bench {

namespace {

constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t k2GiB = 2ull * 1024ull * 1024ull * 1024ull;

std::size_t transfer_bytes(const gpgpu::Device& d) {
  std::uint64_t mem = d.memory().value_or(k2GiB);
  std::uint64_t s = std::min<std::uint64_t>(256ull * kMiB, mem / 8ull);
  s &= ~std::uint64_t(3);
  if (s < 4) s = 4;
  return static_cast<std::size_t>(s);
}

inline std::uint32_t pattern_at(std::size_t i) { return static_cast<std::uint32_t>(i) * 2654435761u; }

bool sample_ok(const std::uint32_t* host, std::size_t count) {
  constexpr std::size_t kSamples = 64;
  if (count == 0) return false;
  for (std::size_t s = 0; s < kSamples; ++s) {
    std::size_t i = (count == 1) ? 0 : (s * (count - 1) / (kSamples - 1));
    if (host[i] != pattern_at(i)) return false;
  }
  return true;
}

}  // namespace

RunResult run_gpu_d2h_oneapi(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "device->host (zeMemoryCopy)";
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

  ze_device_properties_t dp{};
  dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  zeDeviceGetProperties(dev, &dp);
  const std::uint64_t timer_res_ns = dp.timerResolution;

  ze_context_desc_t ctx_desc{};
  ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t ctx = nullptr;
  if (zeContextCreate(drv, &ctx_desc, &ctx) != ZE_RESULT_SUCCESS) {
    r.error = "zeContextCreate failed";
    return r;
  }

  const std::size_t S = transfer_bytes(setup.device);
  const std::size_t count = S / sizeof(std::uint32_t);

  ze_device_mem_alloc_desc_t mad{};
  mad.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void* d_src = nullptr;
  zeMemAllocDevice(ctx, &mad, S, sizeof(std::uint32_t), dev, &d_src);

  ze_host_mem_alloc_desc_t had{};
  had.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  void* h_dst = nullptr;
  zeMemAllocHost(ctx, &had, S, sizeof(std::uint32_t), &h_dst);

  if (!d_src || !h_dst) {
    if (d_src) zeMemFree(ctx, d_src);
    if (h_dst) zeMemFree(ctx, h_dst);
    zeContextDestroy(ctx);
    r.error = "zeMemAlloc failed";
    return r;
  }

  ze_command_queue_desc_t cqd{};
  cqd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  cqd.ordinal = 0;
  cqd.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
  cqd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ze_command_queue_handle_t cqueue = nullptr;
  zeCommandQueueCreate(ctx, dev, &cqd, &cqueue);

  ze_command_list_desc_t cld{};
  cld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  cld.commandQueueGroupOrdinal = 0;
  ze_command_list_handle_t list = nullptr;
  zeCommandListCreate(ctx, dev, &cld, &list);

  // Prefill the device source with the pattern (untimed).
  {
    std::vector<std::uint32_t> seed(count);
    for (std::size_t i = 0; i < count; ++i) seed[i] = pattern_at(i);
    zeCommandListReset(list);
    zeCommandListAppendMemoryCopy(list, d_src, seed.data(), S, nullptr, 0, nullptr);
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(cqueue, 1, &list, nullptr);
    zeCommandQueueSynchronize(cqueue, UINT64_MAX);
  }

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

  auto time_copies = [&](int reps) -> double {
    zeCommandListReset(list);
    zeEventHostReset(ev_first);
    zeEventHostReset(ev_last);
    for (int i = 0; i < reps; ++i) {
      ze_event_handle_t sig = (i == 0) ? ev_first : (i == reps - 1 ? ev_last : nullptr);
      zeCommandListAppendMemoryCopy(list, h_dst, d_src, S, sig, 0, nullptr);
      zeCommandListAppendBarrier(list, nullptr, 0, nullptr);
    }
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(cqueue, 1, &list, nullptr);
    zeCommandQueueSynchronize(cqueue, UINT64_MAX);
    std::uint64_t s0 = 0, e0 = 0, s1 = 0, e1 = 0;
    l0_kernel_ticks(ev_first, s0, e0);
    l0_kernel_ticks(reps == 1 ? ev_first : ev_last, s1, e1);
    return (e1 > s0) ? l0_timestamp_seconds(e1 - s0, timer_res_ns) : 0.0;
  };

  for (int w = 0; w < kWarmups; ++w) time_copies(1);
  const double t_once = time_copies(1);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  const auto wall0 = std::chrono::steady_clock::now();
  double secs = time_copies(reps);
  const auto wall1 = std::chrono::steady_clock::now();
  const double wall = std::chrono::duration<double>(wall1 - wall0).count();
  if (secs <= 0.0) secs = wall;

  // --- verify once: the received host allocation ---
  const bool ok = sample_ok(static_cast<const std::uint32_t*>(h_dst), count);

  zeEventDestroy(ev_first);
  zeEventDestroy(ev_last);
  zeEventPoolDestroy(epool);
  zeMemFree(ctx, h_dst);
  zeMemFree(ctx, d_src);
  zeCommandListDestroy(list);
  zeCommandQueueDestroy(cqueue);
  zeContextDestroy(ctx);

  r.work = static_cast<std::uint64_t>(reps) * S;
  r.measured = std::chrono::duration<double>{secs};
  r.timings.copy_d2h = r.measured;
  r.timings.copy_d2h_size = S;
  r.timings.total = std::chrono::duration<double>{wall};
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  if (!ok) r.error = "verification failed";
  return r;
}

}  // namespace bench
