/**
 * @file oneapi_runner.cpp
 * @brief Level Zero / oneAPI device-local VRAM copy runner. Two device USM buffers; the
 *        source is filled once (untimed) from a host USM staging buffer. The timed loop
 *        records `reps` full-buffer zeCommandListAppendMemoryCopy(device->device)
 *        commands, serialized by barriers and bracketed by device global timestamps.
 *        work = 2 * reps * S (a copy reads and writes the buffer).
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

namespace bench {

RunResult run_gpu_vram_copy_oneapi(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "device copy (zeMemoryCopy D2D)";
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
  const std::uint32_t valid_bits = dev_props.timestampValidBits;
  const std::uint64_t ts_mask = (valid_bits >= 64) ? ~0ull : ((1ull << valid_bits) - 1);

  ze_context_desc_t ctx_desc{};
  ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t ctx = nullptr;
  if (zeContextCreate(drv, &ctx_desc, &ctx) != ZE_RESULT_SUCCESS) {
    r.error = "zeContextCreate failed";
    return r;
  }

  const std::size_t S = vram::buffer_bytes(setup.device);
  const std::uint32_t N = static_cast<std::uint32_t>(S / 4);

  ze_device_mem_alloc_desc_t mad{};
  mad.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  ze_host_mem_alloc_desc_t had{};
  had.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;

  void* d_src = nullptr;
  void* d_dst = nullptr;
  void* h_stage = nullptr;
  void* h_ts = nullptr;
  zeMemAllocDevice(ctx, &mad, S, 16, dev, &d_src);
  zeMemAllocDevice(ctx, &mad, S, 16, dev, &d_dst);
  zeMemAllocHost(ctx, &had, S, 16, &h_stage);
  zeMemAllocHost(ctx, &had, 2 * sizeof(std::uint64_t), 8, &h_ts);

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

  auto cleanup = [&]() {
    if (list) zeCommandListDestroy(list);
    if (queue) zeCommandQueueDestroy(queue);
    if (d_src) zeMemFree(ctx, d_src);
    if (d_dst) zeMemFree(ctx, d_dst);
    if (h_stage) zeMemFree(ctx, h_stage);
    if (h_ts) zeMemFree(ctx, h_ts);
    zeContextDestroy(ctx);
  };

  if (!d_src || !d_dst || !h_stage || !h_ts || !queue || !list) {
    r.error = "L0 allocation failed";
    cleanup();
    return r;
  }

  // --- fill source (untimed) ---
  {
    auto* p = static_cast<std::uint32_t*>(h_stage);
    for (std::uint32_t i = 0; i < N; ++i) p[i] = vram::pattern(i);
    zeCommandListReset(list);
    zeCommandListAppendMemoryCopy(list, d_src, h_stage, S, nullptr, 0, nullptr);
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX);
  }

  auto* ts = static_cast<std::uint64_t*>(h_ts);

  auto time_copies = [&](int reps) -> double {
    zeCommandListReset(list);
    zeCommandListAppendWriteGlobalTimestamp(list, &ts[0], nullptr, 0, nullptr);
    zeCommandListAppendBarrier(list, nullptr, 0, nullptr);
    for (int i = 0; i < reps; ++i) {
      zeCommandListAppendMemoryCopy(list, d_dst, d_src, S, nullptr, 0, nullptr);
      zeCommandListAppendBarrier(list, nullptr, 0, nullptr);
    }
    zeCommandListAppendWriteGlobalTimestamp(list, &ts[1], nullptr, 0, nullptr);
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX);
    const std::uint64_t delta = (ts[1] - ts[0]) & ts_mask;
    return l0_timestamp_seconds(delta, timer_res_ns);
  };

  for (int w = 0; w < kWarmups; ++w) time_copies(1);
  const double t_once = time_copies(1);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  const auto wall0 = std::chrono::steady_clock::now();
  const double secs = time_copies(reps);
  const auto wall1 = std::chrono::steady_clock::now();

  // --- verify: first + last 1 MiB window of dst (via host staging) ---
  bool ok = true;
  std::string verr;
  const std::uint32_t w = std::min<std::uint32_t>(vram::kWindowU32, N);
  auto check_window = [&](std::uint32_t start) {
    zeCommandListReset(list);
    zeCommandListAppendMemoryCopy(list, h_stage,
                                  static_cast<std::uint8_t*>(d_dst) + static_cast<std::size_t>(start) * 4,
                                  static_cast<std::size_t>(w) * 4, nullptr, 0, nullptr);
    zeCommandListClose(list);
    zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX);
    const auto* p = static_cast<const std::uint32_t*>(h_stage);
    for (std::uint32_t i = 0; i < w && ok; ++i) {
      const std::uint32_t idx = start + i;
      if (p[i] != vram::pattern(idx)) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "copy mismatch at u32[%u]: got=%u expected=%u", idx, p[i], vram::pattern(idx));
        verr = buf;
        ok = false;
      }
    }
  };
  check_window(0);
  if (ok && N > w) check_window(N - w);

  cleanup();

  r.work = 2ull * static_cast<std::uint64_t>(reps) * S;
  r.measured = std::chrono::duration<double>{secs};
  r.timings.copy_h2d = r.measured;
  r.timings.copy_h2d_size = static_cast<std::size_t>(reps) * S;
  r.timings.total = wall1 - wall0;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  if (!ok) r.error = verr;
  return r;
}

}  // namespace bench
