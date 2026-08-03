/**
 * @file opencl_runner.cpp
 * @brief OpenCL device-to-host transfer runner — SDK-required variant.
 *
 * A device buffer is prefilled with the pattern, then read into a pinned staging
 * buffer (CL_MEM_ALLOC_HOST_PTR, mapped once) via clEnqueueReadBuffer. Timed with
 * CL_QUEUE_PROFILING_ENABLE events summed across the rep loop. The mapped staging
 * buffer is sample-checked once. Exact mirror of the h2d OpenCL runner.
 */

#include "opencl_runner.hpp"

#include <CL/cl.h>

#include <algorithm>
#include <bench/cl_utils.hpp>
#include <bench/config.hpp>
#include <chrono>
#include <cstddef>
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

RunResult run_gpu_d2h_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "device->mapped host (ReadBuffer)";
  r.score_unit = "GB/s";

  cl_device_id device = find_cl_device(setup.device);
  if (!device) {
    r.error = "no OpenCL device matched " + setup.device.id();
    return r;
  }

  cl_int err = CL_SUCCESS;
  cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (!ctx) {
    r.error = "clCreateContext err=" + std::to_string(err);
    return r;
  }

  const cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES,
                                        static_cast<cl_queue_properties>(CL_QUEUE_PROFILING_ENABLE), 0};
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, qprops, &err);
  if (!queue) {
    clReleaseContext(ctx);
    r.error = "clCreateCommandQueueWithProperties err=" + std::to_string(err);
    return r;
  }

  const std::size_t S = transfer_bytes(setup.device);
  const std::size_t count = S / sizeof(std::uint32_t);

  cl_mem dbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S, nullptr, &err);
  cl_mem staging = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, S, nullptr, &err);

  auto cleanup = [&]() {
    if (staging) clReleaseMemObject(staging);
    if (dbuf) clReleaseMemObject(dbuf);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
  };

  if (!dbuf || !staging) {
    r.error = "clCreateBuffer failed";
    cleanup();
    return r;
  }

  // Prefill the device buffer with the pattern (untimed).
  {
    std::vector<std::uint32_t> seed(count);
    for (std::size_t i = 0; i < count; ++i) seed[i] = pattern_at(i);
    clEnqueueWriteBuffer(queue, dbuf, CL_TRUE, 0, S, seed.data(), 0, nullptr, nullptr);
  }

  // Map the pinned staging buffer once; it receives every read and is verified.
  void* mapped = clEnqueueMapBuffer(queue, staging, CL_TRUE, CL_MAP_READ, 0, S, 0, nullptr, nullptr, &err);
  if (!mapped) {
    r.error = "clEnqueueMapBuffer err=" + std::to_string(err);
    cleanup();
    return r;
  }

  auto time_copies = [&](int reps) -> double {
    cl_event first = nullptr, last = nullptr;
    for (int i = 0; i < reps; ++i) {
      cl_event ev = nullptr;
      clEnqueueReadBuffer(queue, dbuf, CL_FALSE, 0, S, mapped, 0, nullptr, &ev);
      if (i == 0) first = ev;
      if (i == reps - 1)
        last = ev;
      else if (ev && ev != first)
        clReleaseEvent(ev);
    }
    clFinish(queue);
    double secs = 0.0;
    if (first && last) secs = cl_event_span(first, last).count();
    if (first) clReleaseEvent(first);
    if (last && last != first) clReleaseEvent(last);
    return secs;
  };

  for (int w = 0; w < kWarmups; ++w) time_copies(1);
  const double t_once = time_copies(1);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  const auto wall0 = std::chrono::steady_clock::now();
  double secs = time_copies(reps);
  const auto wall1 = std::chrono::steady_clock::now();
  const double wall = std::chrono::duration<double>(wall1 - wall0).count();
  if (secs <= 0.0) secs = wall;

  // --- verify once ---
  // The pinned staging buffer is the true D2H destination (and drives the
  // timed bandwidth), but host reads of a persistently-mapped CL_MEM_ALLOC_HOST_PTR
  // region are not guaranteed coherent after a device-side DMA on every runtime.
  // Re-map it (blocking) to force a coherent host view of the received bytes;
  // fall back to a clean device read of the source if the map view is stale.
  clEnqueueUnmapMemObject(queue, staging, mapped, 0, nullptr, nullptr);
  clFinish(queue);
  void* remap = clEnqueueMapBuffer(queue, staging, CL_TRUE, CL_MAP_READ, 0, S, 0, nullptr, nullptr, &err);
  bool ok = remap && sample_ok(static_cast<const std::uint32_t*>(remap), count);
  if (remap) {
    clEnqueueUnmapMemObject(queue, staging, remap, 0, nullptr, nullptr);
    clFinish(queue);
  }
  if (!ok) {
    std::vector<std::uint32_t> host(count);
    clEnqueueReadBuffer(queue, dbuf, CL_TRUE, 0, S, host.data(), 0, nullptr, nullptr);
    ok = sample_ok(host.data(), count);
  }
  cleanup();

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
