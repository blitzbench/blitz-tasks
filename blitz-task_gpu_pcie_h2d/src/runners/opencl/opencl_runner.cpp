/**
 * @file opencl_runner.cpp
 * @brief OpenCL host-to-device transfer runner.
 *
 * A pinned staging buffer (CL_MEM_ALLOC_HOST_PTR) is mapped once and its mapped
 * pointer is used as the host source of clEnqueueWriteBuffer into a separate
 * device buffer. Timed with CL_QUEUE_PROFILING_ENABLE events summed across the
 * rep loop. Verified once by reading the device buffer back.
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

RunResult run_gpu_h2d_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "mapped host->device (WriteBuffer)";
  r.score_unit = "GB/s";

  const gpgpu::vendor::OpenClFns* cl_fns = gpgpu::vendor::opencl();
  if (!cl_fns) {
    r.error = "OpenCL loader unavailable";
    return r;
  }
  const gpgpu::vendor::OpenClFns& cl = *cl_fns;

  cl_device_id device = find_cl_device(cl, setup.device);
  if (!device) {
    r.error = "no OpenCL device matched " + setup.device.id();
    return r;
  }

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl.clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (!ctx) {
    r.error = "clCreateContext err=" + std::to_string(err);
    return r;
  }

  const cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES,
                                        static_cast<cl_queue_properties>(CL_QUEUE_PROFILING_ENABLE), 0};
  cl_command_queue queue = cl.clCreateCommandQueueWithProperties(ctx, device, qprops, &err);
  if (!queue) {
    cl.clReleaseContext(ctx);
    r.error = "clCreateCommandQueueWithProperties err=" + std::to_string(err);
    return r;
  }

  const std::size_t S = transfer_bytes(setup.device);
  const std::size_t count = S / sizeof(std::uint32_t);

  cl_mem staging = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, S, nullptr, &err);
  cl_mem dbuf = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE, S, nullptr, &err);

  auto cleanup = [&]() {
    if (dbuf) cl.clReleaseMemObject(dbuf);
    if (staging) cl.clReleaseMemObject(staging);
    cl.clReleaseCommandQueue(queue);
    cl.clReleaseContext(ctx);
  };

  if (!staging || !dbuf) {
    r.error = "clCreateBuffer failed";
    cleanup();
    return r;
  }

  // Map the pinned staging buffer once and fill it with the pattern; keep it
  // mapped so its host pointer serves as the WriteBuffer source.
  void* mapped = cl.clEnqueueMapBuffer(queue, staging, CL_TRUE, CL_MAP_WRITE, 0, S, 0, nullptr, nullptr, &err);
  if (!mapped) {
    r.error = "clEnqueueMapBuffer err=" + std::to_string(err);
    cleanup();
    return r;
  }
  {
    std::uint32_t* p = static_cast<std::uint32_t*>(mapped);
    for (std::size_t i = 0; i < count; ++i) p[i] = pattern_at(i);
  }

  auto time_copies = [&](int reps) -> double {
    cl_event first = nullptr, last = nullptr;
    for (int i = 0; i < reps; ++i) {
      cl_event ev = nullptr;
      cl.clEnqueueWriteBuffer(queue, dbuf, CL_FALSE, 0, S, mapped, 0, nullptr, &ev);
      if (i == 0) first = ev;
      if (i == reps - 1)
        last = ev;
      else if (ev && ev != first)
        cl.clReleaseEvent(ev);
    }
    cl.clFinish(queue);
    double secs = 0.0;
    if (first && last) secs = cl_event_span(cl, first, last).count();
    if (first) cl.clReleaseEvent(first);
    if (last && last != first) cl.clReleaseEvent(last);
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
  std::vector<std::uint32_t> host(count);
  cl.clEnqueueReadBuffer(queue, dbuf, CL_TRUE, 0, S, host.data(), 0, nullptr, nullptr);
  const bool ok = sample_ok(host.data(), count);

  cl.clEnqueueUnmapMemObject(queue, staging, mapped, 0, nullptr, nullptr);
  cl.clFinish(queue);
  cleanup();

  r.work = static_cast<std::uint64_t>(reps) * S;
  r.measured = std::chrono::duration<double>{secs};
  r.timings.copy_h2d = r.measured;
  r.timings.copy_h2d_size = S;
  r.timings.total = std::chrono::duration<double>{wall};
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  if (!ok) r.error = "verification failed";
  return r;
}

}  // namespace bench
