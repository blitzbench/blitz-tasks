/**
 * @file opencl_runner.cpp
 * @brief OpenCL device-local VRAM copy runner. Two device buffers, src filled once
 *        (untimed), timed loop of full-buffer clEnqueueCopyBuffer with profiling
 *        events. work = 2 * reps * S (a copy reads and writes the buffer).
 */

#include "opencl_runner.hpp"

#include <CL/cl.h>

#include <algorithm>
#include <bench/cl_utils.hpp>
#include <bench/config.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../../kernel_params.hpp"

namespace bench {

RunResult run_gpu_vram_copy_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "device copy (clEnqueueCopyBuffer)";
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

  const std::size_t S = vram::buffer_bytes(setup.device);
  const std::uint32_t N = static_cast<std::uint32_t>(S / 4);

  cl_mem src = clCreateBuffer(ctx, CL_MEM_READ_ONLY, S, nullptr, &err);
  cl_mem dst = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, S, nullptr, &err);

  auto cleanup = [&]() {
    if (src) clReleaseMemObject(src);
    if (dst) clReleaseMemObject(dst);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
  };

  if (!src || !dst) {
    r.error = "clCreateBuffer failed";
    cleanup();
    return r;
  }

  // --- fill source (untimed) ---
  std::vector<std::uint32_t> host(N);
  for (std::uint32_t i = 0; i < N; ++i) host[i] = vram::pattern(i);
  clEnqueueWriteBuffer(queue, src, CL_TRUE, 0, S, host.data(), 0, nullptr, nullptr);

  auto time_copies = [&](int reps) -> double {
    cl_event first = nullptr, last = nullptr;
    for (int i = 0; i < reps; ++i) {
      cl_event ev = nullptr;
      clEnqueueCopyBuffer(queue, src, dst, 0, 0, S, 0, nullptr, &ev);
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
  const double secs = time_copies(reps);
  const auto wall1 = std::chrono::steady_clock::now();

  // --- verify: first + last 1 MiB window of dst ---
  bool ok = true;
  std::string verr;
  const std::uint32_t w = std::min<std::uint32_t>(vram::kWindowU32, N);
  std::vector<std::uint32_t> chk(w);
  auto check_window = [&](std::uint32_t start) {
    clEnqueueReadBuffer(queue, dst, CL_TRUE, static_cast<std::size_t>(start) * 4, static_cast<std::size_t>(w) * 4,
                        chk.data(), 0, nullptr, nullptr);
    for (std::uint32_t i = 0; i < w && ok; ++i) {
      const std::uint32_t idx = start + i;
      if (chk[i] != vram::pattern(idx)) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "copy mismatch at u32[%u]: got=%u expected=%u", idx, chk[i],
                      vram::pattern(idx));
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
