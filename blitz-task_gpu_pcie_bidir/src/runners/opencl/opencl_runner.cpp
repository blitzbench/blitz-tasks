/**
 * @file opencl_runner.cpp
 * @brief OpenCL concurrent bidirectional transfer runner.
 *
 * Two command queues share one context/device: queue A uploads (host -> device
 * via clEnqueueWriteBuffer), queue B downloads (device -> host via
 * clEnqueueReadBuffer), one direction each, running concurrently. Score is
 * aggregate wall throughput over a single host clock window bracketing first
 * enqueue -> clFinish on both queues. Per-direction profiling-event spans feed
 * the timings block only. path = "2 queues".
 */

#include "opencl_runner.hpp"

#include <CL/cl.h>

#include <bench/cl_utils.hpp>
#include <bench/config.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../../bidir_params.hpp"

namespace bench {

RunResult run_gpu_bidir_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "2 queues";
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
  cl_command_queue qUp = clCreateCommandQueueWithProperties(ctx, device, qprops, &err);
  cl_command_queue qDn = clCreateCommandQueueWithProperties(ctx, device, qprops, &err);
  if (!qUp || !qDn) {
    if (qUp) clReleaseCommandQueue(qUp);
    if (qDn) clReleaseCommandQueue(qDn);
    clReleaseContext(ctx);
    r.error = "clCreateCommandQueueWithProperties failed";
    return r;
  }

  const std::size_t S = bidir::buffer_bytes(setup.device);
  const std::size_t N = bidir::elem_count(S);

  cl_mem up_dst = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S, nullptr, &err);
  cl_mem dn_src = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S, nullptr, &err);

  auto cleanup = [&]() {
    if (up_dst) clReleaseMemObject(up_dst);
    if (dn_src) clReleaseMemObject(dn_src);
    clReleaseCommandQueue(qUp);
    clReleaseCommandQueue(qDn);
    clReleaseContext(ctx);
  };
  if (!up_dst || !dn_src) {
    r.error = "clCreateBuffer failed";
    cleanup();
    return r;
  }

  std::vector<std::uint32_t> host_up(N), host_dn(N, 0);
  bidir::fill_pattern(host_up.data(), N);
  // Seed the device download source with the pattern (blocking, pre-window).
  clEnqueueWriteBuffer(qUp, dn_src, CL_TRUE, 0, S, host_up.data(), 0, nullptr, nullptr);

  auto run_window = [&](int reps, double& h2d_secs, double& d2h_secs) -> double {
    cl_event upFirst = nullptr, upLast = nullptr, dnFirst = nullptr, dnLast = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
      cl_event ev = nullptr;
      clEnqueueWriteBuffer(qUp, up_dst, CL_FALSE, 0, S, host_up.data(), 0, nullptr, &ev);
      if (i == 0) upFirst = ev;
      if (i == reps - 1)
        upLast = ev;
      else if (ev && ev != upFirst)
        clReleaseEvent(ev);
    }
    for (int i = 0; i < reps; ++i) {
      cl_event ev = nullptr;
      clEnqueueReadBuffer(qDn, dn_src, CL_FALSE, 0, S, host_dn.data(), 0, nullptr, &ev);
      if (i == 0) dnFirst = ev;
      if (i == reps - 1)
        dnLast = ev;
      else if (ev && ev != dnFirst)
        clReleaseEvent(ev);
    }
    clFlush(qUp);
    clFlush(qDn);
    clFinish(qUp);
    clFinish(qDn);
    const auto t1 = std::chrono::steady_clock::now();
    h2d_secs = (upFirst && upLast) ? cl_event_span(upFirst, upLast).count() : 0.0;
    d2h_secs = (dnFirst && dnLast) ? cl_event_span(dnFirst, dnLast).count() : 0.0;
    if (upFirst) clReleaseEvent(upFirst);
    if (upLast && upLast != upFirst) clReleaseEvent(upLast);
    if (dnFirst) clReleaseEvent(dnFirst);
    if (dnLast && dnLast != dnFirst) clReleaseEvent(dnLast);
    return std::chrono::duration<double>(t1 - t0).count();
  };

  double dummy_h = 0.0, dummy_d = 0.0;
  for (int w = 0; w < kWarmups; ++w) run_window(1, dummy_h, dummy_d);
  const double t_once = run_window(1, dummy_h, dummy_d);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  double h2d_secs = 0.0, d2h_secs = 0.0;
  const double wall = run_window(reps, h2d_secs, d2h_secs);

  // --- verify both directions once, outside the timed window ---
  std::vector<std::uint32_t> back(N);
  clEnqueueReadBuffer(qUp, up_dst, CL_TRUE, 0, S, back.data(), 0, nullptr, nullptr);
  const bool up_ok = bidir::verify_sample(back.data(), N);     // upload landed?
  const bool dn_ok = bidir::verify_sample(host_dn.data(), N);  // download landed?
  const bool ok = up_ok && dn_ok;
  if (!ok) r.error = up_ok ? "download verification failed" : "upload verification failed";

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
