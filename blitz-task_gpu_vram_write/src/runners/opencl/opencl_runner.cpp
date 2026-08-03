/**
 * @file opencl_runner.cpp
 * @brief OpenCL device-local VRAM write runner. Runtime-built grid-stride kernel of
 *        uint4 stores writing pattern(i) = i * 2654435761u for every u32 element.
 *        Timed with profiling events over a calibrated rep count. work = reps * S.
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

namespace {

constexpr const char* kKernelSource = R"CL(
__kernel void vram_write(__global uint4* data, const uint n_vec4) {
    const uint M = 2654435761u;
    uint stride = get_global_size(0);
    for (uint v = get_global_id(0); v < n_vec4; v += stride) {
        uint base = v * 4u;
        data[v] = (uint4)(base * M, (base + 1u) * M, (base + 2u) * M, (base + 3u) * M);
    }
}
)CL";

}  // namespace

RunResult run_gpu_vram_write_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "simd(uint4 stores)";
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

  std::string log;
  cl_program program = build_program_with_log(ctx, device, kKernelSource, nullptr, log);
  if (!program) {
    r.error = "clBuildProgram failed: " + log;
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return r;
  }
  cl_kernel kernel = clCreateKernel(program, "vram_write", &err);
  if (!kernel) {
    r.error = "clCreateKernel err=" + std::to_string(err);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return r;
  }

  const std::size_t S = vram::buffer_bytes(setup.device);
  const std::uint32_t N = static_cast<std::uint32_t>(S / 4);
  const cl_uint n_vec4 = N / 4;

  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S, nullptr, &err);

  auto cleanup = [&]() {
    if (buf) clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
  };
  if (!buf) {
    r.error = "clCreateBuffer failed";
    cleanup();
    return r;
  }

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
  clSetKernelArg(kernel, 1, sizeof(cl_uint), &n_vec4);

  const std::size_t local = vram::kBlock;
  const std::size_t global = ((vram::thread_count(setup.device) + local - 1) / local) * local;

  auto time_writes = [&](int reps) -> double {
    cl_event first = nullptr, last = nullptr;
    for (int i = 0; i < reps; ++i) {
      cl_event ev = nullptr;
      clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &ev);
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

  for (int w = 0; w < kWarmups; ++w) time_writes(1);
  const double t_once = time_writes(1);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  const auto wall0 = std::chrono::steady_clock::now();
  const double secs = time_writes(reps);
  const auto wall1 = std::chrono::steady_clock::now();

  // --- verify: first + last 1 MiB window ---
  bool ok = true;
  std::string verr;
  const std::uint32_t w = std::min<std::uint32_t>(vram::kWindowU32, N);
  std::vector<std::uint32_t> chk(w);
  auto check_window = [&](std::uint32_t start) {
    clEnqueueReadBuffer(queue, buf, CL_TRUE, static_cast<std::size_t>(start) * 4, static_cast<std::size_t>(w) * 4,
                        chk.data(), 0, nullptr, nullptr);
    for (std::uint32_t i = 0; i < w && ok; ++i) {
      const std::uint32_t idx = start + i;
      if (chk[i] != vram::pattern(idx)) {
        char buf128[128];
        std::snprintf(buf128, sizeof(buf128), "write mismatch at u32[%u]: got=%u expected=%u", idx, chk[i],
                      vram::pattern(idx));
        verr = buf128;
        ok = false;
      }
    }
  };
  check_window(0);
  if (ok && N > w) check_window(N - w);

  cleanup();

  r.work = static_cast<std::uint64_t>(reps) * S;
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.timings.total = wall1 - wall0;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  if (!ok) r.error = verr;
  return r;
}

}  // namespace bench
