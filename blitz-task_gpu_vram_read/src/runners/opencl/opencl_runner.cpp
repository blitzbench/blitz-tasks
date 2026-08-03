/**
 * @file opencl_runner.cpp
 * @brief OpenCL device-local VRAM read runner. Buffer prefilled (untimed) with the
 *        pattern; runtime-built grid-stride uint4-load kernel accumulates per-thread
 *        (u32 wraparound) and writes one uint per thread. Timed with profiling events.
 *        work = reps * S. Verified against the closed-form checksum.
 */

#include "opencl_runner.hpp"

#include <CL/cl.h>

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
__kernel void vram_read(__global const uint4* data, const uint n_vec4, __global uint* out) {
    uint gid    = get_global_id(0);
    uint stride = get_global_size(0);
    uint acc = 0u;
    for (uint v = gid; v < n_vec4; v += stride) {
        uint4 x = data[v];
        acc += x.x + x.y + x.z + x.w;
    }
    out[gid] = acc;
}
)CL";

}  // namespace

RunResult run_gpu_vram_read_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "simd(uint4 loads)";
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
  cl_kernel kernel = clCreateKernel(program, "vram_read", &err);
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
  const std::size_t local = vram::kBlock;
  const std::size_t global = ((vram::thread_count(setup.device) + local - 1) / local) * local;

  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, S, nullptr, &err);
  cl_mem out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, global * 4, nullptr, &err);

  auto cleanup = [&]() {
    if (buf) clReleaseMemObject(buf);
    if (out) clReleaseMemObject(out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
  };
  if (!buf || !out) {
    r.error = "clCreateBuffer failed";
    cleanup();
    return r;
  }

  // --- prefill buffer with the pattern (untimed) ---
  std::vector<std::uint32_t> host(N);
  for (std::uint32_t i = 0; i < N; ++i) host[i] = vram::pattern(i);
  clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, S, host.data(), 0, nullptr, nullptr);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
  clSetKernelArg(kernel, 1, sizeof(cl_uint), &n_vec4);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &out);

  auto time_reads = [&](int reps) -> double {
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

  for (int w = 0; w < kWarmups; ++w) time_reads(1);
  const double t_once = time_reads(1);
  const int reps = calibrate_repeats(t_once, kTransferTargetSeconds, kTransferRepCap);

  const auto wall0 = std::chrono::steady_clock::now();
  const double secs = time_reads(reps);
  const auto wall1 = std::chrono::steady_clock::now();

  // --- verify: exact closed-form checksum ---
  bool ok = true;
  std::string verr;
  std::vector<std::uint32_t> outv(global);
  clEnqueueReadBuffer(queue, out, CL_TRUE, 0, global * 4, outv.data(), 0, nullptr, nullptr);
  std::uint32_t sum = 0u;
  for (std::size_t i = 0; i < global; ++i) sum += outv[i];
  const std::uint32_t expected = vram::expected_sum(S);
  if (sum != expected) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "read checksum mismatch: got=%u expected=%u", sum, expected);
    verr = buf;
    ok = false;
  }

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
