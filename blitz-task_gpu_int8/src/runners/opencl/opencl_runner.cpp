/**
 * @file opencl_runner.cpp
 * @brief OpenCL int8 throughput runner — SDK-required variant.
 *
 * Core OpenCL C has no portable int8 tensor/dot primitive, so this uses the
 * char4->int manual MAC path (path "simd(char4)"). Runtime-built kernel;
 * dependent MAC chains in registers, sum stored to global memory. Timed with
 * profiling events over a calibrated rep count. Unit GOPS.
 *
 * as_char4() reinterprets the 32-bit accumulator as four SIGNED int8 lanes,
 * exactly matching bench::i8::reference_packed, so outputs verify bit-for-bit.
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
__kernel void int8_packed(__global uint* out, const uint n, const uint iters) {
    uint t = get_global_id(0);
    if (t >= n) return;
    const uint W = 0x04030201u;
    char4 wb = as_char4(W);
    uint acc0 = 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * 1u;
    uint acc1 = 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * 2u;
    uint acc2 = 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * 3u;
    uint acc3 = 0x9E3779B9u * (t + 1u) + 0x85EBCA6Bu * 4u;
    for (uint i = 0; i < iters; ++i) {
        for (int u = 0; u < 8; ++u) {
            char4 b0 = as_char4(acc0);
            char4 b1 = as_char4(acc1);
            char4 b2 = as_char4(acc2);
            char4 b3 = as_char4(acc3);
            acc0 += (uint)(b0.x*wb.x + b0.y*wb.y + b0.z*wb.z + b0.w*wb.w);
            acc1 += (uint)(b1.x*wb.x + b1.y*wb.y + b1.z*wb.z + b1.w*wb.w);
            acc2 += (uint)(b2.x*wb.x + b2.y*wb.y + b2.z*wb.z + b2.w*wb.w);
            acc3 += (uint)(b3.x*wb.x + b3.y*wb.y + b3.z*wb.z + b3.w*wb.w);
        }
    }
    out[t] = acc0 + acc1 + acc2 + acc3;
}
)CL";

}  // namespace

RunResult run_gpu_int8_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "simd(char4)";
  r.score_unit = "GOPS";

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
  cl_kernel kernel = clCreateKernel(program, "int8_packed", &err);
  if (!kernel) {
    r.error = "clCreateKernel err=" + std::to_string(err);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return r;
  }

  const std::uint32_t T = i8::thread_count(setup.device);
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, T * sizeof(std::uint32_t), nullptr, &err);

  auto cleanup = [&]() {
    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
  };

  auto time_launches = [&](cl_uint n, cl_uint iters, int reps) -> double {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(kernel, 1, sizeof(cl_uint), &n);
    clSetKernelArg(kernel, 2, sizeof(cl_uint), &iters);
    const std::size_t local = i8::kBlock;
    const std::size_t global = ((n + local - 1) / local) * local;
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

  std::vector<std::uint32_t> host(T);

  auto verify = [&](std::uint32_t count, std::uint32_t iters, std::uint32_t stride) -> bool {
    for (std::uint32_t s = 0; s < count; ++s) {
      const std::uint32_t t = s * stride;
      const std::uint32_t exp = i8::reference_packed(t, iters);
      if (host[t] != exp) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "mismatch at t=%u: got=%u expected=%u", t, host[t], exp);
        r.error = buf;
        return false;
      }
    }
    return true;
  };

  // --- pre-flight exactness ---
  time_launches(i8::kPreflightThreads, i8::kPreflightIters, 1);
  clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, i8::kPreflightThreads * sizeof(std::uint32_t), host.data(), 0, nullptr,
                      nullptr);
  if (!verify(i8::kPreflightThreads, i8::kPreflightIters, 1)) {
    cleanup();
    return r;
  }

  // --- warmup + calibrate ---
  for (int w = 0; w < kWarmups; ++w) time_launches(T, i8::kIters, 1);
  const double t_once = time_launches(T, i8::kIters, 1);
  const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);

  // --- timed run ---
  const double secs = time_launches(T, i8::kIters, reps);

  clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, T * sizeof(std::uint32_t), host.data(), 0, nullptr, nullptr);
  cleanup();

  // --- post-run sampled verification ---
  constexpr std::uint32_t kSamples = 64;
  const std::uint32_t stride = (T - 1) / (kSamples - 1);
  const bool ok = verify(kSamples, i8::kIters, stride);

  r.work = i8::ops_packed(T, i8::kIters, reps);
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  return r;
}

}  // namespace bench
