/**
 * @file opencl_runner.cpp
 * @brief OpenCL fp32 dense-FMA runner — SDK-required variant.
 *
 * Runtime-built kernel source; dependent fma() chains in registers, sum stored
 * to global memory. Timed with profiling events over a calibrated rep count.
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
__kernel void fp32_fma(__global float* out, const uint n, const uint iters) {
    uint t = get_global_id(0);
    if (t >= n) return;
    float acc0 = 1.0f + (float)((t + 0u) & 1023u) * (1.0f / 2048.0f);
    float acc1 = 1.0f + (float)((t + 1u) & 1023u) * (1.0f / 2048.0f);
    float acc2 = 1.0f + (float)((t + 2u) & 1023u) * (1.0f / 2048.0f);
    float acc3 = 1.0f + (float)((t + 3u) & 1023u) * (1.0f / 2048.0f);
    const float m = 0.9999f, add = 0.0001f;
    for (uint i = 0; i < iters; ++i) {
        for (int u = 0; u < 8; ++u) {
            acc0 = fma(acc0, m, add);
            acc1 = fma(acc1, m, add);
            acc2 = fma(acc2, m, add);
            acc3 = fma(acc3, m, add);
        }
    }
    out[t] = acc0 + acc1 + acc2 + acc3;
}
)CL";

}  // namespace

RunResult run_gpu_fp32_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "simd(fp32 fma)";
  r.score_unit = "GFLOPS";

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
  // No fast-math: fma() is correctly-rounded by spec and the seed arithmetic
  // must stay exact so the CPU reference reproduces device output.
  cl_program program = build_program_with_log(ctx, device, kKernelSource, nullptr, log);
  if (!program) {
    r.error = "clBuildProgram failed: " + log;
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return r;
  }
  cl_kernel kernel = clCreateKernel(program, "fp32_fma", &err);
  if (!kernel) {
    r.error = "clCreateKernel err=" + std::to_string(err);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return r;
  }

  const std::uint32_t T = fp32::thread_count(setup.device);
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, T * sizeof(float), nullptr, &err);

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
    const std::size_t local = fp32::kBlock;
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

  std::vector<float> host(T);

  // --- pre-flight exactness ---
  time_launches(fp32::kPreflightThreads, fp32::kPreflightIters, 1);
  clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, fp32::kPreflightThreads * sizeof(float), host.data(), 0, nullptr,
                      nullptr);
  for (std::uint32_t t = 0; t < fp32::kPreflightThreads; ++t) {
    const float e = fp32::reference(t, fp32::kPreflightIters);
    if (!fp32::matches(host[t], e)) {
      char buf160[160];
      std::snprintf(buf160, sizeof(buf160), "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t], e);
      r.error = buf160;
      cleanup();
      return r;
    }
  }

  // --- warmup + calibrate ---
  for (int w = 0; w < kWarmups; ++w) time_launches(T, fp32::kIters, 1);
  const double t_once = time_launches(T, fp32::kIters, 1);
  const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);

  // --- timed run ---
  const double secs = time_launches(T, fp32::kIters, reps);

  clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, T * sizeof(float), host.data(), 0, nullptr, nullptr);
  cleanup();

  // --- post-run sampled verification ---
  bool ok = true;
  constexpr std::uint32_t kSamples = 64;
  for (std::uint32_t s = 0; s < kSamples; ++s) {
    const std::uint32_t t = static_cast<std::uint32_t>(static_cast<std::uint64_t>(s) * (T - 1) / (kSamples - 1));
    const float e = fp32::reference(t, fp32::kIters);
    if (!fp32::matches(host[t], e)) {
      char buf160[160];
      std::snprintf(buf160, sizeof(buf160), "sample mismatch at t=%u: got=%g expected=%g", t, host[t], e);
      r.error = buf160;
      ok = false;
      break;
    }
  }

  r.work = fp32::flops(T, fp32::kIters, reps);
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  return r;
}

}  // namespace bench
