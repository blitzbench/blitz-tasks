/**
 * @file opencl_runner.cpp
 * @brief OpenCL fp16 throughput runner — packed half8 FMA chains.
 *
 * fp16 arithmetic in OpenCL C requires the cl_khr_fp16 extension. When the
 * device does not advertise it the row is a clean unsupported one. When present
 * we run the shared contracting chain over half8 vectors and score in GFLOPS.
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

// half8 dependent-FMA chains. Seeds match kernel_params.hpp seed_val() with
// lanes 0..7. Output written as float so no fp16 storage extension is needed.
constexpr const char* kKernelSource = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
inline half seed_h(uint t, int k, int l) {
    uint idx = (t + 7u*(uint)k + 13u*(uint)l) & 255u;
    return (half)(1.0f + (float)idx * (1.0f/256.0f));
}
__kernel void fp16_fma(__global float* out, const uint n, const uint iters,
                       const float mf, const float af) {
    uint t = get_global_id(0);
    if (t >= n) return;
    half m = (half)mf, a = (half)af;
    half8 acc[4];
    for (int k = 0; k < 4; ++k) {
        acc[k] = (half8)(seed_h(t,k,0), seed_h(t,k,1), seed_h(t,k,2), seed_h(t,k,3),
                         seed_h(t,k,4), seed_h(t,k,5), seed_h(t,k,6), seed_h(t,k,7));
    }
    half8 mv = (half8)(m), av = (half8)(a);
    for (uint i = 0; i < iters; ++i) {
        for (int u = 0; u < 8; ++u) {
            for (int k = 0; k < 4; ++k)
                acc[k] = fma(acc[k], mv, av);
        }
    }
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) {
        float8 f = convert_float8(acc[k]);
        s += f.s0+f.s1+f.s2+f.s3+f.s4+f.s5+f.s6+f.s7;
    }
    out[t] = s;
}
)CL";

}  // namespace

RunResult run_gpu_fp16_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.score_unit = "GFLOPS";
  r.path = "unsupported(no cl_khr_fp16)";

  cl_device_id device = find_cl_device(setup.device);
  if (!device) {
    r.error = "no OpenCL device matched " + setup.device.id();
    return r;
  }

  const std::string exts = cl_info_string(device, CL_DEVICE_EXTENSIONS);
  if (exts.find("cl_khr_fp16") == std::string::npos) {
    r.supported = false;  // clean unsupported, not an error
    return r;
  }
  r.path = "simd(half8 fma)";

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
  cl_kernel kernel = clCreateKernel(program, "fp16_fma", &err);
  if (!kernel) {
    r.error = "clCreateKernel err=" + std::to_string(err);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return r;
  }

  const std::uint32_t T = fp16::thread_count(setup.device);
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, T * sizeof(float), nullptr, &err);

  auto cleanup = [&]() {
    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
  };

  auto time_launches = [&](cl_uint n, cl_uint iters, float mf, float af, int reps) -> double {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(kernel, 1, sizeof(cl_uint), &n);
    clSetKernelArg(kernel, 2, sizeof(cl_uint), &iters);
    clSetKernelArg(kernel, 3, sizeof(cl_float), &mf);
    clSetKernelArg(kernel, 4, sizeof(cl_float), &af);
    const std::size_t local = fp16::kBlock;
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

  // --- pre-flight exactness (m=1,a=1) ---
  time_launches(fp16::kPreflightThreads, fp16::kPreflightIters, fp16::kPreM, fp16::kPreA, 1);
  clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, fp16::kPreflightThreads * sizeof(float), host.data(), 0, nullptr,
                      nullptr);
  for (std::uint32_t t = 0; t < fp16::kPreflightThreads; ++t) {
    const double e = fp16::simd_preflight_out(t, 8, fp16::kPreflightIters);
    if (!fp16::matches_exact(host[t], e)) {
      char b[160];
      std::snprintf(b, sizeof(b), "pre-flight mismatch at t=%u: got=%g expected=%g", t, host[t], e);
      r.error = b;
      cleanup();
      return r;
    }
  }

  // --- warmup + calibrate + timed run (contracting constants) ---
  for (int w = 0; w < kWarmups; ++w) time_launches(T, fp16::kIters, fp16::kMainM, fp16::kMainA, 1);
  const double t_once = time_launches(T, fp16::kIters, fp16::kMainM, fp16::kMainA, 1);
  const int reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
  const double secs = time_launches(T, fp16::kIters, fp16::kMainM, fp16::kMainA, reps);

  clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, T * sizeof(float), host.data(), 0, nullptr, nullptr);
  cleanup();

  bool ok = true;
  for (std::uint32_t s = 0; s < 64; ++s) {
    const std::uint32_t t = static_cast<std::uint32_t>(static_cast<std::uint64_t>(s) * (T - 1) / 63);
    if (!fp16::simd_main_ok(host[t], 8)) {
      char b[160];
      std::snprintf(b, sizeof(b), "sample out of envelope at t=%u: got=%g", t, host[t]);
      r.error = b;
      ok = false;
      break;
    }
  }

  r.work = fp16::simd_flops(T, fp16::kIters, reps, 8);
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  return r;
}

}  // namespace bench
