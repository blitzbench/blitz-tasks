/**
 * @file opencl_runner.cpp
 * @brief OpenCL int8 throughput runner — Intel XMX tensor path with a char4 MAC fallback.
 *
 * On devices with cl_intel_subgroup_matrix_multiply_accumulate (Intel Arc and newer)
 * the tensor kernel drives the XMX units through intel_sub_group_i8_i8_matrix_mad_k32:
 * one 8 x SG x 32 {s8,s8,s32} multiply-accumulate per call and sub-group, SG being the
 * device's minimum sub-group size (8 or 16, bench::intel_matrix_subgroup_size). A and B
 * are all ones, passed in as a kernel argument so no compiler can fold the products, so
 * every C element equals 32*iters (bench::i8::reference_tensor). Should the tensor kernel
 * fail to build or verify, the packed kernel runs instead.
 *
 * Core OpenCL C has no portable int8 tensor/dot primitive, so everywhere else this uses
 * the char4->int manual MAC path. Runtime-built kernel; dependent MAC chains in
 * registers, sum stored to global memory. Timed with profiling events over a calibrated
 * rep count. Unit GOPS. as_char4() reinterprets the 32-bit accumulator as four SIGNED
 * int8 lanes, exactly matching bench::i8::reference_packed, so outputs verify bit-for-bit.
 *
 *   tensor:  "tensor(xmx 8xSGx32 s8->s32)"
 *   packed:  "simd(char4)"
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

// XMX tensor kernel, built with -DSG=8 or -DSG=16. Each call multiplies an 8x32 A by a
// 32xSG B: A packs four int8 per int on SG8 and two per short on SG16, B packs a work item's
// 32-element column into an int8. Four independent accumulators keep the matrix pipe busy,
// and every component is summed so no call can be dropped as dead code.
constexpr const char* kTensorSource = R"CL(
#if SG == 8
typedef int8 a_t;
#define A_INIT(p) ((a_t)((int)(p)))
#else
typedef short8 a_t;
#define A_INIT(p) ((a_t)((short)((p) & 0xFFFFu)))
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void int8_xmx(__global uint* out, const uint iters, const uint ones) {
    const a_t a = A_INIT(ones);
    const int8 b = (int8)((int)ones);
    int8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (uint i = 0; i < iters / 4u; ++i) {
        c0 = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c0);
        c1 = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c1);
        c2 = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c2);
        c3 = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c3);
    }
    const int8 s = c0 + c1 + c2 + c3;
    out[get_global_id(0)] = (uint)(s.s0 + s.s1 + s.s2 + s.s3 + s.s4 + s.s5 + s.s6 + s.s7) / 8u;
}
)CL";

// Four int8 ones packed into one 32-bit word.
constexpr cl_uint kTensorOnes = 0x01010101u;
constexpr int kTensorRows = 8;
constexpr int kTensorK = 32;
static_assert(i8::kIters % 4 == 0 && i8::kPreflightIters % 4 == 0,
              "tensor iteration counts must split evenly over the accumulator chains");

/**
 * @brief Runs the XMX tensor kernel on @p buf (T uints).
 *
 * @param cl OpenCL entry points
 * @param ctx context of @p device
 * @param device device to run on
 * @param queue profiling-enabled queue
 * @param buf output buffer of @p T uints
 * @param T launched work items
 * @param sg sub-group size the built-ins require (8 or 16)
 * @param r filled with the scored result on success
 * @return true when the kernel built, ran and verified
 */
bool run_tensor(const gpgpu::vendor::OpenClFns& cl, cl_context ctx, cl_device_id device, cl_command_queue queue,
                cl_mem buf, std::uint32_t T, std::uint32_t sg, RunResult& r) {
  const std::string options = "-DSG=" + std::to_string(sg);
  std::string log;
  cl_program program = build_program_with_log(cl, ctx, device, kTensorSource, options.c_str(), log);
  if (!program) return false;
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = cl.clCreateKernel(program, "int8_xmx", &err);
  if (!kernel) {
    cl.clReleaseProgram(program);
    return false;
  }

  auto launch = [&](std::uint32_t n, cl_uint iters, int reps) {
    const cl_uint ones = kTensorOnes;
    cl.clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    cl.clSetKernelArg(kernel, 1, sizeof(cl_uint), &iters);
    cl.clSetKernelArg(kernel, 2, sizeof(cl_uint), &ones);
    return cl_time_launches(cl, queue, kernel, n, i8::kBlock, reps);
  };

  auto verify = [&](const std::vector<std::uint32_t>& host, std::uint32_t count, std::uint32_t iters,
                    std::uint32_t stride) {
    const auto exp = static_cast<std::uint32_t>(i8::reference_tensor(iters, kTensorK));
    for (std::uint32_t s = 0; s < count; ++s)
      if (host[s * stride] != exp) return false;
    return true;
  };

  std::vector<std::uint32_t> host(T);
  launch(i8::kPreflightThreads, i8::kPreflightIters, 1);
  cl.clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, i8::kPreflightThreads * sizeof(std::uint32_t), host.data(), 0, nullptr,
                         nullptr);
  bool ok = verify(host, i8::kPreflightThreads, i8::kPreflightIters, 1);

  double secs = 0.0;
  int reps = 0;
  if (ok) {
    for (int w = 0; w < kWarmups; ++w) launch(T, i8::kIters, 1);
    const double t_once = launch(T, i8::kIters, 1);
    reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
    secs = launch(T, i8::kIters, reps);
    cl.clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, T * sizeof(std::uint32_t), host.data(), 0, nullptr, nullptr);
    constexpr std::uint32_t kSamples = 64;
    ok = verify(host, kSamples, i8::kIters, (T - 1) / (kSamples - 1));
  }
  cl.clReleaseKernel(kernel);
  cl.clReleaseProgram(program);
  if (!ok || secs <= 0.0) return false;

  r.path = "tensor(xmx " + std::to_string(kTensorRows) + "x" + std::to_string(sg) + "x" + std::to_string(kTensorK) +
           " s8->s32)";
  r.work = static_cast<std::uint64_t>(T / sg) * static_cast<std::uint64_t>(reps) * i8::kIters *
           (2ull * kTensorRows * sg * kTensorK);
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.score = score_giga(r.work, secs);
  r.correct = true;
  return true;
}

}  // namespace

RunResult run_gpu_int8_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.path = "simd(char4)";
  r.score_unit = "GOPS";

  const gpgpu::vendor::OpenClFns* cl_fns = gpgpu::vendor::opencl();
  if (!cl_fns) {
    r.error = "OpenCL loader unavailable";
    return r;
  }
  const gpgpu::vendor::OpenClFns& cl = *cl_fns;

  cl_device_id device = gpgpu::vendor::find_opencl_device(cl, setup.device);
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

  const std::uint32_t T = i8::thread_count(setup.device);

  // Intel XMX tensor path first; any failure falls through to the packed kernel.
  if (const std::uint32_t sg = intel_matrix_subgroup_size(cl, device)) {
    cl_mem tbuf = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, T * sizeof(std::uint32_t), nullptr, &err);
    const bool done = tbuf && run_tensor(cl, ctx, device, queue, tbuf, T, sg, r);
    if (tbuf) cl.clReleaseMemObject(tbuf);
    if (done) {
      cl.clReleaseCommandQueue(queue);
      cl.clReleaseContext(ctx);
      return r;
    }
  }

  std::string log;
  cl_program program = build_program_with_log(cl, ctx, device, kKernelSource, nullptr, log);
  if (!program) {
    r.error = "clBuildProgram failed: " + log;
    cl.clReleaseCommandQueue(queue);
    cl.clReleaseContext(ctx);
    return r;
  }
  cl_kernel kernel = cl.clCreateKernel(program, "int8_packed", &err);
  if (!kernel) {
    r.error = "clCreateKernel err=" + std::to_string(err);
    cl.clReleaseProgram(program);
    cl.clReleaseCommandQueue(queue);
    cl.clReleaseContext(ctx);
    return r;
  }

  cl_mem buf = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, T * sizeof(std::uint32_t), nullptr, &err);

  auto cleanup = [&]() {
    cl.clReleaseMemObject(buf);
    cl.clReleaseKernel(kernel);
    cl.clReleaseProgram(program);
    cl.clReleaseCommandQueue(queue);
    cl.clReleaseContext(ctx);
  };

  auto time_launches = [&](cl_uint n, cl_uint iters, int reps) -> double {
    cl.clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    cl.clSetKernelArg(kernel, 1, sizeof(cl_uint), &n);
    cl.clSetKernelArg(kernel, 2, sizeof(cl_uint), &iters);
    const std::size_t local = i8::kBlock;
    const std::size_t global = ((n + local - 1) / local) * local;
    cl_event first = nullptr, last = nullptr;
    for (int i = 0; i < reps; ++i) {
      cl_event ev = nullptr;
      cl.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &ev);
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
  cl.clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, i8::kPreflightThreads * sizeof(std::uint32_t), host.data(), 0, nullptr,
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

  cl.clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, T * sizeof(std::uint32_t), host.data(), 0, nullptr, nullptr);
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
