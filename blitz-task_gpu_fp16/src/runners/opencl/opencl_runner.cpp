/**
 * @file opencl_runner.cpp
 * @brief OpenCL fp16 throughput runner — Intel XMX tensor path with packed half8 FMA chains
 *        as the fallback.
 *
 * fp16 arithmetic in OpenCL C requires the cl_khr_fp16 extension. When the
 * device does not advertise it the row is a clean unsupported one.
 *
 * On devices with cl_intel_subgroup_matrix_multiply_accumulate (Intel Arc and newer)
 * the tensor kernel drives the XMX units through intel_sub_group_f16_f16_matrix_mad_k16:
 * one 8 x SG x 16 {f16,f16,f32} multiply-accumulate per call and sub-group, SG being the
 * device's minimum sub-group size (8 or 16, bench::intel_matrix_subgroup_size). A and B
 * are all 1/16, passed in as a kernel argument so no compiler can fold the products, so
 * every C element equals iters*16*(1/16)^2 exactly (fp16::tensor_out). Should the tensor
 * kernel fail to build or verify, the packed kernel runs instead.
 *
 *   tensor:  "tensor(xmx 8xSGx16 f16->f32)"
 *   packed:  "simd(half8 fma)"
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

// XMX tensor kernel, built with -DSG=8 or -DSG=16. Each call multiplies an 8x16 A by a
// 16xSG B: A packs two fp16 per int on SG8 and one per short on SG16, B packs a work item's
// 16-element column into an int8. Four independent accumulators keep the matrix pipe busy,
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
__kernel void fp16_xmx(__global float* out, const uint iters, const uint pair) {
    const a_t a = A_INIT(pair);
    const int8 b = (int8)((int)pair);
    float8 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (uint i = 0; i < iters / 4u; ++i) {
        c0 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c0);
        c1 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c1);
        c2 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c2);
        c3 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c3);
    }
    const float8 s = c0 + c1 + c2 + c3;
    out[get_global_id(0)] = (s.s0 + s.s1 + s.s2 + s.s3 + s.s4 + s.s5 + s.s6 + s.s7) / 8.0f;
}
)CL";

// Two fp16 1/16 (0x2C00) packed into one 32-bit word.
constexpr cl_uint kTensorPair = 0x2C002C00u;
constexpr int kTensorRows = 8;
constexpr int kTensorK = 16;

/**
 * @brief Runs the XMX tensor kernel on @p buf (T floats).
 *
 * @param cl OpenCL entry points
 * @param ctx context of @p device
 * @param device device to run on
 * @param queue profiling-enabled queue
 * @param buf output buffer of @p T floats
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
  cl_kernel kernel = cl.clCreateKernel(program, "fp16_xmx", &err);
  if (!kernel) {
    cl.clReleaseProgram(program);
    return false;
  }

  auto launch = [&](std::uint32_t n, cl_uint iters, int reps) {
    const cl_uint pair = kTensorPair;
    cl.clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    cl.clSetKernelArg(kernel, 1, sizeof(cl_uint), &iters);
    cl.clSetKernelArg(kernel, 2, sizeof(cl_uint), &pair);
    return cl_time_launches(cl, queue, kernel, n, fp16::kBlock, reps);
  };

  std::vector<float> host(T);
  bool ok = true;
  launch(fp16::kPreflightThreads, fp16::kTensorPreflightIters, 1);
  cl.clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, fp16::kPreflightThreads * sizeof(float), host.data(), 0, nullptr,
                         nullptr);
  for (std::uint32_t t = 0; t < fp16::kPreflightThreads && ok; ++t)
    ok = fp16::tensor_ok(host[t], fp16::kTensorPreflightIters, kTensorK);

  double secs = 0.0;
  int reps = 0;
  if (ok) {
    for (int w = 0; w < kWarmups; ++w) launch(T, fp16::kTensorIters, 1);
    const double t_once = launch(T, fp16::kTensorIters, 1);
    reps = calibrate_repeats(t_once, kComputeTargetSeconds, kComputeRepCap);
    secs = launch(T, fp16::kTensorIters, reps);
    cl.clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, T * sizeof(float), host.data(), 0, nullptr, nullptr);
    for (std::uint32_t s = 0; s < 64 && ok; ++s) {
      const std::uint32_t t = static_cast<std::uint32_t>(static_cast<std::uint64_t>(s) * (T - 1) / 63);
      ok = fp16::tensor_ok(host[t], fp16::kTensorIters, kTensorK);
    }
  }
  cl.clReleaseKernel(kernel);
  cl.clReleaseProgram(program);
  if (!ok || secs <= 0.0) return false;

  r.path = "tensor(xmx " + std::to_string(kTensorRows) + "x" + std::to_string(sg) + "x" + std::to_string(kTensorK) +
           " f16->f32)";
  r.work = fp16::tensor_flops(T / sg, fp16::kTensorIters, reps, kTensorRows, static_cast<int>(sg), kTensorK);
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.score = score_giga(r.work, secs);
  r.correct = true;
  return true;
}

}  // namespace

RunResult run_gpu_fp16_opencl(const gpgpu::Setup& setup) {
  RunResult r;
  r.score_unit = "GFLOPS";
  r.path = "unsupported(no cl_khr_fp16)";

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

  const std::string exts = cl_info_string(cl, device, CL_DEVICE_EXTENSIONS);
  if (exts.find("cl_khr_fp16") == std::string::npos) {
    r.supported = false;  // clean unsupported, not an error
    return r;
  }
  r.path = "simd(half8 fma)";

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

  const std::uint32_t T = fp16::thread_count(setup.device);

  // Intel XMX tensor path first; any failure falls through to the packed kernel.
  if (const std::uint32_t sg = intel_matrix_subgroup_size(cl, device)) {
    cl_mem tbuf = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, T * sizeof(float), nullptr, &err);
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
  cl_kernel kernel = cl.clCreateKernel(program, "fp16_fma", &err);
  if (!kernel) {
    r.error = "clCreateKernel err=" + std::to_string(err);
    cl.clReleaseProgram(program);
    cl.clReleaseCommandQueue(queue);
    cl.clReleaseContext(ctx);
    return r;
  }

  cl_mem buf = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, T * sizeof(float), nullptr, &err);

  auto cleanup = [&]() {
    cl.clReleaseMemObject(buf);
    cl.clReleaseKernel(kernel);
    cl.clReleaseProgram(program);
    cl.clReleaseCommandQueue(queue);
    cl.clReleaseContext(ctx);
  };

  auto time_launches = [&](cl_uint n, cl_uint iters, float mf, float af, int reps) -> double {
    cl.clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    cl.clSetKernelArg(kernel, 1, sizeof(cl_uint), &n);
    cl.clSetKernelArg(kernel, 2, sizeof(cl_uint), &iters);
    cl.clSetKernelArg(kernel, 3, sizeof(cl_float), &mf);
    cl.clSetKernelArg(kernel, 4, sizeof(cl_float), &af);
    const std::size_t local = fp16::kBlock;
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

  std::vector<float> host(T);

  // --- pre-flight exactness (m=1,a=1) ---
  time_launches(fp16::kPreflightThreads, fp16::kPreflightIters, fp16::kPreM, fp16::kPreA, 1);
  cl.clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, fp16::kPreflightThreads * sizeof(float), host.data(), 0, nullptr,
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

  cl.clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, T * sizeof(float), host.data(), 0, nullptr, nullptr);
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
