/**
 * @file opencl_runner.cpp
 * @brief OpenCL fp32 GEMM runner — shared-memory tiled multiply.
 *
 * OpenCL has no vendor BLAS this task can call, so it always reports the tiled path:
 *
 *   "tiled(local 16x16 fp32)"
 *
 * gemm_operand_value mirrors bench::gemm::operand_value, and the fill kernel writes A
 * and B on the device so no operand crosses the bus.
 *
 * Timing comes from command-queue profiling events spanning the whole rep batch.
 */

#include "opencl_runner.hpp"

#include <CL/cl.h>

#include <bench/cl_utils.hpp>
#include <bench/config.hpp>
#include <bench/gemm/params.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace bench {

namespace {

constexpr std::uint32_t kTile = 16u;

const char* kGemmSource = R"CLC(
uint gemm_operand_hash(uint row, uint col, uint which, uint seed) {
    uint h = row * 0x9E3779B1u ^ col * 0x85EBCA77u ^ which * 0xC2B2AE3Du ^ seed;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}

float gemm_operand_value(uint row, uint col, uint which, uint seed) {
    return ((float)(gemm_operand_hash(row, col, which, seed) & 7u) - 4.0f) * 0.25f;
}

__kernel void gemm_fill(__global float* a, __global float* b, uint n, uint seed) {
    const uint col = get_global_id(0);
    const uint row = get_global_id(1);
    if (row >= n || col >= n) return;
    const uint idx = row * n + col;
    a[idx] = gemm_operand_value(row, col, 0u, seed);
    b[idx] = gemm_operand_value(row, col, 1u, seed);
}

__kernel void gemm_tiled(__global const float* a, __global const float* b, __global float* c, uint n) {
    __local float sa[16][16];
    __local float sb[16][16];

    const uint col = get_global_id(0);
    const uint row = get_global_id(1);
    const uint lc = get_local_id(0);
    const uint lr = get_local_id(1);

    float acc = 0.0f;
    for (uint t = 0; t < n; t += 16u) {
        sa[lr][lc] = a[row * n + (t + lc)];
        sb[lr][lc] = b[(t + lr) * n + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < 16u; ++k) {
            acc += sa[lr][k] * sb[k][lc];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[row * n + col] = acc;
}
)CLC";

/**
 * @class OpenClGemmContext
 * @brief Context, queue, operand buffers and kernels for one (device, size) pair.
 */
struct OpenClGemmContext : gemm::Context {
  gpgpu::vendor::OpenClFns cl{};
  std::string device_id;
  std::uint32_t n{0};
  std::uint32_t seed{0};

  cl_context ctx{nullptr};
  cl_command_queue queue{nullptr};
  cl_program program{nullptr};
  cl_kernel fill_kernel{nullptr};
  cl_kernel gemm_kernel{nullptr};
  cl_mem a{nullptr}, b{nullptr}, c{nullptr};

  ~OpenClGemmContext() override {
    if (fill_kernel) cl.clReleaseKernel(fill_kernel);
    if (gemm_kernel) cl.clReleaseKernel(gemm_kernel);
    if (program) cl.clReleaseProgram(program);
    if (a) cl.clReleaseMemObject(a);
    if (b) cl.clReleaseMemObject(b);
    if (c) cl.clReleaseMemObject(c);
    if (queue) cl.clReleaseCommandQueue(queue);
    if (ctx) cl.clReleaseContext(ctx);
  }

  /**
   * @brief Enqueue @p reps multiplies, returning the device-side seconds they spanned.
   *
   * @param reps
   * @return 0 when the queue reported no usable profiling span.
   */
  double time_gemm(int reps) {
    const std::size_t global[2] = {n, n};
    const std::size_t local[2] = {kTile, kTile};
    cl_event first = nullptr, last = nullptr;
    for (int i = 0; i < reps; ++i) {
      cl_event ev = nullptr;
      if (cl.clEnqueueNDRangeKernel(queue, gemm_kernel, 2, nullptr, global, local, 0, nullptr, &ev) != CL_SUCCESS) {
        if (first) cl.clReleaseEvent(first);
        if (last && last != first) cl.clReleaseEvent(last);
        return 0.0;
      }
      if (i == 0) {
        first = ev;
      } else {
        if (last) cl.clReleaseEvent(last);
        last = ev;
      }
    }
    if (!last) last = first;
    cl.clFinish(queue);
    const double secs = cl_event_span(cl, first, last).count();
    if (first) cl.clReleaseEvent(first);
    if (last && last != first) cl.clReleaseEvent(last);
    return secs;
  }

  void fill_operands() {
    const std::size_t global[2] = {n, n};
    const std::size_t local[2] = {kTile, kTile};
    cl.clEnqueueNDRangeKernel(queue, fill_kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
    cl.clFinish(queue);
  }

  /**
   * @brief Copy the first gemm::kVerifyRows rows of C into @p host.
   *
   * @param host
   */
  void read_verify_rows(std::vector<float>& host) {
    const std::size_t bytes = static_cast<std::size_t>(gemm::kVerifyRows) * n * sizeof(float);
    cl.clEnqueueReadBuffer(queue, c, CL_TRUE, 0, bytes, host.data(), 0, nullptr, nullptr);
  }
};

/**
 * @brief Build a context for @p setup at problem size @p n.
 *
 * @param cl
 * @param setup
 * @param n
 * @param seed
 * @param out
 * @param error
 * @return false with @p error populated on any failure; @p out is then unusable.
 */
bool build_context(const gpgpu::vendor::OpenClFns& cl, const gpgpu::Setup& setup, std::uint32_t n,
                   std::uint32_t seed, OpenClGemmContext& out, std::string& error) {
  out.cl = cl;
  out.device_id = setup.device.id();
  out.n = n;
  out.seed = seed;

  cl_device_id dev = find_cl_device(cl, setup.device);
  if (!dev) {
    error = "no OpenCL device matched " + setup.device.name();
    return false;
  }

  cl_int err = CL_SUCCESS;
  out.ctx = cl.clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  if (!out.ctx) {
    error = "clCreateContext err=" + std::to_string(err);
    return false;
  }
  const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  out.queue = cl.clCreateCommandQueueWithProperties(out.ctx, dev, props, &err);
  if (!out.queue) {
    error = "clCreateCommandQueueWithProperties err=" + std::to_string(err);
    return false;
  }

  std::string log;
  out.program = build_program_with_log(cl, out.ctx, dev, kGemmSource, nullptr, log);
  if (!out.program) {
    error = "OpenCL build failed: " + log;
    return false;
  }
  out.fill_kernel = cl.clCreateKernel(out.program, "gemm_fill", &err);
  out.gemm_kernel = cl.clCreateKernel(out.program, "gemm_tiled", &err);
  if (!out.fill_kernel || !out.gemm_kernel) {
    error = "clCreateKernel err=" + std::to_string(err);
    return false;
  }

  const std::size_t elems = static_cast<std::size_t>(n) * n;
  out.a = cl.clCreateBuffer(out.ctx, CL_MEM_READ_WRITE, elems * 4, nullptr, &err);
  out.b = cl.clCreateBuffer(out.ctx, CL_MEM_READ_WRITE, elems * 4, nullptr, &err);
  out.c = cl.clCreateBuffer(out.ctx, CL_MEM_READ_WRITE, elems * 4, nullptr, &err);
  if (!out.a || !out.b || !out.c) {
    error = "operand allocation failed at n=" + std::to_string(n);
    return false;
  }

  cl.clSetKernelArg(out.fill_kernel, 0, sizeof(cl_mem), &out.a);
  cl.clSetKernelArg(out.fill_kernel, 1, sizeof(cl_mem), &out.b);
  cl.clSetKernelArg(out.fill_kernel, 2, sizeof(std::uint32_t), &out.n);
  cl.clSetKernelArg(out.fill_kernel, 3, sizeof(std::uint32_t), &out.seed);

  cl.clSetKernelArg(out.gemm_kernel, 0, sizeof(cl_mem), &out.a);
  cl.clSetKernelArg(out.gemm_kernel, 1, sizeof(cl_mem), &out.b);
  cl.clSetKernelArg(out.gemm_kernel, 2, sizeof(cl_mem), &out.c);
  cl.clSetKernelArg(out.gemm_kernel, 3, sizeof(std::uint32_t), &out.n);

  out.fill_operands();
  return true;
}

}  // namespace

RunResult run_gpu_matmul_fp32_opencl(const gpgpu::Setup& setup, gemm::ContextPtr& ctx,
                                     const gemm::RunParams& params) {
  RunResult r;
  r.score_unit = "GFLOPS";
  r.path = "unsupported(fp32 gemm)";

  const gpgpu::vendor::OpenClFns* cl_fns = gpgpu::vendor::opencl();
  if (!cl_fns) {
    r.error = "OpenCL loader unavailable";
    return r;
  }
  const gpgpu::vendor::OpenClFns& cl = *cl_fns;

  auto* gemm_ctx = dynamic_cast<OpenClGemmContext*>(ctx.get());
  if (gemm_ctx && (gemm_ctx->device_id != setup.device.id() || gemm_ctx->seed != params.seed)) gemm_ctx = nullptr;

  // Size the problem on the first round; later rounds reuse what this settled on. The
  // bottom rung is timed first and extrapolated, so a slow device is never asked to run
  // a multiply it cannot finish, and the ladder steps down again if allocation fails.
  if (!gemm_ctx) {
    const std::uint32_t memory_limit_n = gemm::choose_size(gemm::Precision::Fp32, setup.device, params.cap_bytes);
    std::string error;

    auto build_at = [&](std::uint32_t n) -> std::unique_ptr<OpenClGemmContext> {
      while (n) {
        auto fresh = std::make_unique<OpenClGemmContext>();
        if (build_context(cl, setup, n, params.seed, *fresh, error)) return fresh;
        n = gemm::step_down(n);
      }
      return nullptr;
    };

    auto probe_ctx = build_at(gemm::kLadder[0]);
    if (!probe_ctx) {
      r.error = error;
      return r;
    }
    const double t_probe = probe_ctx->time_gemm(1);
    const std::uint32_t target = gemm::largest_within_time(gemm::kLadder[0], t_probe, memory_limit_n);

    if (target > gemm::kLadder[0]) {
      probe_ctx.reset();
      probe_ctx = build_at(target);
      if (!probe_ctx) {
        r.error = error;
        return r;
      }
    }
    ctx = std::move(probe_ctx);
    gemm_ctx = static_cast<OpenClGemmContext*>(ctx.get());
  }

  r.path = "tiled(local 16x16 fp32)";

  for (int w = 0; w < kWarmups; ++w) gemm_ctx->time_gemm(1);
  const double t_once = gemm_ctx->time_gemm(1);
  if (t_once <= 0.0) {
    r.error = "queue reported no profiling span";
    return r;
  }
  const int reps =
      params.pinned_reps > 0 ? params.pinned_reps : calibrate_repeats(t_once, kGemmTargetSeconds, kGemmRepCap);
  const double secs = gemm_ctx->time_gemm(reps);
  if (secs <= 0.0) {
    r.error = "queue reported no profiling span";
    return r;
  }

  std::vector<float> host(static_cast<std::size_t>(gemm::kVerifyRows) * gemm_ctx->n);
  gemm_ctx->read_verify_rows(host);

  bool ok = true;
  for (std::uint32_t s = 0; s < gemm::kVerifySamples && ok; ++s) {
    const std::uint32_t row = gemm::sample_row(s);
    const std::uint32_t col = gemm::sample_col(s, gemm_ctx->n);
    const double got = host[static_cast<std::size_t>(row) * gemm_ctx->n + col];
    const double expected = gemm::reference_element(row, col, gemm_ctx->n, gemm_ctx->seed);
    if (!gemm::element_ok(got, expected, gemm_ctx->n)) {
      char b[160];
      std::snprintf(b, sizeof(b), "sample mismatch at (%u,%u): got=%g expected=%g", row, col, got, expected);
      r.error = b;
      ok = false;
    }
  }

  r.work = gemm::gemm_flops(gemm_ctx->n, reps);
  r.measured = std::chrono::duration<double>{secs};
  r.timings.kernel_compute = r.measured;
  r.score = score_giga(r.work, secs);
  r.correct = ok;
  r.info = {{"gemm_n", std::to_string(gemm_ctx->n)},
            {"reps", std::to_string(reps)},
            {"blas_library", "none"},
            {"math_mode", "fp32 throughout"}};
  return r;
}

}  // namespace bench
