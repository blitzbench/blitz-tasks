#include "gpu_matmul_fp32.hpp"

#include <gpu_harness.h>

#include <gpgpu/setup.hpp>

#include <bench/gemm/params.hpp>
#include <bench/result.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(GPU_MATMUL_FP32_HAVE_CUDA)
#include "runners/cuda/cuda_runner.hpp"
#endif
#if defined(GPU_MATMUL_FP32_HAVE_ROCM)
#include "runners/rocm/rocm_runner.hpp"
#endif
#if defined(GPU_MATMUL_FP32_HAVE_ONEAPI)
#include "runners/oneapi/oneapi_runner.hpp"
#endif
#if defined(GPU_MATMUL_FP32_HAVE_OPENCL)
#include "runners/opencl/opencl_runner.hpp"
#endif
#if defined(GPU_MATMUL_FP32_HAVE_VULKAN)
#include "runners/vulkan/vulkan_runner.hpp"
#endif
#if defined(GPU_MATMUL_FP32_HAVE_METAL)
#include "runners/metal/metal_runner.hpp"
#endif

extern "C" const char* GPU_MATMUL_FP32_INFO_JSON;

namespace gpu_matmul_fp32 {

namespace {

// Allocating and filling the operands dominates a short budget, so these tasks are given
// more room than the register-resident throughput kernels.
constexpr std::uint64_t DEFAULT_BUDGET_MS = 8000;

bench::RunResult missing(const char* name) {
  bench::RunResult r;
  r.supported = false;
  r.path = "n/a";
  r.error = std::string(name) + " runner not built into this binary";
  return r;
}

/**
 * @brief Route a (device, backend) setup to the matching compiled runner, guarded by the
 *        GPU_MATMUL_FP32_HAVE_<BACKEND> defines the linked runner libs export.
 *
 * @param setup
 * @param ctx
 * @param params
 * @return
 */
bench::RunResult dispatch_to(const gpgpu::Setup& setup, bench::gemm::ContextPtr& ctx,
                             const bench::gemm::RunParams& params) {
  switch (setup.backend.id()) {
    case gpgpu::BackendId::CUDA:
#if defined(GPU_MATMUL_FP32_HAVE_CUDA)
      return bench::run_gpu_matmul_fp32_cuda(setup, ctx, params);
#else
      return missing("CUDA");
#endif
    case gpgpu::BackendId::ROCm:
#if defined(GPU_MATMUL_FP32_HAVE_ROCM)
      return bench::run_gpu_matmul_fp32_rocm(setup, ctx, params);
#else
      return missing("ROCm");
#endif
    case gpgpu::BackendId::OneAPI:
#if defined(GPU_MATMUL_FP32_HAVE_ONEAPI)
      return bench::run_gpu_matmul_fp32_oneapi(setup, ctx, params);
#else
      return missing("OneAPI");
#endif
    case gpgpu::BackendId::OpenCL:
#if defined(GPU_MATMUL_FP32_HAVE_OPENCL)
      return bench::run_gpu_matmul_fp32_opencl(setup, ctx, params);
#else
      return missing("OpenCL");
#endif
    case gpgpu::BackendId::Vulkan:
#if defined(GPU_MATMUL_FP32_HAVE_VULKAN)
      return bench::run_gpu_matmul_fp32_vulkan(setup, ctx, params);
#else
      return missing("Vulkan");
#endif
    case gpgpu::BackendId::Metal:
#if defined(GPU_MATMUL_FP32_HAVE_METAL)
      return bench::run_gpu_matmul_fp32_metal(setup, ctx, params);
#else
      return missing("Metal");
#endif
  }
  return bench::RunResult{};
}

}  // namespace

GpuMatmulFp32::GpuMatmulFp32() : timeout_ms_(DEFAULT_BUDGET_MS) {}

GpuMatmulFp32::~GpuMatmulFp32() = default;

std::string_view GpuMatmulFp32::info_json() const noexcept { return GPU_MATMUL_FP32_INFO_JSON; }

blitz::Result GpuMatmulFp32::configure(const blitz::DataConfig& cfg) {
  data_size_bytes_ = cfg.data_size_bytes;
  iterations_ = cfg.iterations;
  seed_ = cfg.seed;
  // A fresh seed means different operands, so anything already on the device is stale.
  context_.reset();
  return BLITZ_OK;
}

blitz::Result GpuMatmulFp32::set_timeout(const std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result GpuMatmulFp32::setSetup(const gpgpu::Setup& setup) {
  gpgpu::Setup resolved;
  const blitz::Result r = bench::gpu::validate_setup(setup, resolved);
  if (r == BLITZ_OK) {
    setup_ = resolved;
    context_.reset();
  }
  return r;
}

std::vector<bench::gpu::ProbeResult> GpuMatmulFp32::probeSetups(const std::vector<gpgpu::Setup>& setups) const {
  // Probing runs every candidate, so it is pinned to the bottom of the size ladder: at
  // the top each candidate would allocate and fill gigabytes before the host has picked
  // one. The pinned run below sizes itself properly.
  bench::gemm::RunParams params;
  params.cap_bytes = bench::gemm::smallest_rung_cap(bench::gemm::Precision::Fp32);
  params.seed = static_cast<std::uint32_t>(seed_ ? seed_ : 1u);

  bench::gemm::ContextPtr probe_ctx;
  return bench::gpu::probe_setups(setups, [&](const gpgpu::Setup& s) {
    bench::RunResult r = dispatch_to(s, probe_ctx, params);
    // Each candidate owns the device it just built on; do not carry it to the next.
    probe_ctx.reset();
    return r;
  });
}

blitz::Result GpuMatmulFp32::run(const blitz::Callbacks& cb) {
  if (!setup_) {
    if (cb.on_error) {
      cb.on_error(BLITZ_ERR_INVALID_CONFIG, "no GPU setup selected; call setSetup() first");
    }
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return BLITZ_ERR_INVALID_CONFIG;
  }

  bench::gemm::RunParams params;
  params.cap_bytes = data_size_bytes_;
  params.seed = static_cast<std::uint32_t>(seed_ ? seed_ : 1u);
  params.pinned_reps = static_cast<int>(iterations_);

  return bench::gpu::run_gpu_benchmark(
      cb, timeout_ms_, "flops", "GFLOPS", BLITZ_DIR_HIGHER_IS_BETTER,
      [&](const gpgpu::Setup& s) { return dispatch_to(s, context_, params); }, *setup_);
}

}  // namespace gpu_matmul_fp32

extern "C" ::BlitzTask* gpu_matmul_fp32_new(void) {
  return blitz::make_c_task(std::make_unique<gpu_matmul_fp32::GpuMatmulFp32>());
}
