#include "gpu_int8.hpp"

#include <gpu_harness.h>

#include <gpgpu/setup.hpp>

#include <bench/result.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(GPU_INT8_HAVE_CUDA)
#include "runners/cuda/cuda_runner.hpp"
#endif
#if defined(GPU_INT8_HAVE_ROCM)
#include "runners/rocm/rocm_runner.hpp"
#endif
#if defined(GPU_INT8_HAVE_ONEAPI)
#include "runners/oneapi/oneapi_runner.hpp"
#endif
#if defined(GPU_INT8_HAVE_OPENCL)
#include "runners/opencl/opencl_runner.hpp"
#endif
#if defined(GPU_INT8_HAVE_VULKAN)
#include "runners/vulkan/vulkan_runner.hpp"
#endif
#if defined(GPU_INT8_HAVE_METAL)
#include "runners/metal/metal_runner.hpp"
#endif

extern "C" const char* GPU_INT8_INFO_JSON;

namespace gpu_int8 {

namespace {

constexpr std::uint64_t DEFAULT_BUDGET_MS = 2000;

bench::RunResult missing(const char* name) {
  bench::RunResult r;
  r.supported = false;
  r.path = "n/a";
  r.error = std::string(name) + " runner not built into this binary";
  return r;
}

/**
 * @brief Route a (device, backend) setup to the matching compiled runner,
 *        guarded by the GPU_INT8_HAVE_<BACKEND> defines the linked runner libs export.
 *
 * @param setup
 * @return
 */
bench::RunResult dispatch(const gpgpu::Setup& setup) {
  switch (setup.backend.id()) {
    case gpgpu::BackendId::CUDA:
#if defined(GPU_INT8_HAVE_CUDA)
      return bench::run_gpu_int8_cuda(setup);
#else
      return missing("CUDA");
#endif
    case gpgpu::BackendId::ROCm:
#if defined(GPU_INT8_HAVE_ROCM)
      return bench::run_gpu_int8_rocm(setup);
#else
      return missing("ROCm");
#endif
    case gpgpu::BackendId::OneAPI:
#if defined(GPU_INT8_HAVE_ONEAPI)
      return bench::run_gpu_int8_oneapi(setup);
#else
      return missing("OneAPI");
#endif
    case gpgpu::BackendId::OpenCL:
#if defined(GPU_INT8_HAVE_OPENCL)
      return bench::run_gpu_int8_opencl(setup);
#else
      return missing("OpenCL");
#endif
    case gpgpu::BackendId::Vulkan:
#if defined(GPU_INT8_HAVE_VULKAN)
      return bench::run_gpu_int8_vulkan(setup);
#else
      return missing("Vulkan");
#endif
    case gpgpu::BackendId::Metal:
#if defined(GPU_INT8_HAVE_METAL)
      return bench::run_gpu_int8_metal(setup);
#else
      return missing("Metal");
#endif
  }
  return bench::RunResult{};
}

}  // namespace

GpuInt8::GpuInt8() : timeout_ms_(DEFAULT_BUDGET_MS) {}

GpuInt8::~GpuInt8() = default;

std::string_view GpuInt8::info_json() const noexcept { return GPU_INT8_INFO_JSON; }

blitz::Result GpuInt8::configure(const blitz::DataConfig& cfg) {
  data_size_bytes_ = cfg.data_size_bytes;
  iterations_ = cfg.iterations;
  seed_ = cfg.seed;
  return BLITZ_OK;
}

blitz::Result GpuInt8::set_timeout(const std::uint64_t timeout_ms) {
  timeout_ms_ = timeout_ms;
  return BLITZ_OK;
}

blitz::Result GpuInt8::setSetup(const gpgpu::Setup& setup) {
  gpgpu::Setup resolved;
  const blitz::Result r = bench::gpu::validate_setup(setup, resolved);
  if (r == BLITZ_OK) setup_ = resolved;
  return r;
}

std::vector<bench::gpu::ProbeResult> GpuInt8::probeSetups(
    const std::vector<gpgpu::Setup>& setups) const {
  return bench::gpu::probe_setups(setups, dispatch);
}

blitz::Result GpuInt8::run(const blitz::Callbacks& cb) {
  if (!setup_) {
    if (cb.on_error) {
      cb.on_error(BLITZ_ERR_INVALID_CONFIG, "no GPU setup selected; call setSetup() first");
    }
    if (cb.on_status) cb.on_status(BLITZ_STATUS_FAILED);
    return BLITZ_ERR_INVALID_CONFIG;
  }
  return bench::gpu::run_gpu_benchmark(cb, timeout_ms_, "iops", "GIOPS",
                                       BLITZ_DIR_HIGHER_IS_BETTER, dispatch, *setup_);
}

}  // namespace gpu_int8

extern "C" ::BlitzTask* gpu_int8_new(void) {
  return blitz::make_c_task(std::make_unique<gpu_int8::GpuInt8>());
}
