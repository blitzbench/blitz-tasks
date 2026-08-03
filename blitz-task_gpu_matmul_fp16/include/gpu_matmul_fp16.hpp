#pragma once

#include <blitz_task.h>
#include <gpu_harness.h>

#include <bench/gemm/context.hpp>
#include <blitz_task.hpp>
#include <cstdint>
#include <gpgpu/setup.hpp>
#include <optional>
#include <vector>

namespace gpu_matmul_fp16 {

/**
 * @class GpuMatmulFp16
 * @brief Measures half-precision dense matrix multiplication on the GPU.
 *
 * Multiplies large square matrices through the vendor's own BLAS where one exists
 * (cuBLASLt, hipBLASLt, MPS) and through a cooperative-matrix or tiled kernel elsewhere,
 * so the score reflects the hardware together with the software stack shipped for it
 * rather than a peak instruction rate. Reported in GFLOPS, higher is better.
 */
class CPP_TASK_DEMO_EXPORT GpuMatmulFp16 : public blitz::Task {
 public:
  GpuMatmulFp16();
  ~GpuMatmulFp16() override;

  [[nodiscard]] std::string_view info_json() const noexcept override;
  blitz::Result configure(const blitz::DataConfig& cfg) override;
  blitz::Result set_timeout(std::uint64_t timeout_ms) override;
  blitz::Result run(const blitz::Callbacks& cb) override;

  /**
   * @brief Runs each setup once and returns the results as sorted list.
   *
   * @param setups
   * @return
   */
  [[nodiscard]] std::vector<bench::gpu::ProbeResult> probeSetups(const std::vector<gpgpu::Setup>& setups) const;

  /**
   * @brief Pin a setup so `run()` only runs for the provided setup.
   *
   * It is mandatory to explicitely provide a setup here.
   *
   * @param setup
   * @return
   */
  blitz::Result setSetup(const gpgpu::Setup& setup);

 private:
  std::uint64_t timeout_ms_;
  std::uint64_t data_size_bytes_{0};
  std::uint64_t iterations_{0};
  std::uint64_t seed_{0};
  std::optional<gpgpu::Setup> setup_{};
  // Device buffers and pipelines, kept alive so the harness's repeated rounds do not
  // pay for allocating and filling the operands again.
  bench::gemm::ContextPtr context_{};
};

}  // namespace gpu_matmul_fp16

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* gpu_matmul_fp16_new(void);
}
