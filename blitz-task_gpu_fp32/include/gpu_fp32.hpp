#pragma once

#include <blitz_task.h>
#include <gpu_harness.h>

#include <blitz_task.hpp>
#include <cstdint>
#include <gpgpu/setup.hpp>
#include <optional>
#include <vector>

namespace gpu_fp32 {

/**
 * @class GpuFp32
 * @brief Measures single-precision (fp32) compute throughput on the GPU.
 *
 * runs a dense dependent fused-multiply-add kernel with data kept on-chip, so
 * the result reflects raw fp32 throughput rather than memory transfer. Reported
 * in GFLOPS, higher is better.
 */
class CPP_TASK_DEMO_EXPORT GpuFp32 : public blitz::Task {
 public:
  GpuFp32();
  ~GpuFp32() override;

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
};

}  // namespace gpu_fp32

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* gpu_fp32_new(void);
}
