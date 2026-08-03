#pragma once

#include <blitz_task.h>
#include <gpu_harness.h>

#include <blitz_task.hpp>
#include <cstdint>
#include <gpgpu/setup.hpp>
#include <optional>
#include <vector>

namespace gpu_int8 {

/**
 * @class GpuInt8
 * @brief Measures 8-bit integer (int8) compute throughput on the GPU.
 *
 * runs a dense dot-product / multiply-accumulate kernel (tensor / dp4a where
 * available, packed int8 otherwise) with data kept on-chip, so the result
 * reflects raw int8 throughput rather than memory transfer. Reported in GIOPS,
 * higher is better.
 */
class CPP_TASK_DEMO_EXPORT GpuInt8 : public blitz::Task {
 public:
  GpuInt8();
  ~GpuInt8() override;

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

}  // namespace gpu_int8

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* gpu_int8_new(void);
}
