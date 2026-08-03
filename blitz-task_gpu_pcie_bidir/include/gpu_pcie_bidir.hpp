#pragma once

#include <blitz_task.h>
#include <gpu_harness.h>

#include <blitz_task.hpp>
#include <cstdint>
#include <gpgpu/setup.hpp>
#include <optional>
#include <vector>

namespace gpu_pcie_bidir {

/**
 * @class GpuPcieBidir
 * @brief Measures bidirectional (PCIe) transfer bandwidth.
 *
 * runs concurrent host-to-device and device-to-host copies over the PCIe link,
 * so the result reflects aggregate full-duplex bandwidth. Reported in GB/s,
 * higher is better.
 */
class CPP_TASK_DEMO_EXPORT GpuPcieBidir : public blitz::Task {
 public:
  GpuPcieBidir();
  ~GpuPcieBidir() override;

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

}  // namespace gpu_pcie_bidir

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* gpu_pcie_bidir_new(void);
}
