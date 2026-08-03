#pragma once

#include <blitz_task.h>
#include <gpu_harness.h>

#include <blitz_task.hpp>
#include <cstdint>
#include <gpgpu/setup.hpp>
#include <optional>
#include <vector>

namespace gpu_vram_write {

/**
 * @class GpuVramWrite
 * @brief Measures on-device VRAM streaming write bandwidth.
 *
 * streams stores into a large VRAM buffer, so the result reflects sustained
 * on-board write bandwidth. Reported in GB/s, higher is better.
 */
class CPP_TASK_DEMO_EXPORT GpuVramWrite : public blitz::Task {
 public:
  GpuVramWrite();
  ~GpuVramWrite() override;

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

}  // namespace gpu_vram_write

extern "C" {
CPP_TASK_DEMO_EXPORT BlitzTask* gpu_vram_write_new(void);
}
